//! Voice encoder (M4): reference WAV → 128-d global embedding (the speaker "fingerprint" the codec
//! decoder conditions on). Port of MioCodec's global-branch encode path — a **truncated** WavLM-base+
//! (conv feature extractor + feature projection + positional conv + 2 of 12 transformer layers) whose
//! layer-1/2 features average into the [`GlobalEncoder`]. WavLM weights are bundled separately (they
//! live in torchaudio, not the codec file).
//!
//! Built stage by stage, each validated against `dump_golden_mio_encode.py`:
//! - [x] conv feature extractor → `conv_out`
//! - [x] feature projection → `feat_proj`
//! - [x] positional conv + 2 gated-rel-pos transformer layers → `tlayer0`/`tlayer1`
//! - [ ] GlobalEncoder (ConvNeXt + attentive-stats pool) → `global`
//! - [ ] resample 24 kHz→16 kHz + SSL padding; wire `encode_ref(wav)` into the engine

mod global;
mod resample;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, Module, VarBuilder};
use global::GlobalEncoder;

const WAVLM_EPS: f64 = 1e-5;
const NUM_HEADS: usize = 12;
const HEAD_DIM: usize = 64;
const NUM_BUCKETS: i64 = 320;
const MAX_DISTANCE: i64 = 800;

/// WavLM conv feature extractor: 7 strided convs (no bias). GroupNorm(512,512)+GELU on conv0, GELU
/// after every conv. Input `(B, samples)` → `(B, T, 512)` @ 50 Hz (hop 320).
struct FeatureExtractor {
    convs: Vec<Conv1d>,
    gn0: GroupNorm,
}

impl FeatureExtractor {
    fn load(vb: VarBuilder) -> Result<Self> {
        // (in, out, kernel, stride) for the 7 layers.
        let specs = [
            (1, 512, 10, 5),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 2, 2),
            (512, 512, 2, 2),
        ];
        let mut convs = Vec::with_capacity(specs.len());
        for (i, (ic, oc, k, s)) in specs.iter().enumerate() {
            let cfg = Conv1dConfig { stride: *s, ..Default::default() };
            convs.push(candle_nn::conv1d_no_bias(*ic, *oc, *k, cfg, vb.pp(format!("conv_layers.{i}.conv")))?);
        }
        let gn0 = candle_nn::group_norm(512, 512, WAVLM_EPS, vb.pp("conv_layers.0.layer_norm"))?;
        Ok(Self { convs, gn0 })
    }

    fn forward(&self, wav: &Tensor) -> Result<Tensor> {
        let mut x = wav.unsqueeze(1)?; // (B, 1, samples)
        for (i, c) in self.convs.iter().enumerate() {
            x = c.forward(&x)?;
            if i == 0 {
                x = self.gn0.forward(&x)?;
            }
            x = x.gelu_erf()?; // torchaudio nn.GELU() is the exact (erf) variant
        }
        x.transpose(1, 2)?.contiguous() // (B, T, 512)
    }
}

/// LayerNorm(512) + Linear(512→768).
struct FeatureProjection {
    ln: LayerNorm,
    proj: Linear,
}

impl FeatureProjection {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            ln: candle_nn::layer_norm(512, WAVLM_EPS, vb.pp("layer_norm"))?,
            proj: candle_nn::linear(512, 768, vb.pp("projection"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(&self.ln.forward(x)?)
    }
}

/// Convolutional positional embedding: weight-normed (pre-folded) grouped Conv1d (768→768, k128,
/// pad64, groups16), trim the last frame (even kernel), erf-GELU. Added to the input as a residual.
struct PosConv {
    conv: Conv1d,
}

impl PosConv {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 64, groups: 16, ..Default::default() };
        Ok(Self { conv: candle_nn::conv1d(768, 768, 128, cfg, vb.pp("conv"))? })
    }
    /// `(B, T, 768)` → `(B, T, 768)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(&x.transpose(1, 2)?.contiguous()?)?; // (B, 768, T+1)
        let t = x.dim(2)?;
        let x = x.narrow(2, 0, t - 1)?.gelu_erf()?; // trim last frame, erf-GELU
        x.transpose(1, 2)?.contiguous()
    }
}

/// Relative-position bucket index for every `(query, key)` pair, bidirectional (WavLM formula 5).
/// Pure integer arithmetic — identical for any sequence of length `t` — gathered into `rel_attn_embed`.
fn relative_position_buckets(t: usize) -> Vec<u32> {
    let nb = NUM_BUCKETS / 2; // 160 (half for each direction)
    let max_exact = nb / 2; // 80
    let log_denom = ((MAX_DISTANCE as f64) / (max_exact as f64)).ln();
    let mut out = vec![0u32; t * t];
    for i in 0..t {
        for j in 0..t {
            let rel = j as i64 - i as i64;
            let mut bucket = if rel > 0 { nb } else { 0 };
            let arel = rel.abs();
            bucket += if arel < max_exact {
                arel
            } else {
                let large = max_exact
                    + (((arel as f64) / (max_exact as f64)).ln() / log_denom * ((nb - max_exact) as f64)) as i64;
                large.min(nb - 1)
            };
            out[i * t + j] = bucket as u32;
        }
    }
    out
}

/// WavLM gated relative-position self-attention. `rel_attn_embed` exists only in layer 0; later layers
/// reuse its (ungated) position bias but re-gate it with their own query via `gru_rel_pos`.
struct WavLmAttention {
    in_proj: Linear,             // fused QKV (2304 = 3×768)
    out_proj: Linear,            // (768→768)
    rel_attn_embed: Option<Tensor>, // (320, 12) — layer 0 only
    gru_lin: Linear,             // (64→8)
    gru_const: Tensor,           // (1, 12, 1, 1)
}

impl WavLmAttention {
    fn load(vb: VarBuilder, has_rel: bool) -> Result<Self> {
        let a = vb.pp("attention"); // nn.MultiheadAttention lives under attention.attention
        let in_proj = Linear::new(
            a.get((3 * 768, 768), "in_proj_weight")?,
            Some(a.get(3 * 768, "in_proj_bias")?),
        );
        let out_proj = candle_nn::linear(768, 768, a.pp("out_proj"))?;
        let rel_attn_embed = if has_rel {
            Some(vb.get((NUM_BUCKETS as usize, NUM_HEADS), "rel_attn_embed.weight")?)
        } else {
            None
        };
        let gru_lin = candle_nn::linear(HEAD_DIM, 8, vb.pp("gru_rel_pos_linear"))?;
        let gru_const = vb.get((1, NUM_HEADS, 1, 1), "gru_rel_pos_const")?;
        Ok(Self { in_proj, out_proj, rel_attn_embed, gru_lin, gru_const })
    }

    /// Ungated position bias `(1, 12, T, T)` from `rel_attn_embed` (layer 0 only).
    fn position_bias(&self, t: usize, dev: &Device) -> Result<Tensor> {
        let embed = self.rel_attn_embed.as_ref().expect("position_bias requires layer-0 rel_attn_embed");
        let idx = Tensor::from_vec(relative_position_buckets(t), t * t, dev)?;
        embed
            .index_select(&idx, 0)? // (T*T, 12)
            .reshape((t, t, NUM_HEADS))?
            .permute((2, 0, 1))? // (12, T, T)
            .unsqueeze(0)? // (1, 12, T, T)
            .contiguous()
    }

    fn forward(&self, x: &Tensor, pos_bias: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        // Gate the position bias by the query (gru_rel_pos): linear(64→8) → sum 2×4 → sigmoid → split.
        let q_layer = x.reshape((b, t, NUM_HEADS, HEAD_DIM))?.permute((0, 2, 1, 3))?.contiguous()?; // (b,12,T,64)
        let g = self
            .gru_lin
            .forward(&q_layer)? // (b,12,T,8)
            .reshape((b, NUM_HEADS, t, 2, 4))?
            .sum(4)?; // (b,12,T,2)
        let g = candle_nn::ops::sigmoid(&g)?;
        let gate_a = g.narrow(3, 0, 1)?; // (b,12,T,1)
        let gate_b = g.narrow(3, 1, 1)?;
        // gate_a_1 = gate_a * (gate_b * const - 1) + 2
        let gate_a1 = gate_a
            .mul(&gate_b.broadcast_mul(&self.gru_const)?.affine(1.0, -1.0)?)?
            .affine(1.0, 2.0)?; // (b,12,T,1)
        let mask = gate_a1.broadcast_mul(pos_bias)?; // (b,12,T,T)

        // Fused QKV → split → heads.
        let qkv = self.in_proj.forward(x)?; // (b,T,2304)
        let to_heads = |t_: Tensor| -> Result<Tensor> {
            t_.reshape((b, t, NUM_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(qkv.narrow(2, 0, 768)?)?;
        let k = to_heads(qkv.narrow(2, 768, 768)?)?;
        let v = to_heads(qkv.narrow(2, 1536, 768)?)?;

        // SDPA with additive relative-position bias.
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?.add(&mask)?;
        let out = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?; // (b,12,T,64)
        let out = out.transpose(1, 2)?.reshape((b, t, NUM_HEADS * HEAD_DIM))?;
        self.out_proj.forward(&out)
    }
}

/// One WavLM transformer layer (`layer_norm_first=False`: post-attention norm, post-FFN norm).
struct EncoderLayer {
    attn: WavLmAttention,
    layer_norm: LayerNorm,
    ff_in: Linear,  // 768→3072
    ff_out: Linear, // 3072→768
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(vb: VarBuilder, has_rel: bool) -> Result<Self> {
        Ok(Self {
            attn: WavLmAttention::load(vb.pp("attention"), has_rel)?,
            layer_norm: candle_nn::layer_norm(768, WAVLM_EPS, vb.pp("layer_norm"))?,
            ff_in: candle_nn::linear(768, 3072, vb.pp("feed_forward.intermediate_dense"))?,
            ff_out: candle_nn::linear(3072, 768, vb.pp("feed_forward.output_dense"))?,
            final_layer_norm: candle_nn::layer_norm(768, WAVLM_EPS, vb.pp("final_layer_norm"))?,
        })
    }
    fn forward(&self, x: &Tensor, pos_bias: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(&(x + self.attn.forward(x, pos_bias)?)?)?; // residual + attn, then norm
        let ff = self.ff_out.forward(&self.ff_in.forward(&x)?.gelu_erf()?)?;
        self.final_layer_norm.forward(&(&x + ff)?)
    }
}

/// WavLM transformer stack (only the first 2 of 12 layers — all the GlobalEncoder needs). MioCodec's
/// config sets the *transformer's* `layer_norm_first=True` (so the encoder `layer_norm` runs inside
/// `_preprocess`, after pos-conv) while each *layer's* `layer_norm_first=False` (post-norm layers).
struct Transformer {
    pos_conv: PosConv,
    enc_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
}

impl Transformer {
    fn load(vb: VarBuilder, n_layers: usize) -> Result<Self> {
        let pos_conv = PosConv::load(vb.pp("pos_conv_embed"))?;
        let enc_norm = candle_nn::layer_norm(768, WAVLM_EPS, vb.pp("layer_norm"))?;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(EncoderLayer::load(vb.pp(format!("layers.{i}")), i == 0)?);
        }
        Ok(Self { pos_conv, enc_norm, layers })
    }

    /// `_preprocess`: residual pos-conv then the encoder `layer_norm` (transformer.layer_norm_first=True).
    fn preprocess(&self, x: &Tensor) -> Result<Tensor> {
        self.enc_norm.forward(&(x + self.pos_conv.forward(x)?)?)
    }

    /// Run the (truncated) stack, returning every per-layer output.
    fn layer_outputs(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut x = self.preprocess(x)?;
        let t = x.dim(1)?;
        let pos_bias = self.layers[0].attn.position_bias(t, x.device())?;
        let mut outs = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            x = layer.forward(&x, &pos_bias)?;
            outs.push(x.clone());
        }
        Ok(outs)
    }
}

/// The voice encoder: reference WAV (16 kHz, padded) → 128-d speaker embedding. The truncated WavLM
/// (conv + projection + 2 transformer layers) lives in the bundled `mio_wavlm.safetensors`; the
/// [`GlobalEncoder`] weights live in the codec checkpoint. Stage 6 (resample) pending.
pub struct VoiceEncoder {
    fe: FeatureExtractor,
    fp: FeatureProjection,
    transformer: Transformer,
    global: GlobalEncoder,
}

impl VoiceEncoder {
    /// Load the truncated WavLM weights from `wavlm_path` (the bundled `mio_wavlm.safetensors`) and
    /// the GlobalEncoder from `codec_path` (the MioCodec checkpoint, prefix `global_encoder.*`).
    pub fn from_safetensors(
        wavlm_path: impl AsRef<std::path::Path>,
        codec_path: impl AsRef<std::path::Path>,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[wavlm_path.as_ref().to_path_buf()], DType::F32, device)?
        };
        let cvb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[codec_path.as_ref().to_path_buf()], DType::F32, device)?
        };
        Ok(Self {
            fe: FeatureExtractor::load(vb.pp("feature_extractor"))?,
            fp: FeatureProjection::load(vb.pp("encoder.feature_projection"))?,
            transformer: Transformer::load(vb.pp("encoder.transformer"), 2)?,
            global: GlobalEncoder::load(cvb.pp("global_encoder"))?,
        })
    }

    /// Load with the WavLM bundle at `wavlm_path`; the GlobalEncoder comes from the HF codec checkpoint.
    pub fn from_hf(wavlm_path: impl AsRef<std::path::Path>, device: &Device) -> anyhow::Result<Self> {
        let codec = crate::weights::hf_file(crate::codec::CODEC_REPO, "model.safetensors")?;
        Ok(Self::from_safetensors(wavlm_path, codec, device)?)
    }

    /// Conv feature extractor output `(B, T, 512)` — stage-1 parity hook.
    pub fn conv_features(&self, wav: &Tensor) -> Result<Tensor> {
        self.fe.forward(wav)
    }

    /// Feature-projection output `(B, T, 768)` — stage-2 parity hook (transformer input).
    pub fn projected(&self, wav: &Tensor) -> Result<Tensor> {
        self.fp.forward(&self.fe.forward(wav)?)
    }

    /// Per-layer transformer outputs `(B, T, 768)` — stage-3/4 parity hook (`tlayer0`, `tlayer1`).
    /// These are the features the GlobalEncoder averages (no final encoder norm applied).
    pub fn transformer_layers(&self, wav: &Tensor) -> Result<Vec<Tensor>> {
        self.transformer.layer_outputs(&self.projected(wav)?)
    }

    /// Reference WAV (16 kHz, SSL-padded) `(B, samples)` → 128-d speaker embedding `(B, 128)`. Averages
    /// WavLM transformer layers 1 & 2 (`global_ssl_layers`) — not normalized — then the GlobalEncoder.
    pub fn encode_global(&self, wav: &Tensor) -> Result<Tensor> {
        let layers = self.transformer_layers(wav)?;
        let avg = ((&layers[0] + &layers[1])? * 0.5)?;
        self.global.forward(&avg)
    }

    /// Clone a voice from a **24 kHz mono** waveform `(samples,)` or `(1, samples)`: peak-normalize →
    /// SSL-pad → resample 24→16 kHz → WavLM → GlobalEncoder → 128-d embedding `(1, 128)`, the same
    /// tensor the codec decoder conditions on. Peak-normalization (matching MioCodec's `load_audio`)
    /// makes the result independent of input gain / WAV bit depth. Loading a file to mono is the
    /// caller's job ([`Mio::encode_ref_file`](crate::Mio::encode_ref_file) handles read+mono+resample).
    pub fn encode_ref(&self, wav24k: &Tensor) -> Result<Tensor> {
        let wav = if wav24k.rank() == 1 { wav24k.unsqueeze(0)? } else { wav24k.clone() };
        let peak = wav.abs()?.max_all()?.to_scalar::<f32>()? as f64 + 1e-8;
        let wav = wav.affine(1.0 / peak, 0.0)?; // normalize to [-1, 1]
        let padding = resample::ssl_padding(wav.dim(D::Minus1)?);
        let padded = wav.pad_with_zeros(D::Minus1, padding, padding)?;
        self.encode_global(&resample::resample_24k_to_16k(&padded)?)
    }
}
