//! Falcon-H1 AR backbone (MioTTS-0.1B): token ids → final hidden state + logits. Each layer runs a
//! Mamba-2 mixer ‖ GQA attention off a shared RMSNorm, summed into the residual, then a SwiGLU MLP.
//! Tied embeddings; `embed·embedding_multiplier` in, `logits·lm_head_multiplier` out.

mod attention;
mod mamba2;
mod rope;

use crate::config::FalconH1Config;
use crate::text::{EOS_IDS, SPEECH_BASE, SPEECH_COUNT};
use crate::weights::hf_file;
use attention::{Attention, AttnCache};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use mamba2::{Mamba2, MambaCache};
use rope::Rope;
use tts_core::{rms_norm, QLinear, Vb};

const AR_REPO: &str = "Aratako/MioTTS-0.1B";

/// SwiGLU MLP: `down(up(x) · silu(gate(x)))` (multipliers are 1.0).
struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}
impl Mlp {
    fn load(cfg: &FalconH1Config, vb: Vb) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: vb.pp("gate_proj").qlinear(h, i, false)?,
            up: vb.pp("up_proj").qlinear(h, i, false)?,
            down: vb.pp("down_proj").qlinear(i, h, false)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&(self.up.forward(x)? * self.gate.forward(x)?.silu()?)?)
    }
}

struct Layer {
    input_ln: Tensor, // RMSNorm weight
    mamba: Mamba2,
    attn: Attention,
    pre_ff_ln: Tensor,
    mlp: Mlp,
    eps: f64,
}
impl Layer {
    fn load(cfg: &FalconH1Config, vb: Vb) -> Result<Self> {
        Ok(Self {
            input_ln: vb.get(cfg.hidden_size, "input_layernorm.weight")?,
            mamba: Mamba2::load(cfg, vb.pp("mamba"))?,
            attn: Attention::load(cfg, vb.pp("self_attn"))?,
            pre_ff_ln: vb.get(cfg.hidden_size, "pre_ff_layernorm.weight")?,
            mlp: Mlp::load(cfg, vb.pp("feed_forward"))?,
            eps: cfg.rms_eps,
        })
    }

    /// Returns `(mamba_out, attn_out, layer_out)` — the parts the golden captures per layer.
    fn forward_parts(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let m = self.mamba.forward(&normed)?;
        let a = self.attn.forward(&normed, rope, mask)?;
        let mixed = (x + (&m + &a)?)?;
        let ff = self.mlp.forward(&rms_norm(&mixed, &self.pre_ff_ln, self.eps)?)?;
        let out = (&mixed + ff)?;
        Ok((m, a, out))
    }

    fn forward(&self, x: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        Ok(self.forward_parts(x, rope, mask)?.2)
    }

    /// Cache-aware layer step (Mamba ‖ attention off a shared RMSNorm, then SwiGLU). Same math as
    /// [`forward`] but the Mamba state + attention K/V carry across calls via `cache`; `pos` is the
    /// global position of the first of `x`'s tokens (for RoPE + the causal span).
    fn forward_cached(&self, x: &Tensor, rope: &Rope, cache: &mut LayerCache, pos: usize) -> Result<Tensor> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let m = self.mamba.forward_cached(&normed, &mut cache.mamba)?;
        let a = self.attn.forward_cached(&normed, rope, &mut cache.attn, pos)?;
        let mixed = (x + (&m + &a)?)?;
        let ff = self.mlp.forward(&rms_norm(&mixed, &self.pre_ff_ln, self.eps)?)?;
        &mixed + ff
    }
}

/// Per-layer decode state (Mamba SSM + conv history, attention K/V).
struct LayerCache {
    mamba: MambaCache,
    attn: AttnCache,
}

/// Whole-model incremental-decode state: one [`LayerCache`] per layer + the current sequence length
/// (the RoPE position / causal span for the next step).
pub struct DecodeCache {
    layers: Vec<LayerCache>,
    seqlen: usize,
}

/// Layer-0 intermediates, for stage-by-stage parity against the Python golden.
pub struct ArStages {
    pub layer0_mamba: Tensor,
    pub layer0_attn: Tensor,
    pub layer0_out: Tensor,
    pub hidden: Tensor,
    pub logits: Tensor,
}

/// The Falcon-H1 causal LM.
pub struct FalconH1 {
    embed: Tensor, // (vocab, hidden) — input embedding + the full-vocab tied LM head (reference path)
    /// Restricted **generation** head: the `Q8_0`-quantized embed rows for the only tokens valid
    /// during the assistant turn — speech tokens then EOS — so decode runs a ~6× smaller, half-read
    /// matmul than the full 78 336-vocab tied head. Rows `[speech_0..speech_{N-1}, EOS…]`.
    gen_head_q: QMatMul,
    layers: Vec<Layer>,
    final_ln: Tensor,
    rope: Rope,
    cfg: FalconH1Config,
    device: Device,
}

impl FalconH1 {
    pub fn from_hf(device: &Device) -> anyhow::Result<Self> {
        let path = hf_file(AR_REPO, "model.safetensors")?;
        Ok(Self::from_safetensors(path, device)?)
    }

    /// Load f32 weights from a `model.safetensors`. Numerically identical to the original path —
    /// `QLinear` over `QMatMul::Tensor` computes `x @ wᵀ` exactly like `candle_nn::Linear`.
    pub fn from_safetensors(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref().to_path_buf()], DType::F32, device)?
        };
        Self::from_vb(Vb::Full(vb), device)
    }

    /// Load a q8 GGUF (Linear weights `Q8_0`, the rest `F16`/`F32`) — the fast Intel-CPU path.
    pub fn from_gguf(path: impl AsRef<std::path::Path>, device: &Device) -> anyhow::Result<Self> {
        Ok(Self::from_vb(Vb::from_gguf(path, device)?, device)?)
    }

    /// Build from either weight source (full-precision safetensors or quantized GGUF).
    fn from_vb(vb: Vb, device: &Device) -> Result<Self> {
        let cfg = FalconH1Config::miotts_0_1b();
        let m = vb.pp("model");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Layer::load(&cfg, m.pp(format!("layers.{i}")))?);
        }
        let rope = Rope::new(cfg.head_dim, 4096, cfg.rope_theta, device)?;
        let embed = m.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?;
        // Build the restricted generation head: gather the embed rows for speech tokens then EOS and
        // quantize to Q8_0. The model only emits these during the assistant turn, so the per-token
        // head matmul shrinks from (·,512)×(512,78336) to (·,512)×(512,12802) on the AVX2 GEMV path.
        let mut gen_ids: Vec<u32> = (SPEECH_BASE..SPEECH_BASE + SPEECH_COUNT).collect();
        gen_ids.extend_from_slice(&EOS_IDS);
        let gen_idx = Tensor::from_vec(gen_ids.clone(), gen_ids.len(), device)?;
        let gen_rows = embed.to_dtype(DType::F32)?.index_select(&gen_idx, 0)?; // (SPEECH_COUNT+2, hidden)
        let gen_head_q = QMatMul::from_qtensor(QTensor::quantize(&gen_rows, GgmlDType::Q8_0)?)?;
        Ok(Self {
            embed,
            gen_head_q,
            layers,
            final_ln: m.get(cfg.hidden_size, "final_layernorm.weight")?,
            rope,
            cfg,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &FalconH1Config {
        &self.cfg
    }

    /// Embedding lookup, scaled by `embedding_multiplier`. `ids`: `(B, T)` int.
    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let flat = ids.to_dtype(DType::U32)?.flatten_all()?;
        let e = self.embed.index_select(&flat, 0)?.reshape((b, t, self.cfg.hidden_size))?;
        e.affine(self.cfg.embedding_multiplier, 0.0)
    }

    /// Tied LM head: `(hidden @ embedᵀ) · lm_head_multiplier`. `h`: `(B, T, hidden)`.
    fn lm_head(&self, h: &Tensor) -> Result<Tensor> {
        let (b, t, d) = h.dims3()?;
        let logits = h.reshape((b * t, d))?.matmul(&self.embed.t()?)?; // (B·T, vocab)
        logits.reshape((b, t, self.cfg.vocab_size))?.affine(self.cfg.lm_head_multiplier, 0.0)
    }

    /// Full forward: `ids (B,T)` → `(hidden (B,T,hidden), logits (B,T,vocab))`.
    pub fn forward(&self, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut h = self.embed(ids)?;
        let t = h.dim(1)?;
        let mask = causal_mask(t, &self.device)?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope, &mask)?;
        }
        let hidden = rms_norm(&h, &self.final_ln, self.cfg.rms_eps)?;
        let logits = self.lm_head(&hidden)?;
        Ok((hidden, logits))
    }

    /// Like [`forward`](Self::forward) but also returns layer-0 intermediates (for parity tests).
    pub fn forward_stages(&self, ids: &Tensor) -> Result<ArStages> {
        let h0 = self.embed(ids)?;
        let t = h0.dim(1)?;
        let mask = causal_mask(t, &self.device)?;
        let (m, a, out0) = self.layers[0].forward_parts(&h0, &self.rope, &mask)?;
        let mut h = out0.clone();
        for layer in &self.layers[1..] {
            h = layer.forward(&h, &self.rope, &mask)?;
        }
        let hidden = rms_norm(&h, &self.final_ln, self.cfg.rms_eps)?;
        let logits = self.lm_head(&hidden)?;
        Ok(ArStages { layer0_mamba: m, layer0_attn: a, layer0_out: out0, hidden, logits })
    }

    /// Restricted generation head on a single hidden vector `(hidden,)` → logits over the speech +
    /// EOS rows `(SPEECH_COUNT + EOS_IDS.len(),)`, `· lm_head_multiplier`. Decode the argmax/sampled
    /// index with [`crate::text::MioTokenizer::gen_index`]. The decode hot path uses this (not the
    /// full tied head) — a ~6× smaller Q8_0 GEMV that halves the per-token weight read.
    pub fn gen_head(&self, h: &Tensor) -> Result<Tensor> {
        let logits = self.gen_head_q.forward(&h.unsqueeze(0)?.contiguous()?)?; // (1, SPEECH_COUNT+2)
        logits.squeeze(0)?.affine(self.cfg.lm_head_multiplier, 0.0)
    }

    /// A fresh decode cache (zero Mamba state, empty attention K/V) for batch size 1.
    fn init_cache(&self) -> Result<DecodeCache> {
        let layers = self
            .layers
            .iter()
            .map(|l| Ok(LayerCache { mamba: l.mamba.init_cache(1, &self.device)?, attn: AttnCache::default() }))
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodeCache { layers, seqlen: 0 })
    }

    /// Prefill the prompt `ids` and return its last-position **hidden state** `(hidden,)` (post
    /// final-norm) plus the populated [`DecodeCache`] for [`decode_step`](Self::decode_step). O(T)
    /// for the whole prompt. Apply [`gen_head`](Self::gen_head) to sample the first speech token.
    pub fn prefill(&self, ids: &[u32]) -> Result<(Tensor, DecodeCache)> {
        let t = ids.len();
        let ids_t = Tensor::from_vec(ids.to_vec(), (1, t), &self.device)?;
        let mut cache = self.init_cache()?;
        let mut h = self.embed(&ids_t)?;
        for (layer, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_cached(&h, &self.rope, lc, 0)?;
        }
        cache.seqlen = t;
        let last = rms_norm(&h, &self.final_ln, self.cfg.rms_eps)?.i((0, t - 1))?; // (hidden,)
        Ok((last, cache))
    }

    /// Decode one `token` from `cache` (O(1) in sequence length): feed it, advance the caches, and
    /// return the next-position **hidden state** `(hidden,)`. Apply [`gen_head`](Self::gen_head) for
    /// the next-token logits.
    pub fn decode_step(&self, token: u32, cache: &mut DecodeCache) -> Result<Tensor> {
        let pos = cache.seqlen;
        let ids_t = Tensor::from_vec(vec![token], (1, 1), &self.device)?;
        let mut h = self.embed(&ids_t)?;
        for (layer, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = layer.forward_cached(&h, &self.rope, lc, pos)?;
        }
        cache.seqlen += 1;
        rms_norm(&h, &self.final_ln, self.cfg.rms_eps)?.i((0, 0)) // (hidden,)
    }
}

/// Additive causal mask `(1,1,T,T)`: `0` on/below the diagonal, `-inf` above.
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            v[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, (1, 1, t, t), device)
}
