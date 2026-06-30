//! Falcon-H1 AR backbone (MioTTS-0.1B): token ids → final hidden state + logits. Each layer runs a
//! Mamba-2 mixer ‖ GQA attention off a shared RMSNorm, summed into the residual, then a SwiGLU MLP.
//! Tied embeddings; `embed·embedding_multiplier` in, `logits·lm_head_multiplier` out.

mod attention;
mod mamba2;
mod rope;

use crate::config::FalconH1Config;
use crate::weights::hf_file;
use attention::Attention;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use mamba2::Mamba2;
use rope::Rope;
use tts_core::rms_norm;

const AR_REPO: &str = "Aratako/MioTTS-0.1B";

/// SwiGLU MLP: `down(up(x) · silu(gate(x)))` (multipliers are 1.0).
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn load(cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up: candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
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
    fn load(cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
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
    embed: Tensor, // (vocab, hidden) — also the tied LM head
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

    pub fn from_safetensors(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let cfg = FalconH1Config::miotts_0_1b();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref().to_path_buf()], DType::F32, device)?
        };
        let m = vb.pp("model");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Layer::load(&cfg, m.pp(format!("layers.{i}")))?);
        }
        let rope = Rope::new(cfg.head_dim, 4096, cfg.rope_theta, device)?;
        Ok(Self {
            embed: m.get((cfg.vocab_size, cfg.hidden_size), "embed_tokens.weight")?,
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
