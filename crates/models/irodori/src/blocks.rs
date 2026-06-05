//! Shared transformer building blocks for the encoders and the DiT.
//!
//! Loaded from the F32 DiT safetensors (or a q8 GGUF) via the shared [`Vb`]. Conventions match
//! `ref/mlx-audio/.../irodori_tts/model.py`: per-head q/k RMSNorm applied in `(B,S,H,Dh)` layout,
//! **interleaved** RoPE, a `sigmoid(gate)` on the attention output, and SwiGLU MLPs.

use candle_core::{Result, Tensor};
use tts_core::{rms_norm, sdpa, QLinear, RotaryEmbedding, Vb};

/// RMSNorm with a learned weight (channel `(D,)` or per-head `(H, Dh)` — both broadcast over the
/// leading axes since the norm is over the last dim).
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load<S: Into<candle_core::Shape>>(vb: Vb, shape: S, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(shape, "weight")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_norm(x, &self.weight, self.eps)
    }
}

/// SwiGLU MLP: `w2(silu(w1(x)) · w3(x))`.
pub struct SwiGlu {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl SwiGlu {
    pub fn load(vb: Vb, dim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w1: vb.pp("w1").qlinear(dim, hidden, false)?,
            w2: vb.pp("w2").qlinear(hidden, dim, false)?,
            w3: vb.pp("w3").qlinear(dim, hidden, false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }
}

/// Build a `(B,1,1,S)` additive attention mask from a `(B,S)` float key mask (1.0 valid / 0.0 pad):
/// valid → 0, pad → -1e9 (matching MLX's `_bool_to_additive_mask`).
pub fn additive_key_mask(mask: &Tensor) -> Result<Tensor> {
    let (b, s) = mask.dims2()?;
    ((mask - 1.0)? * 1e9)?.reshape((b, 1, 1, s))
}

/// Non-causal self-attention with full interleaved RoPE and a sigmoid output gate (encoders).
pub struct SelfAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    gate: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    pub fn load(vb: Vb, dim: usize, heads: usize, eps: f64) -> Result<Self> {
        let head_dim = dim / heads;
        Ok(Self {
            wq: vb.pp("wq").qlinear(dim, dim, false)?,
            wk: vb.pp("wk").qlinear(dim, dim, false)?,
            wv: vb.pp("wv").qlinear(dim, dim, false)?,
            wo: vb.pp("wo").qlinear(dim, dim, false)?,
            gate: vb.pp("gate").qlinear(dim, dim, false)?,
            q_norm: RmsNorm::load(vb.pp("q_norm"), (heads, head_dim), eps)?,
            k_norm: RmsNorm::load(vb.pp("k_norm"), (heads, head_dim), eps)?,
            heads,
            head_dim,
        })
    }

    /// `x`: `(B,S,dim)`. `mask`: optional additive `(B,1,1,S)`. RoPE positions start at 0.
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, rope: &RotaryEmbedding) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        // q/k RMSNorm in (B,S,H,Dh) layout — weight (H,Dh) broadcasts over the leading B,S.
        let q = self.q_norm.forward(&self.wq.forward(x)?.reshape((b, s, h, hd))?)?;
        let k = self.k_norm.forward(&self.wk.forward(x)?.reshape((b, s, h, hd))?)?;
        let v = self.wv.forward(x)?.reshape((b, s, h, hd))?;
        let gate = self.gate.forward(x)?; // (B,S,dim)

        // → (B,H,S,Dh), then RoPE (seq on axis 2).
        let q = rope.apply(&q.transpose(1, 2)?.contiguous()?, 0)?;
        let k = rope.apply(&k.transpose(1, 2)?.contiguous()?, 0)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (hd as f64).sqrt();
        let out = sdpa(&q, &k, &v, scale, mask)?; // (B,H,S,Dh)
        let out = out.transpose(1, 2)?.reshape((b, s, h * hd))?;
        self.wo.forward(&(out * candle_nn::ops::sigmoid(&gate)?)?)
    }
}

/// Pre-norm Transformer block: SelfAttention + SwiGLU (used in both encoders).
pub struct TextBlock {
    attention_norm: RmsNorm,
    attention: SelfAttention,
    mlp_norm: RmsNorm,
    mlp: SwiGlu,
}

impl TextBlock {
    pub fn load(vb: Vb, dim: usize, heads: usize, mlp_hidden: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            attention_norm: RmsNorm::load(vb.pp("attention_norm"), dim, eps)?,
            attention: SelfAttention::load(vb.pp("attention"), dim, heads, eps)?,
            mlp_norm: RmsNorm::load(vb.pp("mlp_norm"), dim, eps)?,
            mlp: SwiGlu::load(vb.pp("mlp"), dim, mlp_hidden)?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>, rope: &RotaryEmbedding) -> Result<Tensor> {
        let x = (x + self.attention.forward(&self.attention_norm.forward(x)?, mask, rope)?)?;
        let x = (&x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?)?;
        Ok(x)
    }
}
