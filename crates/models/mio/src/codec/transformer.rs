//! MioCodec transformer stack — a Llama-3-style block (interleaved RoPE, SwiGLU FFN, banded
//! non-causal windowed attention) with optional AdaLN-Zero conditioning. Port of
//! `Aratako/MioCodec`'s `module/transformer.py` (decode path; no KV cache — full-sequence).

use crate::config::TfConfig;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use tts_core::{sdpa, RotaryEmbedding};

/// SwiGLU FFN hidden size, matching Llama's `FeedForward` (`multiple_of`-rounded `2·(4·dim)/3`).
fn ffn_hidden(dim: usize, multiple_of: usize) -> usize {
    let h = 2 * (4 * dim) / 3;
    multiple_of * h.div_ceil(multiple_of)
}

/// Plain (affine) LayerNorm over the last dim — `candle_nn::LayerNorm`-equivalent, loaded from
/// `{prefix}.weight` / `{prefix}.bias`.
struct LayerNorm {
    inner: candle_nn::LayerNorm,
}
impl LayerNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { inner: candle_nn::layer_norm(dim, eps, vb)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// LayerNorm with `elementwise_affine=False` (no learnable params).
fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    xc.broadcast_div(&(var + eps)?.sqrt()?)
}

/// AdaLN-Zero: `LN(no-affine)` then `cond → SiLU → Linear → (shift, scale[, gate])`,
/// `x_norm·(1+scale)+shift`. Port of `module/adaln_zero.py`.
struct AdaLnZero {
    proj: Linear, // condition_proj.1 : cond_dim -> (3 or 2)·dim
    dim: usize,
    eps: f64,
    gate: bool,
}
impl AdaLnZero {
    fn load(dim: usize, cond_dim: usize, eps: f64, gate: bool, vb: VarBuilder) -> Result<Self> {
        let out = if gate { 3 * dim } else { 2 * dim };
        let proj = candle_nn::linear(cond_dim, out, vb.pp("condition_proj.1"))?;
        Ok(Self { proj, dim, eps, gate })
    }
    /// `x`: `(B, L, dim)`, `cond`: `(B, 1, cond_dim)`. Returns `(modulated, gate?)`.
    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let x_norm = layer_norm_no_affine(x, self.eps)?;
        let params = self.proj.forward(&cond.silu()?)?; // (B,1,*)
        let shift = params.narrow(D::Minus1, 0, self.dim)?;
        let scale = params.narrow(D::Minus1, self.dim, self.dim)?;
        let modulated = x_norm
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        let gate = if self.gate {
            Some(params.narrow(D::Minus1, 2 * self.dim, self.dim)?)
        } else {
            None
        };
        Ok((modulated, gate))
    }
}

/// Either a plain LayerNorm or AdaLN-Zero, selected per `use_adaln_zero`.
enum Norm {
    Ln(LayerNorm),
    Ada(AdaLnZero),
}
impl Norm {
    /// Returns `(normed, gate?)`. `cond` is required iff this is the AdaLN-Zero variant.
    fn forward(&self, x: &Tensor, cond: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        match self {
            Norm::Ln(ln) => Ok((ln.forward(x)?, None)),
            Norm::Ada(a) => a.forward(x, cond.expect("AdaLN-Zero requires a condition")),
        }
    }
}

/// Multi-head attention with interleaved RoPE and an optional additive (banded) mask. No bias.
struct Attention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
    scale: f64,
}
impl Attention {
    fn load(dim: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / n_heads;
        Ok(Self {
            wq: candle_nn::linear_no_bias(dim, dim, vb.pp("wq"))?,
            wk: candle_nn::linear_no_bias(dim, dim, vb.pp("wk"))?,
            wv: candle_nn::linear_no_bias(dim, dim, vb.pp("wv"))?,
            wo: candle_nn::linear_no_bias(dim, dim, vb.pp("wo"))?,
            n_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, rope: &RotaryEmbedding, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let to_heads = |t_: Tensor| -> Result<Tensor> {
            t_.reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)? // (B, H, T, Dh)
                .contiguous()
        };
        let q = rope.apply(&to_heads(self.wq.forward(x)?)?, 0)?;
        let k = rope.apply(&to_heads(self.wk.forward(x)?)?, 0)?;
        let v = to_heads(self.wv.forward(x)?)?;
        let out = sdpa(&q, &k, &v, self.scale, mask)?; // (B, H, T, Dh)
        let out = out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.wo.forward(&out)
    }
}

/// SwiGLU feed-forward: `w2(silu(w1 x) ⊙ w3 x)`.
struct FeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl FeedForward {
    fn load(dim: usize, multiple_of: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = ffn_hidden(dim, multiple_of);
        Ok(Self {
            w1: candle_nn::linear_no_bias(dim, hidden, vb.pp("w1"))?,
            w2: candle_nn::linear_no_bias(hidden, dim, vb.pp("w2"))?,
            w3: candle_nn::linear_no_bias(dim, hidden, vb.pp("w3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.w1.forward(x)?.silu()?;
        let b = self.w3.forward(x)?;
        self.w2.forward(&(a * b)?)
    }
}

struct Block {
    attention_norm: Norm,
    attention: Attention,
    ffn_norm: Norm,
    feed_forward: FeedForward,
}
impl Block {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        mask: Option<&Tensor>,
        cond: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (attn_normed, attn_gate) = self.attention_norm.forward(x, cond)?;
        let attn_out = self.attention.forward(&attn_normed, rope, mask)?;
        let h = match &attn_gate {
            Some(g) => (x + attn_out.broadcast_mul(g)?)?,
            None => (x + attn_out)?,
        };
        let (ffn_normed, ffn_gate) = self.ffn_norm.forward(&h, cond)?;
        let ffn_out = self.feed_forward.forward(&ffn_normed)?;
        match &ffn_gate {
            Some(g) => &h + ffn_out.broadcast_mul(g)?,
            None => &h + ffn_out,
        }
    }
}

/// RoPE table length (frames) for the codec transformers. The AR emits up to `max_tokens` speech
/// tokens (700 by default) and the codec runs **2 frames/token**, so the config's `max_seq_len*2`
/// (=1024 ≈ 512 tokens) overruns on long clips — `decode_speech` then panics in `rope.apply`. RoPE
/// values are a pure function of position, so a larger table is bit-identical for existing
/// positions; size it for ~2048 tokens (4096 frames, ≈82 s @ 24 kHz) to cover any realistic clip.
const ROPE_MAX_FRAMES: usize = 4096;

/// A MioCodec transformer stack (`wave_prenet` / `wave_decoder`).
pub struct Transformer {
    layers: Vec<Block>,
    final_norm: Norm,
    output_proj: Option<Linear>,
    rope: RotaryEmbedding,
    window_per_side: Option<usize>,
}

impl Transformer {
    pub fn load(vb: VarBuilder, cfg: &TfConfig, device: &Device) -> Result<Self> {
        let head_dim = cfg.dim / cfg.n_heads;
        let rope = RotaryEmbedding::new(head_dim, ROPE_MAX_FRAMES.max(cfg.max_seq_len * 2), cfg.rope_theta, device)?;
        let mk_norm = |vb: VarBuilder| -> Result<Norm> {
            if cfg.use_adaln_zero {
                let cd = cfg.adanorm_condition_dim.expect("adaln needs condition dim");
                Ok(Norm::Ada(AdaLnZero::load(cfg.dim, cd, cfg.norm_eps, true, vb)?))
            } else {
                Ok(Norm::Ln(LayerNorm::load(cfg.dim, cfg.norm_eps, vb)?))
            }
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lvb = vb.pp(format!("layers.{i}"));
            layers.push(Block {
                attention_norm: mk_norm(lvb.pp("attention_norm"))?,
                attention: Attention::load(cfg.dim, cfg.n_heads, lvb.pp("attention"))?,
                ffn_norm: mk_norm(lvb.pp("ffn_norm"))?,
                feed_forward: FeedForward::load(cfg.dim, cfg.multiple_of, lvb.pp("feed_forward"))?,
            });
        }
        // Final norm: AdaLN-Zero here is the no-gate (2·dim) variant.
        let final_norm = if cfg.use_adaln_zero {
            let cd = cfg.adanorm_condition_dim.expect("adaln needs condition dim");
            Norm::Ada(AdaLnZero::load(cfg.dim, cd, cfg.norm_eps, false, vb.pp("norm"))?)
        } else {
            Norm::Ln(LayerNorm::load(cfg.dim, cfg.norm_eps, vb.pp("norm"))?)
        };
        let output_proj = match cfg.output_dim {
            Some(od) => Some(candle_nn::linear(cfg.dim, od, vb.pp("output_proj"))?),
            None => None,
        };
        Ok(Self {
            layers,
            final_norm,
            output_proj,
            rope,
            window_per_side: cfg.window_size.map(|w| w / 2),
        })
    }

    /// `x`: `(B, T, dim)`; `cond`: `(B, 1, cond_dim)` for AdaLN-Zero stacks (else `None`).
    pub fn forward(&self, x: &Tensor, cond: Option<&Tensor>) -> Result<Tensor> {
        let t = x.dim(1)?;
        let mask = match self.window_per_side {
            Some(w) => Some(banded_mask(t, w, x.device())?),
            None => None,
        };
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x, &self.rope, mask.as_ref(), cond)?;
        }
        let (x, _) = self.final_norm.forward(&x, cond)?;
        match &self.output_proj {
            Some(p) => p.forward(&x),
            None => Ok(x),
        }
    }
}

/// Additive non-causal banded mask `(1,1,T,T)`: `0` where `|i-j| ≤ window_per_side`, else `-inf`.
fn banded_mask(t: usize, window_per_side: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if i.abs_diff(j) > window_per_side {
                v[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(v, (1, 1, t, t), device)
}
