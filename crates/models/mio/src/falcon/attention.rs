//! Falcon-H1 GQA attention (8 query / 2 KV heads, rotate-half RoPE, causal). No biases;
//! `key_multiplier` is 1.0 so it's dropped. Full-sequence prefill (no KV cache yet — M3).

use super::rope::Rope;
use crate::config::FalconH1Config;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tts_core::sdpa;

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn load(cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
        let (h, kv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let dim = cfg.hidden_size;
        Ok(Self {
            q_proj: candle_nn::linear_no_bias(dim, h * hd, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(dim, kv * hd, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear_no_bias(dim, kv * hd, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(h * hd, dim, vb.pp("o_proj"))?,
            n_heads: h,
            n_kv: kv,
            head_dim: hd,
            scale: (hd as f64).powf(-0.5),
        })
    }

    /// `h`: `(B, T, dim)`, `mask`: additive causal `(1,1,T,T)`.
    pub fn forward(&self, h: &Tensor, rope: &Rope, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let heads = |t_: Tensor, n: usize| -> Result<Tensor> {
            t_.reshape((b, t, n, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = rope.apply(&heads(self.q_proj.forward(h)?, self.n_heads)?, 0)?;
        let k = rope.apply(&heads(self.k_proj.forward(h)?, self.n_kv)?, 0)?;
        let v = heads(self.v_proj.forward(h)?, self.n_kv)?;

        let n_rep = self.n_heads / self.n_kv;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let out = sdpa(&q, &k, &v, self.scale, Some(mask))?; // (B, H, T, Dh)
        let out = out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Expand `(B, n_kv, T, Dh)` → `(B, n_kv·n_rep, T, Dh)` (GQA head repetition).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, n_rep, t, d))?
        .reshape((b, kv * n_rep, t, d))
}
