//! Falcon-H1 GQA attention (8 query / 2 KV heads, rotate-half RoPE, causal). No biases;
//! `key_multiplier` is 1.0 so it's dropped. Full-sequence prefill (no KV cache yet — M3).

use super::rope::Rope;
use crate::config::FalconH1Config;
use candle_core::{Device, Result, Tensor};
use tts_core::{sdpa, QLinear, Vb};

/// Per-layer attention decode state: the accumulated K and V `(B, n_kv, T, head_dim)`.
#[derive(Default)]
pub struct AttnCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

pub struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn load(cfg: &FalconH1Config, vb: Vb) -> Result<Self> {
        let (h, kv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let dim = cfg.hidden_size;
        Ok(Self {
            q_proj: vb.pp("q_proj").qlinear(dim, h * hd, false)?,
            k_proj: vb.pp("k_proj").qlinear(dim, kv * hd, false)?,
            v_proj: vb.pp("v_proj").qlinear(dim, kv * hd, false)?,
            o_proj: vb.pp("o_proj").qlinear(h * hd, dim, false)?,
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

    /// Cache-aware attention: RoPE the `T` new tokens at `pos..pos+T`, append their K/V to `cache`,
    /// and attend against the full cached span. For `T=1` (decode) this is one query vs all keys; for
    /// the prompt (`pos=0`, empty cache) it reduces to the plain causal attention of [`forward`].
    pub fn forward_cached(&self, h: &Tensor, rope: &Rope, cache: &mut AttnCache, pos: usize) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let heads = |t_: Tensor, n: usize| -> Result<Tensor> {
            t_.reshape((b, t, n, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = rope.apply(&heads(self.q_proj.forward(h)?, self.n_heads)?, pos)?;
        let k = rope.apply(&heads(self.k_proj.forward(h)?, self.n_kv)?, pos)?;
        let v = heads(self.v_proj.forward(h)?, self.n_kv)?;

        let k = match cache.k.take() {
            Some(pk) => Tensor::cat(&[&pk, &k], 2)?,
            None => k,
        };
        let v = match cache.v.take() {
            Some(pv) => Tensor::cat(&[&pv, &v], 2)?,
            None => v,
        };
        cache.k = Some(k.clone());
        cache.v = Some(v.clone());
        let tc = k.dim(2)?;

        let n_rep = self.n_heads / self.n_kv;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;
        let mask = cached_mask(t, pos, tc, h.device())?;
        let out = sdpa(&q, &k, &v, self.scale, Some(&mask))?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Additive mask `(1,1,T,Tc)` for `T` queries at global positions `pos..pos+T` over `Tc` cached
/// keys: `0` where `key_pos ≤ query_pos`, `-inf` above. Causal for prefill; all-zero for a 1-token step.
fn cached_mask(t: usize, pos: usize, tc: usize, device: &Device) -> Result<Tensor> {
    let mut v = vec![0f32; t * tc];
    for i in 0..t {
        for j in (pos + i + 1)..tc {
            v[i * tc + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(v, (1, 1, t, tc), device)
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
