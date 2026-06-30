//! Falcon-H1 GQA attention (8 query / 2 KV heads, rotate-half RoPE, causal). No biases;
//! `key_multiplier` is 1.0 so it's dropped. [`Attention::forward`] is the full-sequence reference
//! (candle SDPA, golden-validated); [`Attention::forward_cached`] is the incremental decode path,
//! hand-rolled over a flat K/V history (no SDPA / `repeat_kv` / per-step `cat`).

use super::rope::Rope;
use crate::config::FalconH1Config;
use candle_core::{Result, Tensor};
use tts_core::{sdpa, QLinear, Vb};

/// Per-layer attention decode state: the K/V history as flat f32, time-major
/// `k[(j·n_kv + kv)·head_dim + d]` (so appending a timestep is a cheap `extend`, no full-cache copy).
#[derive(Default)]
pub struct AttnCache {
    k: Vec<f32>,
    v: Vec<f32>,
    tc: usize, // cached timesteps
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

    /// `h`: `(B, T, dim)`, `mask`: additive causal `(1,1,T,T)`. Reference / prefill-golden path.
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

    /// Cache-aware attention, hand-rolled over the flat K/V history — no candle SDPA, no `repeat_kv`
    /// expansion, no per-step `cat` of the whole cache. RoPE the `T` new tokens at `pos..pos+T`,
    /// append their K/V (O(T)), then for each query head attend over its GQA KV head's cached keys
    /// with a causal span. Matches [`forward`] to the f32-accumulation floor (`ar_tests`). Batch-1.
    pub fn forward_cached(&self, h: &Tensor, rope: &Rope, cache: &mut AttnCache, pos: usize) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        assert_eq!(b, 1, "forward_cached is batch-1 (incremental decode)");
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv);
        let n_rep = nh / nkv;
        let heads = |t_: Tensor, n: usize| -> Result<Tensor> {
            t_.reshape((b, t, n, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = rope.apply(&heads(self.q_proj.forward(h)?, nh)?, pos)?; // (1, nh, t, hd)
        let k = rope.apply(&heads(self.k_proj.forward(h)?, nkv)?, pos)?; // (1, nkv, t, hd)
        let v = heads(self.v_proj.forward(h)?, nkv)?;

        // Append the T new timesteps to the flat history (time-major `[i][kv][d]`), O(T).
        let k_tm = k.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let v_tm = v.transpose(1, 2)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        cache.k.extend_from_slice(&k_tm);
        cache.v.extend_from_slice(&v_tm);
        cache.tc += t;

        let qv = q.flatten_all()?.to_vec1::<f32>()?; // (nh·t·hd), `[h][i][d]`
        let scale = self.scale as f32;
        let mut out = vec![0f32; t * nh * hd]; // `[i][h][d]` → reshape (1, t, nh·hd)
        let mut scores = vec![0f32; cache.tc];
        for i in 0..t {
            let valid = pos + i + 1; // query at global pos+i attends cached keys 0..=pos+i
            for hh in 0..nh {
                let kv = hh / n_rep;
                let qbase = (hh * t + i) * hd;
                // scores[j] = scale · q·k_j , tracking the max for a stable softmax
                let mut maxs = f32::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate().take(valid) {
                    let kbase = (j * nkv + kv) * hd;
                    let mut s = 0f32;
                    for d in 0..hd {
                        s += qv[qbase + d] * cache.k[kbase + d];
                    }
                    s *= scale;
                    *sj = s;
                    if s > maxs {
                        maxs = s;
                    }
                }
                // softmax over 0..valid
                let mut sum = 0f32;
                for sj in scores.iter_mut().take(valid) {
                    let e = (*sj - maxs).exp();
                    *sj = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                // out[h] = Σ_j softmax_j · v_j
                let obase = (i * nh + hh) * hd;
                for d in 0..hd {
                    let mut acc = 0f32;
                    for (j, &sj) in scores.iter().enumerate().take(valid) {
                        acc += sj * cache.v[(j * nkv + kv) * hd + d];
                    }
                    out[obase + d] = acc * inv;
                }
            }
        }
        let out = Tensor::from_vec(out, (b, t, nh * hd), h.device())?;
        self.o_proj.forward(&out)
    }
}

/// Expand `(B, n_kv, T, Dh)` → `(B, n_kv·n_rep, T, Dh)` (GQA head repetition; the reference path).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, n_rep, t, d))?
        .reshape((b, kv * n_rep, t, d))
}
