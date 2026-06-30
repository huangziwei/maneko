//! Falcon-H1 Mamba-2 mixer (the `torch_forward` / naive path, which the CPU golden runs).
//!
//! `in_proj → [gate, xBC, dt]`; depthwise causal conv1d + SiLU on `xBC`; split `→ [x, B, C]`; the
//! selective-scan recurrence `state = state·exp(dt·A) + (dt·x)·B`, `y = state·C + D·x`
//! (`A = −exp(A_log)`); gate (no RMSNorm here, `mamba_rms_norm=false`) `y·silu(gate)`; `out_proj`.
//! The recurrence is run sequentially in f32 — it reproduces both the chunked prefill and the
//! cached single-step decode of the reference.
//!
//! Two entry points share the same scan ([`Mamba2::scan`]): the full-sequence [`Mamba2::forward`]
//! (golden-validated, state starts at zero) and the cache-aware [`Mamba2::forward_cached`] used by
//! incremental decode (state + the last `d_conv-1` conv inputs carry across steps via [`MambaCache`]).

use crate::config::FalconH1Config;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
use tts_core::{QLinear, Vb};

/// `softplus` matching `torch.nn.functional.softplus` (beta 1, threshold 20).
fn softplus(z: f32) -> f32 {
    if z > 20.0 { z } else { (z.exp() + 1.0).ln() }
}

/// Per-layer Mamba decode state: the SSM recurrent state and the rolling conv-input window.
pub struct MambaCache {
    /// SSM state, flat `(B · n_heads · head_dim · d_state)`, carried across steps.
    ssm: Vec<f32>,
    /// The last `d_conv-1` conv inputs `(B, conv_dim, d_conv-1)` (the depthwise-conv history).
    conv: Tensor,
}

pub struct Mamba2 {
    in_proj: QLinear,
    conv1d: Conv1d,    // padding d_conv-1 — the full-sequence (golden / prefill-reference) path
    conv_w: Vec<f32>,  // flat depthwise kernel `(conv_dim · d_conv)` for the hand-rolled cached path
    conv_b: Vec<f32>,  // conv bias `(conv_dim,)`
    out_proj: QLinear,
    a_log: Vec<f32>,  // (n_heads,)
    d: Vec<f32>,      // (n_heads,)
    dt_bias: Vec<f32>, // (n_heads,)
    d_ssm: usize,
    conv_dim: usize,
    d_conv: usize,
    n_heads: usize,   // mamba heads (24)
    head_dim: usize,  // 32
    state: usize,     // d_state (64)
    g_state: usize,   // n_groups · d_state (64)
}

impl Mamba2 {
    pub fn load(cfg: &FalconH1Config, vb: Vb) -> Result<Self> {
        let conv_dim = cfg.conv_dim();
        let proj_size = cfg.mamba_d_ssm + conv_dim + cfg.mamba_n_heads;
        let conv_cfg = Conv1dConfig {
            padding: cfg.mamba_d_conv - 1,
            groups: conv_dim,
            ..Default::default()
        };
        let conv1d = vb.pp("conv1d").conv1d(conv_dim, conv_dim, cfg.mamba_d_conv, true, conv_cfg)?;
        // Flat f32 copies of the depthwise kernel for the cached path's hand-rolled conv — candle's
        // `Conv1d` routes a depthwise (groups=channels) conv through per-group BLAS GEMM, which
        // dominates single-token decode (~40% of generation in profiling). The reference `forward`
        // (prefill / golden) path still uses `conv1d` above. `(conv_dim, 1, d_conv)` flattens
        // row-major to `c·d_conv + k`.
        let conv_w = conv1d.weight().to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let conv_b = conv1d.bias().expect("conv1d has bias").to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(Self {
            in_proj: vb.pp("in_proj").qlinear(cfg.hidden_size, proj_size, false)?,
            conv1d,
            conv_w,
            conv_b,
            out_proj: vb.pp("out_proj").qlinear(cfg.mamba_d_ssm, cfg.hidden_size, false)?,
            a_log: vb.get(cfg.mamba_n_heads, "A_log")?.to_vec1::<f32>()?,
            d: vb.get(cfg.mamba_n_heads, "D")?.to_vec1::<f32>()?,
            dt_bias: vb.get(cfg.mamba_n_heads, "dt_bias")?.to_vec1::<f32>()?,
            d_ssm: cfg.mamba_d_ssm,
            conv_dim,
            d_conv: cfg.mamba_d_conv,
            n_heads: cfg.mamba_n_heads,
            head_dim: cfg.mamba_d_head,
            state: cfg.mamba_d_state,
            g_state: cfg.mamba_n_groups * cfg.mamba_d_state,
        })
    }

    /// A fresh decode cache (zero SSM state, zero conv history) for batch size `b`.
    pub fn init_cache(&self, b: usize, device: &Device) -> Result<MambaCache> {
        Ok(MambaCache {
            ssm: vec![0f32; b * self.n_heads * self.head_dim * self.state],
            conv: Tensor::zeros((b, self.conv_dim, self.d_conv - 1), candle_core::DType::F32, device)?,
        })
    }

    /// Sequential selective scan over `t` steps, advancing `state` in place (flat
    /// `(b · n_heads · head_dim · d_state)`). `xv/bv/cv/dtv` are the row-major `(b·t, …)` slices.
    /// Returns `y` flat `(b · t · d_ssm)`. Shared by the full and cached forwards. (n_groups=1 ⇒ B/C
    /// shared across heads.)
    fn scan(&self, xv: &[f32], bv: &[f32], cv: &[f32], dtv: &[f32], state: &mut [f32], (b, t): (usize, usize)) -> Vec<f32> {
        let (nh, p, n, ssm) = (self.n_heads, self.head_dim, self.state, self.d_ssm);
        let mut y = vec![0f32; b * t * ssm]; // (b, t, d_ssm), head-major (h·head_dim + p)
        for bi in 0..b {
            let st_off = bi * nh * p * n;
            for ti in 0..t {
                let row = bi * t + ti;
                for hh in 0..nh {
                    let dt_h = softplus(dtv[row * nh + hh] + self.dt_bias[hh]);
                    let da = (dt_h * -self.a_log[hh].exp()).exp(); // exp(dt · (−exp(A_log)))
                    let d_h = self.d[hh];
                    for pp in 0..p {
                        let xval = xv[row * ssm + hh * p + pp];
                        let dtx = dt_h * xval;
                        let base = st_off + (hh * p + pp) * n;
                        let mut acc = 0f32;
                        for nn in 0..n {
                            let s = &mut state[base + nn];
                            *s = *s * da + dtx * bv[row * n + nn];
                            acc += *s * cv[row * n + nn];
                        }
                        y[row * ssm + hh * p + pp] = acc + d_h * xval;
                    }
                }
            }
        }
        y
    }

    /// Gate / split / scan / out_proj shared tail, given the post-conv (pre-SiLU is applied here)
    /// `xbc` `(B, T, conv_dim)`, the `gate` slice `(B, T, d_ssm)`, and `dt` `(B, T, n_heads)`.
    fn finish(&self, xbc: &Tensor, gate: &Tensor, dt: &Tensor, state: &mut [f32], b: usize, t: usize) -> Result<Tensor> {
        let xbc = xbc.silu()?;
        let x = xbc.narrow(2, 0, self.d_ssm)?;
        let bmat = xbc.narrow(2, self.d_ssm, self.g_state)?;
        let cmat = xbc.narrow(2, self.d_ssm + self.g_state, self.g_state)?;
        let xv = x.flatten_all()?.to_vec1::<f32>()?;
        let bv = bmat.flatten_all()?.to_vec1::<f32>()?;
        let cv = cmat.flatten_all()?.to_vec1::<f32>()?;
        let dtv = dt.flatten_all()?.to_vec1::<f32>()?;
        let y = self.scan(&xv, &bv, &cv, &dtv, state, (b, t));
        let y = Tensor::from_vec(y, (b, t, self.d_ssm), xbc.device())?;
        let scan = (y * gate.silu()?)?; // gated, no norm
        self.out_proj.forward(&scan)
    }

    /// `h`: `(B, T, hidden)` → `(B, T, hidden)`. Full-sequence path (state starts at zero).
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let proj = self.in_proj.forward(h)?; // (B, T, d_ssm + conv_dim + n_heads)
        let gate = proj.narrow(2, 0, self.d_ssm)?;
        let xbc = proj.narrow(2, self.d_ssm, self.conv_dim)?;
        let dt = proj.narrow(2, self.d_ssm + self.conv_dim, self.n_heads)?;
        // Depthwise causal conv1d (pad k-1, take first T).
        let xbc = self.conv1d.forward(&xbc.transpose(1, 2)?)?; // (B, conv_dim, T + k-1)
        let xbc = xbc.narrow(2, 0, t)?.transpose(1, 2)?.contiguous()?; // (B, T, conv_dim)
        let mut state = vec![0f32; b * self.n_heads * self.head_dim * self.state];
        self.finish(&xbc, &gate, &dt, &mut state, b, t)
    }

    /// Cache-aware path: same math as [`forward`] but the conv history and SSM state carry across
    /// calls via `cache`, so a single new token is O(1) in sequence length. `h`: `(B, T, hidden)`.
    pub fn forward_cached(&self, h: &Tensor, cache: &mut MambaCache) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let proj = self.in_proj.forward(h)?;
        let gate = proj.narrow(2, 0, self.d_ssm)?;
        let xbc = proj.narrow(2, self.d_ssm, self.conv_dim)?;
        let dt = proj.narrow(2, self.d_ssm + self.conv_dim, self.n_heads)?;
        // Prepend the cached history, depthwise-conv with no padding (output is exactly the T new
        // positions), then refresh the history to the last d_conv-1 inputs.
        let full = Tensor::cat(&[&cache.conv, &xbc.transpose(1, 2)?], 2)?; // (B, conv_dim, d_conv-1 + T)
        let conv_out = self.depthwise_conv(&full, b, t)?; // (B, conv_dim, T) — hand-rolled, no BLAS
        let keep = self.d_conv - 1;
        cache.conv = full.narrow(2, full.dim(2)? - keep, keep)?.contiguous()?;
        let xbc = conv_out.transpose(1, 2)?.contiguous()?; // (B, T, conv_dim)
        self.finish(&xbc, &gate, &dt, &mut cache.ssm, b, t)
    }

    /// Hand-rolled depthwise causal conv1d for the cached path: `full` `(B, conv_dim, d_conv-1+T)` →
    /// `(B, conv_dim, T)`, `out[b,c,ti] = conv_b[c] + Σ_k conv_w[c,k]·full[b,c,ti+k]`. Each channel is
    /// independent (depthwise), so this is a tight scalar loop — no candle `Conv1d` / BLAS dispatch.
    fn depthwise_conv(&self, full: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        let (cd, k) = (self.conv_dim, self.d_conv);
        let w = k - 1 + t; // input width
        let fv = full.flatten_all()?.to_vec1::<f32>()?; // (b · conv_dim · w), row-major
        let mut out = vec![0f32; b * cd * t];
        for bi in 0..b {
            for c in 0..cd {
                let wbase = c * k;
                let ibase = (bi * cd + c) * w;
                let obase = (bi * cd + c) * t;
                let bias = self.conv_b[c];
                for ti in 0..t {
                    let mut acc = bias;
                    for kk in 0..k {
                        acc += self.conv_w[wbase + kk] * fv[ibase + ti + kk];
                    }
                    out[obase + ti] = acc;
                }
            }
        }
        Tensor::from_vec(out, (b, cd, t), full.device())
    }
}
