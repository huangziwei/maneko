//! Falcon-H1 Mamba-2 mixer (the `torch_forward` / naive path, which the CPU golden runs).
//!
//! `in_proj → [gate, xBC, dt]`; depthwise causal conv1d + SiLU on `xBC`; split `→ [x, B, C]`; the
//! selective-scan recurrence `state = state·exp(dt·A) + (dt·x)·B`, `y = state·C + D·x`
//! (`A = −exp(A_log)`); gate (no RMSNorm here, `mamba_rms_norm=false`) `y·silu(gate)`; `out_proj`.
//! The recurrence is run sequentially in f32 — it reproduces both the chunked prefill and the
//! cached single-step decode of the reference.

use crate::config::FalconH1Config;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

/// `softplus` matching `torch.nn.functional.softplus` (beta 1, threshold 20).
fn softplus(z: f32) -> f32 {
    if z > 20.0 { z } else { (z.exp() + 1.0).ln() }
}

pub struct Mamba2 {
    in_proj: Linear,
    conv1d: Conv1d,
    out_proj: Linear,
    a_log: Vec<f32>,  // (n_heads,)
    d: Vec<f32>,      // (n_heads,)
    dt_bias: Vec<f32>, // (n_heads,)
    d_ssm: usize,
    conv_dim: usize,
    n_heads: usize,   // mamba heads (24)
    head_dim: usize,  // 32
    state: usize,     // d_state (64)
    g_state: usize,   // n_groups · d_state (64)
}

impl Mamba2 {
    pub fn load(cfg: &FalconH1Config, vb: VarBuilder) -> Result<Self> {
        let conv_dim = cfg.conv_dim();
        let proj_size = cfg.mamba_d_ssm + conv_dim + cfg.mamba_n_heads;
        let conv_cfg = Conv1dConfig {
            padding: cfg.mamba_d_conv - 1,
            groups: conv_dim,
            ..Default::default()
        };
        Ok(Self {
            in_proj: candle_nn::linear_no_bias(cfg.hidden_size, proj_size, vb.pp("in_proj"))?,
            conv1d: candle_nn::conv1d(conv_dim, conv_dim, cfg.mamba_d_conv, conv_cfg, vb.pp("conv1d"))?,
            out_proj: candle_nn::linear_no_bias(cfg.mamba_d_ssm, cfg.hidden_size, vb.pp("out_proj"))?,
            a_log: vb.get(cfg.mamba_n_heads, "A_log")?.to_vec1::<f32>()?,
            d: vb.get(cfg.mamba_n_heads, "D")?.to_vec1::<f32>()?,
            dt_bias: vb.get(cfg.mamba_n_heads, "dt_bias")?.to_vec1::<f32>()?,
            d_ssm: cfg.mamba_d_ssm,
            conv_dim,
            n_heads: cfg.mamba_n_heads,
            head_dim: cfg.mamba_d_head,
            state: cfg.mamba_d_state,
            g_state: cfg.mamba_n_groups * cfg.mamba_d_state,
        })
    }

    /// `h`: `(B, T, hidden)` → `(B, T, hidden)`.
    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (b, t, _) = h.dims3()?;
        let proj = self.in_proj.forward(h)?; // (B, T, d_ssm + conv_dim + n_heads)
        let gate = proj.narrow(2, 0, self.d_ssm)?;
        let xbc = proj.narrow(2, self.d_ssm, self.conv_dim)?;
        let dt = proj.narrow(2, self.d_ssm + self.conv_dim, self.n_heads)?;

        // Depthwise causal conv1d (pad k-1, take first T) + SiLU.
        let xbc = self.conv1d.forward(&xbc.transpose(1, 2)?)?; // (B, conv_dim, T + k-1)
        let xbc = xbc.narrow(2, 0, t)?.transpose(1, 2)?.contiguous()?.silu()?; // (B, T, conv_dim)
        let x = xbc.narrow(2, 0, self.d_ssm)?;
        let bmat = xbc.narrow(2, self.d_ssm, self.g_state)?;
        let cmat = xbc.narrow(2, self.d_ssm + self.g_state, self.g_state)?;

        // Sequential selective scan in f32. (n_groups=1 ⇒ B/C shared across heads.)
        let xv = x.flatten_all()?.to_vec1::<f32>()?;
        let bv = bmat.flatten_all()?.to_vec1::<f32>()?;
        let cv = cmat.flatten_all()?.to_vec1::<f32>()?;
        let dtv = dt.flatten_all()?.to_vec1::<f32>()?;
        let (nh, p, n, ssm) = (self.n_heads, self.head_dim, self.state, self.d_ssm);

        let mut y = vec![0f32; b * t * ssm]; // (B, T, d_ssm), head-major (h·head_dim + p)
        for bi in 0..b {
            let mut st = vec![0f32; nh * p * n];
            for ti in 0..t {
                let row = bi * t + ti;
                for hh in 0..nh {
                    let dt_h = softplus(dtv[row * nh + hh] + self.dt_bias[hh]);
                    let da = (dt_h * -self.a_log[hh].exp()).exp(); // exp(dt · (−exp(A_log)))
                    let d_h = self.d[hh];
                    for pp in 0..p {
                        let xval = xv[row * ssm + hh * p + pp];
                        let dtx = dt_h * xval;
                        let base = (hh * p + pp) * n;
                        let mut acc = 0f32;
                        for nn in 0..n {
                            let s = &mut st[base + nn];
                            *s = *s * da + dtx * bv[row * n + nn];
                            acc += *s * cv[row * n + nn];
                        }
                        y[row * ssm + hh * p + pp] = acc + d_h * xval;
                    }
                }
            }
        }
        let y = Tensor::from_vec(y, (b, t, ssm), h.device())?;
        let scan = (y * gate.silu()?)?; // gated, no norm
        self.out_proj.forward(&scan)
    }
}
