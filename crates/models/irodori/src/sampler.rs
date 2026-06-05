//! Rectified-Flow Euler sampler with Classifier-Free Guidance (`independent` mode).
//!
//! Port of `sample_euler_cfg` (sampling.py) for the default speaker-conditioned path: text and
//! speaker guidance in a single 3×-batch forward `[cond, text-uncond, speaker-uncond]`. The uncond
//! passes reuse the conditional KV but zero the corresponding attention mask (a zero mask removes
//! that context entirely), so only the masks differ across the batch.
//!
//! `v = v_cond + s_text·(v_cond − v_text⁰) + s_spk·(v_cond − v_spk⁰)`, guidance gated to
//! `t ∈ [cfg_min_t, cfg_max_t]`; Euler update `x += v · (t_next − t)` over the schedule
//! `linspace(0.999, 0, N+1)`. The caller supplies the initial Gaussian noise `x_init` (production
//! draws it; parity tests inject MLX's noise — diffusion is deterministic given the init).

use crate::dit::IrodoriDiT;
use candle_core::{DType, Result, Tensor};

/// Sampler settings. CFG scales/thresholds match the v2 `config.json` `sampler` block; `num_steps`
/// defaults to maneko's v3-validated **8** (the duration predictor sizes the clip, so 8 holds
/// intelligibility on the Whisper round-trip; raise for more prosody fidelity on long-form).
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub num_steps: usize,
    pub cfg_scale_text: f64,
    pub cfg_scale_speaker: f64,
    pub cfg_min_t: f64,
    pub cfg_max_t: f64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            num_steps: 8,
            cfg_scale_text: 3.0,
            cfg_scale_speaker: 5.0,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
        }
    }
}

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    (0..n)
        .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
        .collect()
}

/// Run the RF Euler/CFG sampler. `text_input_ids`/`text_mask`: `(B,St)`; `ref_latent`/`ref_mask`:
/// `(B,T,latent_dim)`/`(B,T)`; `x_init`: `(B, seq, latent_dim)` initial noise. Returns the final
/// latent `(B, seq, latent_dim)`.
pub fn sample_euler_cfg(
    dit: &IrodoriDiT,
    text_input_ids: &Tensor,
    text_mask: &Tensor,
    ref_latent: &Tensor,
    ref_mask: &Tensor,
    x_init: &Tensor,
    cfg: &SamplerConfig,
) -> Result<Tensor> {
    let dev = x_init.device().clone();
    let (text_state, speaker_state) = dit.encode_conditions(text_input_ids, text_mask, ref_latent, ref_mask)?;
    let kv_cond = dit.build_kv_cache(&text_state, &speaker_state)?;

    let has_text = cfg.cfg_scale_text > 0.0;
    let has_speaker = cfg.cfg_scale_speaker > 0.0;
    let do_cfg = has_text && has_speaker; // this port implements the both-active independent path

    // CFG batch (×3) caches + masks: [cond, text-uncond, speaker-uncond].
    let kv_cfg = if do_cfg { Some(kv_cond.replicate_batch(3)?) } else { None };
    let text_mask_uncond = text_mask.zeros_like()?;
    let spk_mask_uncond = ref_mask.zeros_like()?;
    let text_mask_cfg = Tensor::cat(&[text_mask, &text_mask_uncond, text_mask], 0)?;
    let spk_mask_cfg = Tensor::cat(&[ref_mask, ref_mask, &spk_mask_uncond], 0)?;

    let schedule = linspace(0.999, 0.0, cfg.num_steps + 1);
    let mut x = x_init.clone();

    for i in 0..cfg.num_steps {
        let t = schedule[i];
        let t_next = schedule[i + 1];
        let use_cfg = do_cfg && cfg.cfg_min_t <= t && t <= cfg.cfg_max_t;

        let v_pred = if use_cfg {
            let x_cfg = Tensor::cat(&[&x, &x, &x], 0)?; // (3B, seq, D)
            let t_cfg = Tensor::full(t as f32, x_cfg.dim(0)?, &dev)?;
            let v_out = dit.forward_with_conditions(
                &x_cfg,
                &t_cfg,
                &text_mask_cfg,
                &spk_mask_cfg,
                kv_cfg.as_ref().unwrap(),
                0,
            )?;
            let p = v_out.chunk(3, 0)?;
            let (vc, vt, vs) = (&p[0], &p[1], &p[2]);
            let g_text = ((vc - vt)? * cfg.cfg_scale_text)?;
            let g_spk = ((vc - vs)? * cfg.cfg_scale_speaker)?;
            ((vc + g_text)? + g_spk)?
        } else {
            let t_arr = Tensor::full(t as f32, x.dim(0)?, &dev)?;
            dit.forward_with_conditions(&x, &t_arr, text_mask, ref_mask, &kv_cond, 0)?
        };

        x = (&x + (v_pred * (t_next - t))?)?;
    }
    x.to_dtype(DType::F32)
}
