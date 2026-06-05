//! Duration predictor (Irodori v3).
//!
//! Port of `DurationPredictor` / `DurationSwiGLUBlock` from `ref/mlx-audio/.../irodori_tts/model.py`
//! for the v3 default architecture `token_sum_adarn_zero_no_aux`: project each text token to the
//! hidden dim, run it through SwiGLU blocks modulated by the speaker vector (AdaLN-Zero), emit a
//! per-token softplus frame count, and sum over the (masked) tokens. The result is the predicted
//! number of latent frames, used to set the sampler's sequence length instead of a fixed duration.
//!
//! The predictor is purely per-token (no attention), so padded positions never influence the real
//! ones — the trailing masked sum drops them. The other v3 fusions (pooled / cross-attn) and the
//! 14-dim aux features (`no_aux`) are not ported: the shipped checkpoint uses only this branch.

use crate::blocks::{RmsNorm, SwiGlu};
use crate::config::DitConfig;
use candle_core::{DType, Result, Tensor, D};
use tts_core::{QLinear, Vb};

/// One duration block: `x + tanh(gate) · SwiGLU(AdaLN-Zero(RMSNorm(x); speaker))`.
///
/// Modulation is `Linear(silu(cond))` → `[shift, scale, gate]` (each `dim`-wide), applied as
/// `h·(1+scale) + shift` before the SwiGLU with a `tanh(gate)` residual after (zero-init at train
/// time; loaded from weights here).
struct DurationSwiGluBlock {
    norm: RmsNorm,
    mlp: SwiGlu,
    modulation: QLinear,
}

impl DurationSwiGluBlock {
    fn load(vb: Vb, dim: usize, cond_dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(vb.pp("norm"), dim, eps)?,
            mlp: SwiGlu::load(vb.pp("mlp"), dim, dim)?,
            modulation: vb.pp("modulation").qlinear(cond_dim, dim * 3, true)?,
        })
    }

    /// `x`: `(B,S,dim)`, `cond`: `(B,cond_dim)` → `(B,S,dim)`.
    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let m = self.modulation.forward(&candle_nn::ops::silu(cond)?)?; // (B, 3*dim)
        let parts = m.chunk(3, D::Minus1)?;
        let shift = parts[0].unsqueeze(1)?; // (B,1,dim) — broadcast over S
        let scale = parts[1].unsqueeze(1)?;
        let gate = parts[2].unsqueeze(1)?;
        let h = h.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        let mlp_out = self.mlp.forward(&h)?;
        x.broadcast_add(&gate.tanh()?.broadcast_mul(&mlp_out)?)
    }
}

/// Irodori v3 duration predictor (`token_sum_adarn_zero_no_aux`).
pub struct DurationPredictor {
    token_input_proj: QLinear,
    token_blocks: Vec<DurationSwiGluBlock>,
    token_out_norm: RmsNorm,
    token_out_proj: QLinear,
    null_speaker: Tensor,
    speaker_dim: usize,
}

impl DurationPredictor {
    /// Load `duration_predictor.*` from the v3 DiT safetensors. Loading succeeds only if every
    /// tensor's shape matches `cfg` — so a successful load is itself a structural check.
    pub fn load(vb: Vb, cfg: &DitConfig) -> Result<Self> {
        let hidden = cfg.duration_hidden_dim;
        let token_blocks = (0..cfg.duration_layers)
            .map(|i| {
                DurationSwiGluBlock::load(
                    vb.pp("token_blocks").pp(i),
                    hidden,
                    cfg.speaker_dim,
                    cfg.norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_input_proj: vb.pp("token_input_proj").qlinear(cfg.text_dim, hidden, true)?,
            token_blocks,
            token_out_norm: RmsNorm::load(vb.pp("token_out_norm"), hidden, cfg.norm_eps)?,
            token_out_proj: vb.pp("token_out_proj").qlinear(hidden, 1, true)?,
            null_speaker: vb.get(cfg.speaker_dim, "null_speaker")?,
            speaker_dim: cfg.speaker_dim,
        })
    }

    /// Predict latent frames for `text_state` `(B,S,text_dim)` conditioned on the speaker.
    /// `text_mask` `(B,S)` (1 valid / 0 pad) masks the per-token sum. When `has_speaker`, the
    /// speaker vector is `speaker_state[:,0]`; otherwise the learned `null_speaker`. Returns a
    /// `(B,)` f32 tensor of non-negative frame counts. (The reference returns `log1p(frames)` and
    /// the caller `expm1`s it; that round-trip is the identity, so it is collapsed here.)
    pub fn forward(
        &self,
        text_state: &Tensor,
        text_mask: &Tensor,
        speaker_state: &Tensor,
        has_speaker: bool,
    ) -> Result<Tensor> {
        let (b, _s, _d) = text_state.dims3()?;
        let speaker_vec = if has_speaker {
            speaker_state.narrow(1, 0, 1)?.squeeze(1)? // (B, speaker_dim)
        } else {
            self.null_speaker
                .reshape((1, self.speaker_dim))?
                .broadcast_as((b, self.speaker_dim))?
                .contiguous()?
        };

        let mut h = self.token_input_proj.forward(text_state)?; // (B,S,hidden)
        for block in &self.token_blocks {
            h = block.forward(&h, &speaker_vec)?;
        }
        let h = self.token_out_norm.forward(&h)?;
        let token_logits = self.token_out_proj.forward(&h)?.squeeze(D::Minus1)?; // (B,S)

        // Per-token softplus = log(1 + exp(x)) in f32 (matches the reference's naive form).
        let token_logits = token_logits.to_dtype(DType::F32)?;
        let token_frames = (token_logits.exp()? + 1.0)?.log()?; // (B,S)
        let mask = text_mask.to_dtype(DType::F32)?;
        let total = token_frames.broadcast_mul(&mask)?.sum(D::Minus1)?; // (B,)
        total.relu() // max(total, 0)
    }
}
