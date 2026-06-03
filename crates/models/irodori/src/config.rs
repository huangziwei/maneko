//! Irodori / DACVAE configuration.
//!
//! Mirrors `ref/mlx-audio/.../irodori_tts/config.py` and the DACVAE `config.json`. Only the
//! fields the Rust runtime needs are kept; the v3 duration-predictor fields are included (see
//! `v3()`), while the caption (Voice Design) knobs are still omitted until that path is ported.

use serde::Deserialize;

/// DACVAE audio codec config (`Aratako/Semantic-DACVAE-Japanese-32dim`, the v2 32-dim variant).
///
/// `latent_dim` is the decoder's working channel count; `codebook_dim` (32) is the VAE latent the
/// DiT operates in (the decoder's `quantizer_out_proj` lifts 32 → `latent_dim`). `hop_length`
/// = ∏`decoder_rates` = 1920 samples/frame at 48 kHz.
#[derive(Debug, Clone, Deserialize)]
pub struct DacvaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub codebook_dim: usize,
    pub sample_rate: usize,
}

impl DacvaeConfig {
    /// The v2 Japanese DACVAE config. The `Aratako` repo ships only `weights.pth` (no
    /// `config.json`); these values come from the checkpoint metadata and the mlx-community
    /// `dacvae/config.json`.
    pub fn v2() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: vec![2, 8, 10, 12],
            latent_dim: 1024,
            decoder_dim: 1536,
            decoder_rates: vec![12, 10, 8, 2],
            codebook_dim: 32,
            sample_rate: 48_000,
        }
    }

    /// Samples per latent frame (= ∏ decoder_rates).
    pub fn hop_length(&self) -> usize {
        self.decoder_rates.iter().product()
    }
}

/// Irodori DiT config (the `dit` block of the model `config.json`). Speaker-conditioned; the v3
/// duration-predictor fields are included (`use_duration_predictor` gates the module), while
/// caption (Voice Design) fields are omitted until ported.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    pub latent_dim: usize,
    pub latent_patch_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub text_mlp_ratio: f64,
    pub speaker_mlp_ratio: f64,
    pub text_vocab_size: usize,
    pub text_dim: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub speaker_dim: usize,
    pub speaker_layers: usize,
    pub speaker_heads: usize,
    pub speaker_patch_size: usize,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f64,

    /// Duration predictor (v3). When false (v2) the module is absent and `generate` falls back to
    /// the `seconds` arg / 30 s default. `hidden`/`layers` size the `token_sum_adarn_zero_no_aux`
    /// predictor; `text_dim`/`speaker_dim` above are its other dims.
    #[serde(default)]
    pub use_duration_predictor: bool,
    #[serde(default)]
    pub duration_hidden_dim: usize,
    #[serde(default)]
    pub duration_layers: usize,
}

impl DitConfig {
    /// `Aratako/Irodori-TTS-500M-v2` (matches the mlx-community `config.json` `dit` block).
    pub fn v2() -> Self {
        Self {
            latent_dim: 32,
            latent_patch_size: 1,
            model_dim: 1280,
            num_layers: 12,
            num_heads: 20,
            mlp_ratio: 2.875,
            text_mlp_ratio: 2.6,
            speaker_mlp_ratio: 2.6,
            text_vocab_size: 99574,
            text_dim: 512,
            text_layers: 10,
            text_heads: 8,
            speaker_dim: 768,
            speaker_layers: 8,
            speaker_heads: 12,
            speaker_patch_size: 1,
            timestep_embed_dim: 512,
            adaln_rank: 192,
            norm_eps: 1e-5,
            use_duration_predictor: false,
            duration_hidden_dim: 1024,
            duration_layers: 3,
        }
    }

    /// `Aratako/Irodori-TTS-500M-v3` — the v2 architecture **plus** the integrated Duration
    /// Predictor (`token_sum_adarn_zero_no_aux`). Same DiT/encoder/DACVAE family as v2 (the v3
    /// checkpoint adds only the `duration_predictor.*` tensors); enabling it lets `generate`
    /// predict the output length from text + speaker instead of using a fixed `seconds`.
    pub fn v3() -> Self {
        Self {
            use_duration_predictor: true,
            ..Self::v2()
        }
    }

    /// Head dimension of the DiT backbone (= text/speaker head dim too; all are 64 in v2).
    pub fn head_dim(&self) -> usize {
        self.model_dim / self.num_heads
    }

    /// Patched latent dim the DiT operates on (`latent_dim · latent_patch_size`).
    pub fn patched_latent_dim(&self) -> usize {
        self.latent_dim * self.latent_patch_size
    }

    /// Speaker encoder input dim (`patched_latent_dim · speaker_patch_size`).
    pub fn speaker_in_dim(&self) -> usize {
        self.patched_latent_dim() * self.speaker_patch_size
    }
}
