//! Irodori / DACVAE configuration.
//!
//! Mirrors `ref/mlx-audio/.../irodori_tts/config.py` and the DACVAE `config.json`. Only the
//! fields the Rust runtime needs are kept; the v3 duration-predictor and caption (Voice Design)
//! knobs are omitted until those paths are ported.

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
