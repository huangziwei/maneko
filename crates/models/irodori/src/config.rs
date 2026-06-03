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

/// Irodori DiT config (the `dit` block of the model `config.json`). Speaker-conditioned v2;
/// caption (Voice Design) and v3 duration-predictor fields are omitted until ported.
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
