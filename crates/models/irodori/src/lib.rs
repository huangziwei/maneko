//! # irodori — Japanese TTS engine (port in progress)
//!
//! Rust/Candle port of Irodori-TTS: a flow-matching **DiT** + **DACVAE** codec (48 kHz), the
//! engine behind `nik`. Built on [`tts_core`]. Port plan: `ref/port-irodori-to-rust.md` and
//! `.claude/plans/maneko.md` §7.
//!
//! Status by milestone:
//! - **M1 DACVAE decode** — [`dacvae`] (32-dim VAE latent → waveform). ✅
//! - **M2 encoders** — [`encoders`] (text + reference-latent/speaker states). ✅
//! - **M3 DiT step** — [`dit`] (JointAttention + dual LowRankAdaLN → velocity). ✅
//! - **M4 RF/CFG sampler** — [`sampler`] (Euler + independent CFG → latent). ✅
//! - **M5 JP frontend + E2E** — [`text`] + [`engine`] (text + voice → 48 kHz wav). ✅

pub mod blocks;
pub mod config;
pub mod dacvae;
pub mod dit;
pub mod duration;
pub mod encoders;
pub mod engine;
pub mod sampler;
pub mod text;
pub mod weights;

pub use config::{DacvaeConfig, DitConfig};
pub use dacvae::Dacvae;
pub use dit::IrodoriDiT;
pub use duration::DurationPredictor;
pub use encoders::Encoders;
pub use engine::{GenerateOptions, Irodori};
pub use sampler::{sample_euler_cfg, SamplerConfig};
pub use text::{normalize_text, IrodoriTokenizer};
pub use weights::Weights;
