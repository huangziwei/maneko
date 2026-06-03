//! # tts-core — shared primitives for maneko's TTS engines
//!
//! Model-agnostic math that the model crates build on. Today its first consumer is `irodori`
//! (the DiT + DACVAE port, P3); `pocket` still carries its own copies of the streaming-specific
//! variants and migrates onto this crate later (see `.claude/plans/maneko.md` §4). Everything
//! here is fresh, unit-tested code rather than a refactor of `pocket`, so bringing `irodori` up
//! cannot regress the working `pocket` engine.
//!
//! Modules:
//! - [`ops`]   — Snake, RMSNorm (f32-accumulated), scaled-dot-product attention
//! - [`rope`]  — interleaved (rope_i) rotary embedding + half-by-heads variant
//! - [`wnorm`] — PyTorch weight-norm fold for conv weights
//! - [`audio`] — WAV output

pub mod audio;
pub mod ops;
pub mod rope;
pub mod wnorm;

pub use ops::{rms_norm, sdpa, snake};
pub use rope::RotaryEmbedding;
pub use wnorm::fold_weight_norm;
