//! # tts-core — shared primitives for maneko's TTS engines
//!
//! Model-agnostic math + loaders that the model crates build on. Both engines consume it —
//! `irodori` (DiT + DACVAE) and `pocket` (Mimi + FlowLM). [`quant`] is the first capability shared
//! by both (q8 GGUF), per the rule that cross-cutting infra lives here, not baked into one model
//! crate (see `.claude/plans/maneko.md` §4).
//!
//! Modules:
//! - [`ops`]   — Snake, RMSNorm (f32-accumulated), scaled-dot-product attention
//! - [`quant`] — `Vb` (Full safetensors | Quant GGUF) + `QLinear` over `QMatMul` — pocket + irodori
//! - [`rope`]  — interleaved (rope_i) rotary embedding + half-by-heads variant
//! - [`wnorm`] — PyTorch weight-norm fold for conv weights
//! - [`audio`] — WAV output

pub mod audio;
pub mod ops;
pub mod quant;
pub mod rope;
pub mod wnorm;

pub use ops::{rms_norm, sdpa, snake};
pub use quant::{QLinear, Vb};
pub use rope::RotaryEmbedding;
pub use wnorm::fold_weight_norm;
