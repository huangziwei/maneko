//! # tts-core — shared primitives for maneko's TTS engines
//!
//! Foundation crate that both `pocket` and `irodori` build on. Intentionally
//! empty for now: modules are extracted from `pocket` once the shared surface
//! stabilises, so the crate graph and import paths are settled ahead of the work.
//!
//! Planned modules (see `.claude/plans/maneko.md` §4):
//! - `ops`       — tensor helpers shared across models
//! - `conv`      — SEANet / causal conv blocks
//! - `attention` — streaming-transformer attention
//! - `audio`     — wav I/O, resample, peak-normalise
//! - `weights`   — safetensors + hf-hub loading
//! - `device`    — device / dtype selection
//! - `quantize`  — quantised tensor support
