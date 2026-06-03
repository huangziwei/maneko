//! # irodori — Japanese TTS engine (port in progress)
//!
//! Rust/Candle port of Irodori-TTS (DiT backbone + DAC-VAE codec), the engine
//! behind `nik`. Not yet implemented — this scaffold exists so the workspace,
//! crate graph, and `interfaces/{cli,python}` wiring are in place before the port.
//!
//! Port plan: `ref/port-irodori-to-rust.md`.
//!
//! Planned modules:
//! - `dit`      — diffusion-transformer backbone
//! - `dacvae`   — DAC-VAE encode / decode
//! - `sampler`  — classifier-free guidance (text + speaker)
//! - `frontend` — Japanese text frontend (normalisation, phonemisation)
//!
//! Will build on `tts-core` once shared primitives are extracted.
