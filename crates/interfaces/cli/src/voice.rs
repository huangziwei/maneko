//! Voice resolution for the CLI.
//!
//! The implementation now lives in [`pocket::voice`] so the engine owns it (a voice-state is
//! Mimi-encoded per model, and [`pocket::Engine`] caches it per `(voice, language)`). This module
//! re-exports it so existing CLI call sites are unchanged.

pub use pocket::voice::{PREDEFINED_VOICES, resolve_voice, resolve_voice_spec, voice_cache_key};
