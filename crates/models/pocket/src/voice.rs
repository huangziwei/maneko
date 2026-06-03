//! Voice resolution: turn a voice *spec* into a [`ModelState`] (voice conditioning).
//!
//! A spec is one of:
//! - a predefined stock-voice name (`alba`, `marius`, …) — fetched from HF as embeddings,
//! - a local file path — `.wav` (cloned via Mimi) or `.safetensors` (precomputed embeddings),
//! - an `hf://owner/repo/file` URL,
//! - base64-encoded WAV (a `data:audio/...;base64,…` URL or a long raw base64 string).
//!
//! Resolution is **per model**: a voice-state is Mimi-encoded against a specific [`TTSModel`],
//! so the same spec resolves to a *different* state for each language model. [`crate::Engine`]
//! caches the result keyed `(voice_cache_key(spec), language)`; [`voice_cache_key`] builds that key.

use crate::TTSModel;
use crate::voice_state::ModelState;
use crate::weights::download_if_necessary;
use anyhow::{Context, Result};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

/// Predefined stock voices from `kyutai/pocket-tts-without-voice-cloning` (the pocket-tts v2 set).
///
/// **Voices are language-specific.** Five are native to a non-English language; the rest are
/// English. Pair each with the matching language model (resolution is per-model anyway):
/// `juergen` → de, `lola` → es, `estelle` → fr, `giovanni` → it, `rafael` → pt; all others → en.
/// See [`predefined_voice_language`].
pub const PREDEFINED_VOICES: &[&str] = &[
    "alba",
    "anna",
    "azelma",
    "bill_boerst",
    "caro_davy",
    "charles",
    "cosette",
    "eponine",
    "estelle",
    "eve",
    "fantine",
    "george",
    "giovanni",
    "jane",
    "javert",
    "jean",
    "juergen",
    "lola",
    "marius",
    "mary",
    "michael",
    "paul",
    "peter_yearsley",
    "rafael",
    "stuart_bell",
    "vera",
];

/// The language a predefined voice is native to (config-stem family: `en`/`de`/`es`/`fr`/`it`/`pt`).
/// Returns `None` for unknown names. The five non-English voices are the only ones not `en`.
pub fn predefined_voice_language(name: &str) -> Option<&'static str> {
    if !PREDEFINED_VOICES.contains(&name) {
        return None;
    }
    Some(match name {
        "juergen" => "de",
        "lola" => "es",
        "estelle" => "fr",
        "giovanni" => "it",
        "rafael" => "pt",
        _ => "en",
    })
}

/// HuggingFace repo for stock voice embeddings.
const STOCK_VOICE_REPO: &str = "kyutai/pocket-tts-without-voice-cloning";

/// Pinned revision of the stock-voice repo. Bumped from the original `d29db79` (which predated the
/// non-English voices) to `main` @ `e041936c`, which ships all 26 — including the per-language
/// native voices (`juergen`/`lola`/`estelle`/`giovanni`/`rafael`).
const STOCK_VOICE_REV: &str = "e041936c75475d350b405bc870bcf7c22da4e9e6";

/// HF path of a **per-language** predefined voice embedding (pocket-tts v2 layout).
pub fn predefined_voice_hf_path(language: &str, name: &str) -> String {
    format!(
        "hf://{STOCK_VOICE_REPO}/languages/{language}/embeddings/{name}.safetensors@{STOCK_VOICE_REV}"
    )
}

/// Build a stable cache key for a voice specification.
///
/// File keys include mtime/size so on-disk updates invalidate cached entries. The key is
/// model-independent — pair it with a language/variant to key a per-model voice-state cache.
pub fn voice_cache_key(spec: &str) -> String {
    let spec = spec.trim();

    if PREDEFINED_VOICES.contains(&spec) {
        return format!("stock:{spec}");
    }
    if spec.starts_with("hf://") {
        return format!("hf:{spec}");
    }

    let path = PathBuf::from(spec);
    if path.exists() {
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let (mtime_secs, size) = std::fs::metadata(&canonical)
            .ok()
            .map(|m| {
                let modified = m
                    .modified()
                    .ok()
                    .and_then(|ts| ts.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                (modified, m.len())
            })
            .unwrap_or((0, 0));
        return format!("file:{}:{mtime_secs}:{size}", canonical.display());
    }

    if is_base64_audio(spec) {
        return format!("b64:{:016x}", hash_str(spec));
    }

    format!("raw:{}:{:016x}", spec.len(), hash_str(spec))
}

fn hash_str(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Resolve an optional voice spec to a [`ModelState`] against `model`.
///
/// `None` defaults to the `alba` stock voice. `language` (the model's config stem) is used only
/// for predefined voices, whose embeddings are per-language in v2.
pub fn resolve_voice(
    model: &TTSModel,
    voice_spec: Option<&str>,
    language: Option<&str>,
) -> Result<ModelState> {
    match voice_spec {
        Some(spec) => resolve_voice_spec(model, spec, language),
        None => resolve_predefined_voice(model, "alba", language),
    }
}

/// Resolve a concrete voice spec to a [`ModelState`] against `model`. See [`resolve_voice`].
pub fn resolve_voice_spec(
    model: &TTSModel,
    spec: &str,
    language: Option<&str>,
) -> Result<ModelState> {
    let spec = spec.trim();

    if PREDEFINED_VOICES.contains(&spec) {
        return resolve_predefined_voice(model, spec, language);
    }
    if spec.starts_with("hf://") {
        return resolve_hf_voice(model, spec);
    }
    let path = PathBuf::from(spec);
    if path.exists() {
        return resolve_file_voice(model, &path);
    }
    if is_base64_audio(spec) {
        return resolve_base64_voice(model, spec);
    }

    anyhow::bail!(
        "Voice '{}' not found. Expected one of:\n\
         - Predefined name: {}\n\
         - File path: /path/to/voice.wav or /path/to/embeddings.safetensors\n\
         - HuggingFace URL: hf://owner/repo/file.wav\n\
         - Base64 audio: data:audio/wav;base64,...",
        spec,
        PREDEFINED_VOICES.join(", ")
    )
}

/// Resolve a predefined voice name to embeddings via HF Hub.
///
/// v2 ships per-language embeddings (`…/languages/{lang}/embeddings/{name}.safetensors`); when a
/// `language` is known we use that path, falling back to the flat `…/embeddings/{name}` path if the
/// per-language file is unavailable.
fn resolve_predefined_voice(
    model: &TTSModel,
    name: &str,
    language: Option<&str>,
) -> Result<ModelState> {
    if let Some(lang) = language {
        let per_lang = predefined_voice_hf_path(lang, name);
        match download_if_necessary(&per_lang) {
            Ok(path) => {
                return model
                    .get_voice_state_from_prompt_file(&path)
                    .with_context(|| format!("Failed to load voice embeddings from {:?}", path));
            }
            Err(e) => tracing::debug!(
                "per-language voice '{name}' for '{lang}' unavailable ({e}); falling back to flat path"
            ),
        }
    }

    let flat = format!("hf://{}/embeddings/{}.safetensors", STOCK_VOICE_REPO, name);
    let local_path = download_if_necessary(&flat)
        .with_context(|| format!("Failed to download stock voice '{}'", name))?;
    model
        .get_voice_state_from_prompt_file(&local_path)
        .with_context(|| format!("Failed to load voice embeddings from {:?}", local_path))
}

/// Resolve an `hf://` URL (audio or safetensors).
fn resolve_hf_voice(model: &TTSModel, url: &str) -> Result<ModelState> {
    let local_path = download_if_necessary(url)
        .with_context(|| format!("Failed to download voice from '{}'", url))?;

    resolve_file_voice(model, &local_path)
}

/// Resolve a local file (WAV audio or safetensors embeddings).
fn resolve_file_voice(model: &TTSModel, path: &PathBuf) -> Result<ModelState> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "safetensors" => model
            .get_voice_state_from_prompt_file(path)
            .with_context(|| format!("Failed to load embeddings from {:?}", path)),
        "wav" | "wave" => model
            .get_voice_state(path)
            .with_context(|| format!("Failed to process audio from {:?}", path)),
        _ => anyhow::bail!(
            "Unsupported file extension '{}' for voice file. Expected .wav or .safetensors",
            ext
        ),
    }
}

/// Check if a string looks like base64 audio.
fn is_base64_audio(spec: &str) -> bool {
    if spec.starts_with("data:audio/") && spec.contains("base64,") {
        return true;
    }
    // Raw base64: WAV header is 44 bytes, ~60 base64 chars minimum; require some length.
    if spec.len() > 100 {
        let clean = spec.trim();
        return clean
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=');
    }
    false
}

/// Resolve base64-encoded WAV audio.
fn resolve_base64_voice(model: &TTSModel, spec: &str) -> Result<ModelState> {
    let b64_str = if spec.starts_with("data:") {
        spec.split(',').nth(1).unwrap_or(spec)
    } else {
        spec
    };

    use base64::{Engine as _, engine::general_purpose};
    let bytes = general_purpose::STANDARD
        .decode(b64_str)
        .context("Failed to decode base64 audio")?;

    model
        .get_voice_state_from_bytes(&bytes)
        .context("Failed to encode base64 audio for voice cloning")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predefined_voices_list() {
        assert!(PREDEFINED_VOICES.contains(&"alba"));
        assert!(PREDEFINED_VOICES.contains(&"marius"));
        assert!(!PREDEFINED_VOICES.contains(&"unknown"));
    }

    #[test]
    fn base64_audio_detection() {
        assert!(is_base64_audio(
            "data:audio/wav;base64,UklGRi4AAABXQVZFZm10IBAAAAABAAIAQB8AAEAfAAABAAgAZGF0YQoAAAAA"
        ));
        assert!(!is_base64_audio("alba"));
        assert!(!is_base64_audio("/path/to/file.wav"));
        assert!(!is_base64_audio("short"));
    }

    #[test]
    fn cache_key_stock() {
        assert_eq!(voice_cache_key("alba"), "stock:alba");
    }

    #[test]
    fn cache_key_base64_differs() {
        let k1 = voice_cache_key("data:audio/wav;base64,AAAA");
        let k2 = voice_cache_key("data:audio/wav;base64,AAAB");
        assert!(k1.starts_with("b64:"));
        assert!(k2.starts_with("b64:"));
        assert_ne!(k1, k2);
    }
}
