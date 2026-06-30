//! HuggingFace weight resolution (respects `HF_HOME`), mirroring irodori's loader.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// maneko's self-contained weight bundle (q8 AR GGUF + WavLM voice-encoder), mirroring how irodori
/// and pocket ship their q8 GGUF here. `mio-tts/ar.q8.gguf`, `mio-tts/mio_wavlm.safetensors`.
pub const MANEKO_REPO: &str = "zwaiwng/maneko";

/// Resolve `file` from `repo` via the HF cache / hub (honours `HF_HOME`).
pub fn hf_file(repo: &str, file: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    Ok(api.model(repo.to_string()).get(file)?)
}

/// Resolve the q8 AR GGUF (the fast Intel-CPU path): `explicit` → `$MIO_AR_Q8` → the project-local
/// `.cache/mio_ar.q8.gguf` → `zwaiwng/maneko/mio-tts/ar.q8.gguf`.
pub fn resolve_ar_q8(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_asset(explicit, "MIO_AR_Q8", ".cache/mio_ar.q8.gguf", "mio-tts/ar.q8.gguf")
}

/// Resolve the WavLM voice-encoder bundle: `explicit` → `$MIO_WAVLM` → the project-local
/// `.cache/mio_wavlm.safetensors` → `zwaiwng/maneko/mio-tts/mio_wavlm.safetensors`.
pub fn resolve_wavlm(explicit: Option<&Path>) -> Result<PathBuf> {
    resolve_asset(explicit, "MIO_WAVLM", ".cache/mio_wavlm.safetensors", "mio-tts/mio_wavlm.safetensors")
}

/// Shared asset resolution: explicit override → env var → project-local dev copy → maneko HF repo.
fn resolve_asset(explicit: Option<&Path>, env: &str, local: &str, hf: &str) -> Result<PathBuf> {
    if let Some(p) = explicit {
        anyhow::ensure!(p.exists(), "{} not found: {}", hf, p.display());
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os(env) {
        let p = PathBuf::from(p);
        anyhow::ensure!(p.exists(), "{env}={} does not exist", p.display());
        return Ok(p);
    }
    let local = PathBuf::from(local);
    if local.exists() {
        return Ok(local);
    }
    hf_file(MANEKO_REPO, hf)
}
