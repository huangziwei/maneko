//! HuggingFace weight resolution (respects `HF_HOME`), mirroring irodori's loader.

use anyhow::Result;
use std::path::PathBuf;

/// Resolve `file` from `repo` via the HF cache / hub (honours `HF_HOME`).
pub fn hf_file(repo: &str, file: &str) -> Result<PathBuf> {
    let api = hf_hub::api::sync::Api::new()?;
    Ok(api.model(repo.to_string()).get(file)?)
}
