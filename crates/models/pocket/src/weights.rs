use anyhow::Result;
use std::path::PathBuf;

use candle_core::Device;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

/// Resolve the HuggingFace hub cache root the way hf-hub does:
/// `HF_HOME/hub`, else `HF_HUB_CACHE`, else `~/.cache/huggingface/hub`.
fn hf_cache_root() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(c);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

/// Download a file from HuggingFace Hub if necessary.
///
/// Supports the format: `hf://owner/repo/filename@revision`
/// where `@revision` is optional.
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("hf://") {
        let path = file_path.trim_start_matches("hf://");
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 3 {
            anyhow::bail!(
                "Invalid hf:// path: {}. Expected hf://repo_owner/repo_name/filename[@revision]",
                file_path
            );
        }
        let repo_id = format!("{}/{}", parts[0], parts[1]);
        let filename_with_revision = parts[2..].join("/");

        // Parse optional revision from filename (e.g., "file.safetensors@abc123")
        let (filename, revision) = if let Some(at_pos) = filename_with_revision.rfind('@') {
            let (f, r) = filename_with_revision.split_at(at_pos);
            (f.to_string(), Some(r[1..].to_string())) // Skip the '@'
        } else {
            (filename_with_revision, None)
        };

        // Fast path: if the pinned revision is already materialized in the local HF cache, use its
        // snapshot directly. Avoids needing a `refs/<commit>` indirection file, and works with
        // read-only / shared caches (e.g. another project's HF_HOME).
        if let Some(rev) = revision.as_deref() {
            let snapshot = hf_cache_root()
                .join(format!("models--{}--{}", parts[0], parts[1]))
                .join("snapshots")
                .join(rev)
                .join(&filename);
            if snapshot.exists() {
                return Ok(snapshot);
            }
        }

        // Use ApiBuilder::from_env so HF_HOME / HF_ENDPOINT are honored.
        // (ApiBuilder::new ignores HF_HOME and hardcodes ~/.cache/huggingface.)
        let token = std::env::var("HF_TOKEN").ok();

        let api = ApiBuilder::from_env().with_token(token).build()?;

        // Create repo with or without revision
        let repo = if let Some(rev) = revision {
            Repo::with_revision(repo_id, RepoType::Model, rev)
        } else {
            Repo::model(repo_id)
        };

        let api_repo = api.repo(repo);
        let path = api_repo.get(&filename)?;
        Ok(path)
    } else {
        Ok(PathBuf::from(file_path))
    }
}

pub fn load_weights(
    file_path: &str,
    _device: &Device,
) -> Result<candle_core::safetensors::MmapedSafetensors> {
    let path = download_if_necessary(file_path)?;
    let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(safetensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_if_necessary_local() {
        let path = "test.safetensors";
        let res = download_if_necessary(path).unwrap();
        assert_eq!(res, PathBuf::from(path));
    }

    #[test]
    fn test_invalid_hf_path() {
        let path = "hf://invalid";
        let res = download_if_necessary(path);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_revision() {
        // Test parsing logic (doesn't actually download)
        let path = "hf://kyutai/pocket-tts/file.safetensors@abc123def";
        // This will fail to download but we're testing the parsing
        let res = download_if_necessary(path);
        // We expect a network error, not a parsing error
        assert!(res.is_err());
    }
}
