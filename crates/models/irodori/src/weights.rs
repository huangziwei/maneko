//! Weight loading for Irodori.
//!
//! Two sources, both read **natively** with no Python/torch step:
//! - the DiT (`Aratako/Irodori-TTS-500M-v2/model.safetensors`, already f32) via safetensors;
//! - the DACVAE (`Aratako/Semantic-DACVAE-Japanese-32dim/weights.pth`, a PyTorch checkpoint) via
//!   candle's pickle reader (`read_all_with_key(.., "state_dict")`).
//!
//! Conv weights in the torch `.pth` are already in Candle's layout (conv1d `(out,in,k)`,
//! conv_transpose1d `(in,out,k)`) — so no permutation is needed, unlike a port from MLX. Weight-
//! norm is folded at build time by the model modules via [`tts_core::fold_weight_norm`].

use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

/// A flat name → tensor map, moved onto the target device.
pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    /// Load a PyTorch `.pth` checkpoint. `key` selects a sub-dict (e.g. `Some("state_dict")`).
    pub fn from_pth<P: AsRef<std::path::Path>>(
        path: P,
        key: Option<&str>,
        device: &Device,
    ) -> Result<Self> {
        let tensors = candle_core::pickle::read_all_with_key(path, key)?;
        let mut map = HashMap::with_capacity(tensors.len());
        for (name, t) in tensors {
            map.insert(name, t.to_device(device)?);
        }
        Ok(Self { map })
    }

    /// Load a safetensors file.
    pub fn from_safetensors<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let map = candle_core::safetensors::load(path, device)?;
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.map
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Resolve a file from a HuggingFace repo, honoring `HF_HOME` (already-cached files resolve to
/// their snapshot path without a network call).
pub fn hf_file(repo: &str, file: &str) -> anyhow::Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let api = ApiBuilder::from_env().build()?;
    let path = api.model(repo.to_string()).get(file)?;
    Ok(path)
}

/// Like [`hf_file`] but pinned to an exact commit `rev` (reproducible) — for maneko's own published
/// weights repo. An already-cached revision resolves without a network call.
pub fn hf_file_rev(repo: &str, file: &str, rev: &str) -> anyhow::Result<std::path::PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};
    let api = ApiBuilder::from_env().build()?;
    let path = api
        .repo(Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string()))
        .get(file)?;
    Ok(path)
}
