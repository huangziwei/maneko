//! Multi-language pocket-tts engine.
//!
//! pocket is monolingual — one model per language. A multilingual book is rendered by switching
//! models per chunk on the chunk's language. [`Engine`] makes that cheap: it holds several language
//! models in memory (lazy-loaded, LRU-evicted) and caches voice-states keyed `(voice, language)`,
//! so switching languages between chunks is a hash lookup, not a reload + re-encode.
//!
//! ```no_run
//! use pocket::Engine;
//! use candle_core::Device;
//! let mut engine = Engine::new(Device::Cpu);
//! let en = engine.generate("Hello.",        "english_2026-04", Some("alba"))?;
//! let de = engine.generate("Hallo, Welt.",  "german",   Some("alba"))?; // loads german, reuses cache next time
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! `language` is a **config stem** (e.g. `"english_2026-04"`, `"german"`, `"french_24l"`) — the same
//! identifier [`TTSModel::load`] accepts and a file under `config/`. Mapping a human language name
//! (+ layer choice) to a stem is the caller's concern; the engine routes on the stem.

use crate::config::defaults;
use crate::tts_model::TTSModel;
use crate::voice::{resolve_voice_spec, voice_cache_key};
use crate::voice_state::ModelState;
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Default ceiling on how many language models the engine holds at once.
/// Lazy loading means only the languages actually used are resident; this just bounds the worst
/// case. Each FlowLM+Mimi model is ≈225 MB+ (more for 24-layer), so raise it deliberately.
pub const DEFAULT_CAPACITY: usize = 8;

/// Size the candle CPU thread pool once, before any op runs — **x86_64 only**. pocket's
/// autoregressive generation + gen/decode overlap saturates ~physical cores and *regresses* under
/// SMT oversubscription (e.g. 16 logical threads on an 8-core x86 ran ~15% slower than 8), so cap
/// to physical ≈ logical/2. Apple Silicon / other arches are left untouched (they keep their
/// default threading and Accelerate/Metal paths). An explicit `RAYON_NUM_THREADS` always wins.
fn init_rayon_pool() {
    #[cfg(target_arch = "x86_64")]
    {
        static RAYON_INIT: std::sync::Once = std::sync::Once::new();
        RAYON_INIT.call_once(|| {
            if std::env::var_os("RAYON_NUM_THREADS").is_some() {
                return;
            }
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let threads = (logical / 2).max(1);
            // candle's matmul/conv backend (the `gemm` crate) sizes its worker pool from
            // RAYON_NUM_THREADS (verified: the env var changes throughput; rayon's `build_global`
            // alone does not). Set it before the first op; also size rayon's pool for direct users.
            //
            // SAFETY: runs once during engine construction, before any worker thread is spawned and
            // before any op reads the environment, so there is no concurrent env access.
            unsafe {
                std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            }
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
        });
    }
}

/// Generation parameters applied to every language model the engine loads.
#[derive(Clone, Copy, Debug)]
pub struct GenParams {
    pub temp: f32,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub noise_clamp: Option<f32>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temp: defaults::TEMPERATURE,
            lsd_decode_steps: defaults::LSD_DECODE_STEPS,
            eos_threshold: defaults::EOS_THRESHOLD,
            noise_clamp: None,
        }
    }
}

struct ModelEntry {
    model: Arc<TTSModel>,
    last_used: u64,
}

/// A cached, multi-language pocket-tts engine. See module docs.
pub struct Engine {
    device: Device,
    params: GenParams,
    /// Max language models held at once; `0` means unbounded.
    capacity: usize,
    /// Monotonic LRU clock (no wall-clock needed).
    tick: u64,
    models: HashMap<String, ModelEntry>,
    /// `(voice_cache_key(spec), language)` -> voice-state for that model.
    voice_states: HashMap<(String, String), ModelState>,
}

impl Engine {
    /// Create an engine on `device` with default generation params and [`DEFAULT_CAPACITY`].
    pub fn new(device: Device) -> Self {
        init_rayon_pool();
        Self {
            device,
            params: GenParams::default(),
            capacity: DEFAULT_CAPACITY,
            tick: 0,
            models: HashMap::new(),
            voice_states: HashMap::new(),
        }
    }

    /// Set the model-cache capacity (`0` = unbounded). Builder-style.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Override generation parameters. Builder-style. Takes effect for models loaded *after* this
    /// call (already-resident models keep the params they were loaded with).
    pub fn with_params(mut self, params: GenParams) -> Self {
        self.params = params;
        self
    }

    /// The device models are loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Languages currently resident in the model cache.
    pub fn loaded_languages(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Number of language models currently resident.
    pub fn loaded_count(&self) -> usize {
        self.models.len()
    }

    /// Borrow (loading + caching if needed) the model for `language`, bumping its LRU recency.
    ///
    /// Returns an [`Arc`] so an in-flight generation keeps the model alive even if a later call
    /// evicts it from the cache.
    pub fn model(&mut self, language: &str) -> Result<Arc<TTSModel>> {
        self.tick += 1;
        let tick = self.tick;

        if let Some(entry) = self.models.get_mut(language) {
            entry.last_used = tick;
            return Ok(Arc::clone(&entry.model));
        }

        // Evict down to capacity-1 *before* loading, so peak residency stays at `capacity`.
        if self.capacity != 0 {
            while self.models.len() >= self.capacity {
                self.evict_lru();
            }
        }

        let model = Arc::new(TTSModel::load_with_params_device(
            language,
            self.params.temp,
            self.params.lsd_decode_steps,
            self.params.eos_threshold,
            self.params.noise_clamp,
            &self.device,
        )?);

        self.models.insert(
            language.to_string(),
            ModelEntry {
                model: Arc::clone(&model),
                last_used: tick,
            },
        );
        Ok(model)
    }

    /// Resolve `voice_spec` against `language`'s model, caching the result by `(voice, language)`.
    ///
    /// `voice_spec` is required — `None` is an error (pocket has no default voice). Subsequent calls
    /// with the same `(spec, language)` return a clone of the cached voice-state (cheap — candle
    /// tensors are reference-counted).
    pub fn voice_state(&mut self, language: &str, voice_spec: Option<&str>) -> Result<ModelState> {
        let spec = voice_spec.ok_or_else(|| {
            anyhow::anyhow!(
                "pocket: no default voice — pass a voice (a predefined name like `alba`, a \
                 .wav/.safetensors path, an hf:// URL, or base64 WAV)"
            )
        })?;
        let key = (voice_cache_key(spec), language.to_string());

        if let Some(state) = self.voice_states.get(&key) {
            return Ok(state.clone());
        }

        let model = self.model(language)?;
        let state = resolve_voice_spec(&model, spec, Some(language))?;
        self.voice_states.insert(key, state.clone());
        Ok(state)
    }

    /// Generate audio for `text` in `language` with `voice_spec`, routing to the cached model.
    ///
    /// `voice_spec` accepts anything [`crate::voice`] understands: a predefined name (`alba`, …),
    /// a `.wav`/`.safetensors` path, an `hf://` URL, or base64 WAV. `None` is an error — pocket has
    /// no default voice.
    pub fn generate(
        &mut self,
        text: &str,
        language: &str,
        voice_spec: Option<&str>,
    ) -> Result<Tensor> {
        let voice_state = self.voice_state(language, voice_spec)?;
        let model = self.model(language)?;
        model.generate(text, &voice_state)
    }

    /// Sample rate of `language`'s model (loads it if needed).
    pub fn sample_rate(&mut self, language: &str) -> Result<usize> {
        Ok(self.model(language)?.sample_rate)
    }

    /// Drop a language's model and its voice-states from the cache.
    pub fn unload(&mut self, language: &str) {
        self.models.remove(language);
        self.voice_states.retain(|(_, lang), _| lang != language);
    }

    /// Drop all cached models and voice-states.
    pub fn clear(&mut self) {
        self.models.clear();
        self.voice_states.clear();
    }

    /// Evict the least-recently-used model (and its voice-states).
    fn evict_lru(&mut self) {
        if let Some(victim) = self
            .models
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(lang, _)| lang.clone())
        {
            self.unload(&victim);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_tts_model() {
        let p = GenParams::default();
        assert_eq!(p.temp, defaults::TEMPERATURE);
        assert_eq!(p.lsd_decode_steps, defaults::LSD_DECODE_STEPS);
        assert_eq!(p.eos_threshold, defaults::EOS_THRESHOLD);
        assert!(p.noise_clamp.is_none());
    }

    #[test]
    fn new_engine_starts_empty() {
        let engine = Engine::new(Device::Cpu).with_capacity(3);
        assert_eq!(engine.loaded_count(), 0);
        assert!(engine.loaded_languages().is_empty());
        assert_eq!(engine.capacity, 3);
    }
}
