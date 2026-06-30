//! Python bindings for the maneko TTS engines — importable as `maneko`.
//!
//! ```python
//! import maneko
//! p = maneko.Pocket()                 # device="cpu" (default) or "metal" (wheel built --features metal)
//! audio = p.generate("Hello world.", language="english_2026-04", voice="alba")  # list[float], 24 kHz
//! maneko.save_wav("out.wav", audio, p.sample_rate("english_2026-04"))
//!
//! i = maneko.Irodori(device="metal")            # GPU; default "cpu"
//! jp = i.generate("こんにちは。", voice="ref.wav")  # one-shot (encodes the ref each call)
//! ref = i.encode_ref("ref.wav")                 # …or encode once + reuse across a book:
//! jp = i.generate_with_ref("こんにちは。", ref)     # skips the per-call DACVAE-encode
//! maneko.save_wav("jp.wav", jp, i.sample_rate)
//!
//! m = maneko.Mio()                              # Japanese, 24 kHz; clones any reference WAV on-device
//! jp = m.generate("こんにちは。", voice="ref.wav")  # temperature 0.8 / top_p 1.0 by default
//! maneko.save_wav("jp.wav", jp, m.sample_rate)  # 24000
//! ```
//!
//! Weights resolve from `HF_HOME` (or pull from the public `zwaiwng/maneko`). Build with
//! `--features accelerate,metal` for one wheel that does fast CPU **and** GPU (pick via `device=`).

use candle_core::{Device, Tensor};
use pyo3::prelude::*;

fn rt_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}

fn flatten_audio(t: Tensor) -> PyResult<Vec<f32>> {
    t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(rt_err)
}

/// Map a device name to a candle `Device`. `"metal"` requires a wheel built `--features metal`
/// (mirrors the CLI's `--metal`). Compute stays on this device; only WAV I/O is CPU.
fn select_device(name: &str) -> PyResult<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(rt_err)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err(rt_err("device=\"metal\" requires a wheel built with --features metal"))
            }
        }
        other => Err(rt_err(format!("unknown device {other:?} (expected \"cpu\" or \"metal\")"))),
    }
}

/// Irodori generate options from the Python kwargs (shared by `generate` / `generate_with_ref`).
fn gen_opts(seconds: Option<f64>, steps: usize) -> irodori::GenerateOptions {
    irodori::GenerateOptions {
        seconds,
        sampler: irodori::SamplerConfig { num_steps: steps, ..irodori::SamplerConfig::default() },
        ..Default::default()
    }
}

/// pocket-tts: multilingual (en/de/es/fr/it/pt), 24 kHz. Loads/caches one model per language.
#[pyclass]
struct Pocket {
    engine: pocket::Engine,
}

#[pymethods]
impl Pocket {
    /// `device` is `"cpu"` (default) or `"metal"` (requires a wheel built `--features metal`).
    #[new]
    #[pyo3(signature = (device="cpu"))]
    fn new(device: &str) -> PyResult<Self> {
        Ok(Self { engine: pocket::Engine::new(select_device(device)?) })
    }

    /// Generate speech → mono `list[float]` at 24 kHz.
    ///
    /// `language` is a config stem (`english_2026-04`, `german`, `french_24l`, …).
    /// `voice` is a predefined name / `.wav` / `.safetensors` / `hf://` / base64 (required; `None` errors).
    #[pyo3(signature = (text, language="english_2026-04", voice=None))]
    fn generate(&mut self, text: &str, language: &str, voice: Option<&str>) -> PyResult<Vec<f32>> {
        let audio = self.engine.generate(text, language, voice).map_err(rt_err)?;
        flatten_audio(audio)
    }

    /// Sample rate (Hz) of `language`'s model (loads it if needed).
    fn sample_rate(&mut self, language: &str) -> PyResult<usize> {
        self.engine.sample_rate(language).map_err(rt_err)
    }
}

/// Irodori: Japanese, 48 kHz, flow-matching diffusion with voice cloning.
#[pyclass]
struct Irodori {
    inner: irodori::Irodori,
}

/// A reference voice encoded once (its DACVAE latent) — reuse across many clips in one voice via
/// `Irodori.generate_with_ref`. For a book (one narrator, many chunks) this skips the per-call
/// DACVAE-encode. Obtain it from `Irodori.encode_ref(...)`; not constructible directly.
#[pyclass]
struct RefVoice {
    inner: irodori::RefVoice,
}

#[pymethods]
impl Irodori {
    /// `device` is `"cpu"` (default) or `"metal"` (requires a wheel built `--features metal`).
    #[new]
    #[pyo3(signature = (device="cpu"))]
    fn new(device: &str) -> PyResult<Self> {
        let dev = select_device(device)?;
        Ok(Self { inner: irodori::Irodori::from_hf(&dev).map_err(rt_err)? })
    }

    /// Generate Japanese speech → mono `list[float]` at 48 kHz.
    ///
    /// `voice` is a reference `.wav` to clone (default: none). `seconds` sets the duration
    /// (default: model fallback, trimmed to silence). `steps` is the diffusion step count.
    #[pyo3(signature = (text, voice=None, seconds=None, steps=8))]
    fn generate(
        &self,
        text: &str,
        voice: Option<&str>,
        seconds: Option<f64>,
        steps: usize,
    ) -> PyResult<Vec<f32>> {
        let audio = self.inner.generate(text, voice, &gen_opts(seconds, steps)).map_err(rt_err)?;
        flatten_audio(audio)
    }

    /// Encode a reference voice **once** → a [`RefVoice`] to reuse across clips (one narrator, many
    /// chunks) via `generate_with_ref`, skipping the per-call DACVAE-encode. `voice` is a reference
    /// `.wav` (or `None` for the speaker-less default).
    #[pyo3(signature = (voice=None))]
    fn encode_ref(&self, voice: Option<&str>) -> PyResult<RefVoice> {
        Ok(RefVoice { inner: self.inner.encode_ref(voice).map_err(rt_err)? })
    }

    /// Generate using a pre-encoded [`RefVoice`] (from `encode_ref`) — like `generate`, but reuses the
    /// encoded voice instead of re-encoding it each call. `seconds`/`steps` as in `generate`.
    #[pyo3(signature = (text, voice, seconds=None, steps=8))]
    fn generate_with_ref(
        &self,
        text: &str,
        voice: &RefVoice,
        seconds: Option<f64>,
        steps: usize,
    ) -> PyResult<Vec<f32>> {
        let audio = self
            .inner
            .generate_with_ref(text, &voice.inner, &gen_opts(seconds, steps))
            .map_err(rt_err)?;
        flatten_audio(audio)
    }

    /// Sample rate (Hz) — always 48000.
    #[getter]
    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}

/// MioTTS: Japanese, 24 kHz, Falcon-H1 codec-LM. Clones the speaker on-device from a reference WAV
/// (truncated WavLM + GlobalEncoder), so every voice comes from a `.wav` — no presets needed.
#[pyclass]
struct Mio {
    inner: mio_tts::Mio,
}

/// A reference voice cloned to its 128-d speaker embedding — reuse across many clips in one voice via
/// `Mio.generate_with_ref` (skips the per-call WavLM encode). From `Mio.encode_ref(...)`; opaque.
#[pyclass]
struct MioVoice {
    global: Tensor,
}

#[pymethods]
impl Mio {
    /// `device` is `"cpu"` (default) or `"metal"` (requires a wheel built `--features metal`).
    /// The AR is q8 on x86_64 (the fast Intel-CPU path) and f32 on arm64; `gguf` forces a local q8
    /// GGUF. `wavlm` overrides the voice-encoder bundle (else `$MIO_WAVLM` / local / maneko HF).
    #[new]
    #[pyo3(signature = (device="cpu", wavlm=None, gguf=None))]
    fn new(device: &str, wavlm: Option<&str>, gguf: Option<&str>) -> PyResult<Self> {
        let dev = select_device(device)?;
        let mut inner =
            mio_tts::Mio::load_default(&dev, gguf.map(std::path::Path::new)).map_err(rt_err)?;
        let wavlm = mio_tts::weights::resolve_wavlm(wavlm.map(std::path::Path::new)).map_err(rt_err)?;
        inner.load_voice_encoder(wavlm).map_err(rt_err)?;
        Ok(Self { inner })
    }

    /// Generate Japanese speech → mono `list[float]` at 24 kHz. `voice` is a reference `.wav` to
    /// clone (cloned each call; use `encode_ref` + `generate_with_ref` to reuse it). Sampling
    /// defaults match the MioTTS server (`temperature 0.8`, `top_p 1.0`); `temperature 0` ⇒ greedy.
    #[pyo3(signature = (text, voice, temperature=0.8, top_p=1.0, max_tokens=700, seed=None))]
    fn generate(
        &self,
        text: &str,
        voice: &str,
        temperature: f32,
        top_p: f32,
        max_tokens: usize,
        seed: Option<u64>,
    ) -> PyResult<Vec<f32>> {
        let v = self.encode_ref(voice)?;
        self.generate_with_ref(text, &v, temperature, top_p, max_tokens, seed)
    }

    /// Clone a reference `.wav` **once** → a [`MioVoice`] to reuse across clips (one speaker, many
    /// chunks) via `generate_with_ref`, skipping the per-call WavLM encode.
    #[pyo3(signature = (voice))]
    fn encode_ref(&self, voice: &str) -> PyResult<MioVoice> {
        Ok(MioVoice { global: self.inner.encode_ref_file(voice).map_err(rt_err)? })
    }

    /// Generate using a pre-cloned [`MioVoice`] (from `encode_ref`) — like `generate`, but reuses the
    /// embedding instead of re-cloning it each call. Sampling args as in `generate`.
    #[pyo3(signature = (text, voice, temperature=0.8, top_p=1.0, max_tokens=700, seed=None))]
    fn generate_with_ref(
        &self,
        text: &str,
        voice: &MioVoice,
        temperature: f32,
        top_p: f32,
        max_tokens: usize,
        seed: Option<u64>,
    ) -> PyResult<Vec<f32>> {
        let normalized = mio_tts::normalize_text(text);
        let cfg = mio_tts::GenConfig { max_new: max_tokens, temperature, top_p, seed };
        let audio = self.inner.generate_with(&normalized, &voice.global, &cfg).map_err(rt_err)?;
        flatten_audio(audio)
    }

    /// Sample rate (Hz) — always 24000.
    #[getter]
    fn sample_rate(&self) -> usize {
        self.inner.sample_rate()
    }
}

/// Save mono float samples to a 16-bit PCM WAV.
#[pyfunction]
fn save_wav(path: &str, samples: Vec<f32>, sample_rate: u32) -> PyResult<()> {
    let n = samples.len();
    let t = Tensor::from_vec(samples, (1, n), &Device::Cpu).map_err(rt_err)?;
    tts_core::audio::write_wav(path, &t, sample_rate).map_err(rt_err)
}

#[pymodule]
fn maneko(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pocket>()?;
    m.add_class::<Irodori>()?;
    m.add_class::<RefVoice>()?;
    m.add_class::<Mio>()?;
    m.add_class::<MioVoice>()?;
    m.add_function(wrap_pyfunction!(save_wav, m)?)?;
    Ok(())
}
