//! Python bindings for the maneko TTS engines — importable as `maneko`.
//!
//! ```python
//! import maneko
//! p = maneko.Pocket()
//! audio = p.generate("Hello world.", language="b6369a24")        # list[float], 24 kHz
//! maneko.save_wav("out.wav", audio, p.sample_rate("b6369a24"))
//!
//! i = maneko.Irodori()
//! jp = i.generate("こんにちは。", voice="ref.wav", seconds=4, steps=40)  # 48 kHz
//! maneko.save_wav("jp.wav", jp, i.sample_rate)
//! ```
//!
//! Weights resolve from `HF_HOME` — point it at the cache holding that engine's repos.

use candle_core::{Device, Tensor};
use pyo3::prelude::*;

fn rt_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}

fn flatten_audio(t: Tensor) -> PyResult<Vec<f32>> {
    t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(rt_err)
}

/// pocket-tts: multilingual (en/de/es/fr/it/pt), 24 kHz. Loads/caches one model per language.
#[pyclass]
struct Pocket {
    engine: pocket::Engine,
}

#[pymethods]
impl Pocket {
    #[new]
    fn new() -> Self {
        Self { engine: pocket::Engine::new(Device::Cpu) }
    }

    /// Generate speech → mono `list[float]` at 24 kHz.
    ///
    /// `language` is a config stem (`english`, `german`, `french_24l`, `b6369a24`, …).
    /// `voice` is a predefined name / `.wav` / `.safetensors` / `hf://` / base64 (default: stock).
    #[pyo3(signature = (text, language="b6369a24", voice=None))]
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

#[pymethods]
impl Irodori {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self { inner: irodori::Irodori::from_hf(&Device::Cpu).map_err(rt_err)? })
    }

    /// Generate Japanese speech → mono `list[float]` at 48 kHz.
    ///
    /// `voice` is a reference `.wav` to clone (default: none). `seconds` sets the duration
    /// (default: model fallback, trimmed to silence). `steps` is the diffusion step count.
    #[pyo3(signature = (text, voice=None, seconds=None, steps=40))]
    fn generate(
        &self,
        text: &str,
        voice: Option<&str>,
        seconds: Option<f64>,
        steps: usize,
    ) -> PyResult<Vec<f32>> {
        let opts = irodori::GenerateOptions {
            seconds,
            sampler: irodori::SamplerConfig { num_steps: steps, ..irodori::SamplerConfig::default() },
        };
        let audio = self.inner.generate(text, voice, &opts).map_err(rt_err)?;
        flatten_audio(audio)
    }

    /// Sample rate (Hz) — always 48000.
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
    m.add_function(wrap_pyfunction!(save_wav, m)?)?;
    Ok(())
}
