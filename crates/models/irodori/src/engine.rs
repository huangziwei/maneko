//! End-to-end Irodori engine: text (+ optional reference voice) → 48 kHz waveform.
//!
//! Wires the validated pieces together (mirrors `irodori_tts.py::Model.generate`):
//! tokenize → encode conditions → RF/CFG sampler → DACVAE decode → trailing-silence trim. The
//! reference voice is DACVAE-encoded to a latent (voice cloning); with no reference, a zero latent
//! is used. Long-form chunked decode (the MLX `chunk_size`/crossfade path) is not yet ported —
//! decode is single-pass, which matches the chunked path for clips up to one chunk.

use crate::config::{DacvaeConfig, DitConfig};
use crate::sampler::{sample_euler_cfg, SamplerConfig};
use crate::weights::hf_file;
use crate::{Dacvae, IrodoriDiT, IrodoriTokenizer};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

const DIT_REPO_V2: &str = "Aratako/Irodori-TTS-500M-v2";
const DIT_REPO_V3: &str = "Aratako/Irodori-TTS-500M-v3";

/// Detect trailing silence in a generated latent `(T, D)`: the first window (size 20) whose values
/// have `std < 0.05` and `|mean| < 0.1`, else `T`. Verbatim port of `_find_silence_point`.
fn find_silence_point(latent: &Tensor) -> Result<usize> {
    let (t, d) = latent.dims2()?;
    let window = 20usize;
    let zeros = Tensor::zeros((window, d), latent.dtype(), latent.device())?;
    let padded = Tensor::cat(&[latent, &zeros], 0)?;
    for i in 0..t {
        let w = padded.narrow(0, i, window)?;
        let mean = w.mean_all()?.to_scalar::<f32>()?;
        let m2 = w.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let std = (m2 - mean * mean).max(0.0).sqrt();
        if std < 0.05 && mean.abs() < 0.1 {
            return Ok(i);
        }
    }
    Ok(t)
}

/// Generation knobs. With **v3**, when `seconds` is `None` the duration predictor sets the length
/// (predicted frames × `duration_scale`, clamped to `[min_seconds, max_seconds]`). With v2 (or any
/// model lacking the predictor), `seconds` sets the length, else the 30 s fallback (750 frames). An
/// explicit `seconds` always wins and is itself clamped to `[min_seconds, max_seconds]`.
pub struct GenerateOptions {
    pub seconds: Option<f64>,
    pub sampler: SamplerConfig,
    /// v3: scale the predicted duration (1.0 = as predicted; >1 = longer/slower speech).
    pub duration_scale: f64,
    /// Clamp the (predicted or explicit) duration to this range, in seconds.
    pub min_seconds: f64,
    pub max_seconds: f64,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            seconds: None,
            sampler: SamplerConfig::default(),
            duration_scale: 1.0,
            min_seconds: 0.5,
            max_seconds: 30.0,
        }
    }
}

/// The Irodori TTS engine: DiT + DACVAE + llm-jp tokenizer.
pub struct Irodori {
    dit: IrodoriDiT,
    dacvae: Dacvae,
    tokenizer: IrodoriTokenizer,
    device: Device,
}

impl Irodori {
    /// Load the **v3** model (DiT + integrated Duration Predictor) — the default Irodori. With v3,
    /// `generate` predicts the output length from text + speaker when `seconds` is `None` (instead
    /// of the v2 30 s fallback).
    pub fn from_hf(device: &Device) -> anyhow::Result<Self> {
        Self::load_repo(device, DIT_REPO_V3, DitConfig::v3())
    }

    /// Load the v2 model (no duration predictor — `generate` uses `seconds`, else the 30 s
    /// fallback). Kept for the MLX parity goldens and back-compat.
    pub fn from_hf_v2(device: &Device) -> anyhow::Result<Self> {
        Self::load_repo(device, DIT_REPO_V2, DitConfig::v2())
    }

    /// Load a **q8 GGUF** DiT (v3): the DiT's Linear weights are Q8_0 via `Vb::from_gguf`; the
    /// DACVAE and llm-jp tokenizer stay full precision. ~4× smaller DiT than f32.
    pub fn from_gguf(device: &Device, gguf_path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let vb = tts_core::Vb::from_gguf(gguf_path, device)?;
        Self::from_dit_vb(device, vb, DitConfig::v3())
    }

    fn load_repo(device: &Device, repo: &str, cfg: DitConfig) -> anyhow::Result<Self> {
        let dit_path = hf_file(repo, "model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, device)? };
        Self::from_dit_vb(device, tts_core::Vb::Full(vb), cfg)
    }

    /// Assemble the engine from a loaded DiT weight source (f32 `Vb::Full` or q8 `Vb::Quant`) plus
    /// the full-precision DACVAE and the llm-jp tokenizer (shared by v2/v3).
    fn from_dit_vb(device: &Device, dit_vb: tts_core::Vb, cfg: DitConfig) -> anyhow::Result<Self> {
        let dit = IrodoriDiT::load(dit_vb, cfg, 8192)?;
        let dacvae = Dacvae::from_hf(device)?;
        let tokenizer = IrodoriTokenizer::v2(device)?; // llm-jp tokenizer is shared by v2 and v3
        Ok(Self { dit, dacvae, tokenizer, device: device.clone() })
    }

    pub fn sample_rate(&self) -> usize {
        self.dacvae.sample_rate()
    }

    /// Core pipeline from a precomputed reference latent and initial noise (no audio I/O, no RNG):
    /// tokenize → sample → decode → silence-trim. Returns the waveform `(1, samples)`.
    pub fn run_from_latent(
        &self,
        text: &str,
        ref_latent: &Tensor,
        ref_mask: &Tensor,
        x_init: &Tensor,
        sampler: &SamplerConfig,
    ) -> anyhow::Result<Tensor> {
        let (ids, text_mask) = self.tokenizer.encode_tensor(text, &self.device)?;
        let latent = sample_euler_cfg(&self.dit, &ids, &text_mask, ref_latent, ref_mask, x_init, sampler)?;
        let steps = latent.dim(1)?;
        let hop = self.dacvae.hop_length();

        // Chunked decode (chunk_size=50, overlap=4) — matches mlx-audio's generate and bounds the
        // conv-transpose memory; for ≤50-frame clips this is single-pass.
        let audio = self.dacvae.decode_chunked(&latent.transpose(1, 2)?.contiguous()?, 50, 4)?; // (1,1,L)
        let silence_t = find_silence_point(&latent.get(0)?)?;
        let trim = (silence_t * hop).min(steps * hop).min(audio.dim(2)?);
        Ok(audio.narrow(2, 0, trim)?.reshape((1, trim))?)
    }

    /// Generate speech for `text` in the voice of an optional reference WAV (any sample rate;
    /// resampled to 48 kHz and DACVAE-encoded). Draws fresh Gaussian noise. Returns `(1, samples)`.
    pub fn generate(
        &self,
        text: &str,
        ref_wav: Option<&str>,
        opts: &GenerateOptions,
    ) -> anyhow::Result<Tensor> {
        let has_speaker = ref_wav.is_some();
        let (ref_latent, ref_mask) = match ref_wav {
            Some(path) => {
                let (audio, sr) = tts_core::audio::read_wav(path)?;
                let mono = if audio.dim(0)? > 1 { audio.mean_keepdim(0)? } else { audio };
                let mono = tts_core::audio::resample(&mono, sr, self.sample_rate() as u32)?;
                let m = mono.dim(1)?;
                // read_wav/resample produce a CPU tensor; move it onto the engine's device
                // (else conv1d weights-on-Metal vs input-on-CPU → device mismatch).
                let audio_in = mono.reshape((1, 1, m))?.to_device(&self.device)?;
                let ref_latent = self.dacvae.encode(&audio_in)?; // (1,T,32)
                let t = ref_latent.dim(1)?;
                (ref_latent, Tensor::ones((1, t), DType::F32, &self.device)?)
            }
            None => {
                let d = DacvaeConfig::v2().codebook_dim;
                (
                    Tensor::zeros((1, 1, d), DType::F32, &self.device)?,
                    Tensor::zeros((1, 1), DType::F32, &self.device)?,
                )
            }
        };

        let hop = self.dacvae.hop_length() as f64;
        let sr = self.sample_rate() as f64;
        let secs_to_frames = |s: f64| ((s * sr / hop).ceil() as usize).max(1);
        let min_f = secs_to_frames(opts.min_seconds);
        let max_f = ((opts.max_seconds * sr / hop).floor() as usize).max(1);
        let steps = if let Some(s) = opts.seconds {
            // Explicit duration (v2 + v3): clamp to range, then convert to frames.
            secs_to_frames(s.clamp(opts.min_seconds, opts.max_seconds))
        } else {
            // No explicit duration: v3 predicts it from text + speaker; v2 falls back to 30 s.
            let (ids, text_mask) = self.tokenizer.encode_tensor(text, &self.device)?;
            match self
                .dit
                .predict_duration_frames(&ids, &text_mask, &ref_latent, &ref_mask, has_speaker)?
            {
                Some(frames) => ((frames * opts.duration_scale).round() as usize).clamp(min_f, max_f),
                None => 750,
            }
        };
        let dim = DacvaeConfig::v2().codebook_dim;
        let x_init = Tensor::randn(0f32, 1f32, (1, steps, dim), &self.device)?;
        self.run_from_latent(text, &ref_latent, &ref_mask, &x_init, &opts.sampler)
    }
}
