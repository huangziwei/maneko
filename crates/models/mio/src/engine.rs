//! End-to-end MioTTS engine: text → ChatML → Falcon-H1 AR (speech tokens) → MioCodec → 24 kHz wav.
//!
//! Decoding is greedy ([`Mio::generate`]) or temperature / top-p sampling ([`Mio::generate_with`] +
//! [`crate::sampler::GenConfig`]): the prompt is prefilled once, then tokens decode incrementally via
//! the AR's KV / conv / SSM caches (O(T), vs the old re-prefill-every-step O(T²)). Normalize input
//! text at the frontend with [`crate::text::normalize_text`] — the model layer takes the prompt
//! verbatim. The reference is
//! **not** in the AR prompt — speaker identity is the 128-d `global` embedding handed to the codec.
//! Clone it on-device from any WAV with the WavLM [`VoiceEncoder`] ([`Mio::load_voice_encoder`] +
//! [`Mio::encode_ref_file`]).

use crate::codec::MioCodec;
use crate::encoder::VoiceEncoder;
use crate::falcon::FalconH1;
use crate::sampler::{GenConfig, Sampler};
use crate::text::MioTokenizer;
use anyhow::Context;
use candle_core::{Device, Tensor};

/// Cap candle's CPU worker pool to physical cores on x86_64 (logical/2) before the first matmul.
/// candle's `gemm`/BLAS backend sizes its pool from `RAYON_NUM_THREADS`; oversubscribing the
/// hyper-threaded siblings regresses these memory-bound batch-1 matmuls (measured on an i9-9880H:
/// 16 threads ≈ 10 tok/s vs 8 ≈ 13). An explicit `RAYON_NUM_THREADS` always wins. Mirrors pocket's
/// `init_rayon_pool`; called from the [`Mio`] constructors (before voice-encode / generation matmuls).
fn init_thread_cap() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            if std::env::var_os("RAYON_NUM_THREADS").is_some() {
                return;
            }
            let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            let threads = (logical / 2).max(1);
            // SAFETY: runs once during model construction, before candle spawns any worker thread
            // or reads the env, so there is no concurrent env access.
            unsafe {
                std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
            }
        });
    }
}

pub struct Mio {
    ar: FalconH1,
    codec: MioCodec,
    tok: MioTokenizer,
    voice_enc: Option<VoiceEncoder>,
    device: Device,
}

impl Mio {
    /// Reference loader: full-precision f32 AR (from `Aratako/MioTTS-0.1B`). Used by the parity
    /// goldens. For deployment prefer [`from_gguf`](Self::from_gguf) (q8 AR, the fast Intel path).
    pub fn from_hf(device: &Device) -> anyhow::Result<Self> {
        init_thread_cap();
        Ok(Self {
            ar: FalconH1::from_hf(device)?,
            codec: MioCodec::from_hf(device)?,
            tok: MioTokenizer::from_hf()?,
            voice_enc: None,
            device: device.clone(),
        })
    }

    /// Deployment loader: q8 AR from a GGUF (Linear weights `Q8_0`), with the f32 codec + tokenizer
    /// from their HF repos. The fast CPU path — on x86_64 candle's `Q8_0` GEMV needs an AVX2 build
    /// (see the crate-root guard). Resolve the GGUF with [`crate::weights::resolve_ar_q8`].
    pub fn from_gguf(device: &Device, ar_gguf: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        init_thread_cap();
        Ok(Self {
            ar: FalconH1::from_gguf(ar_gguf, device)?,
            codec: MioCodec::from_hf(device)?,
            tok: MioTokenizer::from_hf()?,
            voice_enc: None,
            device: device.clone(),
        })
    }

    /// The deployment default, picking the right AR precision for the build target:
    /// **q8 on x86_64** (candle's AVX2 `Q8_0` GEMV beats f32 there — the Intel-Mac path) and **f32
    /// elsewhere** (on arm64/NEON the f32 `gemm` is faster than q8 for these small matmuls, and
    /// exact). An explicit `gguf` always forces the q8 path (resolved via
    /// [`crate::weights::resolve_ar_q8`]). Load the voice encoder separately for cloning.
    pub fn load_default(device: &Device, gguf: Option<&std::path::Path>) -> anyhow::Result<Self> {
        if gguf.is_some() || cfg!(target_arch = "x86_64") {
            Self::from_gguf(device, crate::weights::resolve_ar_q8(gguf)?)
        } else {
            Self::from_hf(device)
        }
    }

    /// Load the WavLM voice encoder so [`encode_ref`](Self::encode_ref) /
    /// [`encode_ref_file`](Self::encode_ref_file) can clone a voice from a WAV. `wavlm_path` is the
    /// bundled WavLM weights (`mio_wavlm.safetensors`); the GlobalEncoder comes from the codec ckpt.
    pub fn load_voice_encoder(&mut self, wavlm_path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        self.voice_enc = Some(VoiceEncoder::from_hf(wavlm_path, &self.device)?);
        Ok(())
    }

    pub fn tokenizer(&self) -> &MioTokenizer {
        &self.tok
    }

    pub fn sample_rate(&self) -> usize {
        self.codec.sample_rate()
    }

    /// Greedy-generate the MioCodec content-token indices for `text` (stops on EOS or `max_new`).
    pub fn generate_tokens(&self, text: &str, max_new: usize) -> anyhow::Result<Vec<i64>> {
        self.generate_tokens_with(text, &GenConfig::greedy(max_new))
    }

    /// Generate the MioCodec content-token indices for `text` under `cfg` (greedy or temperature /
    /// top-p sampling; stops on EOS, a non-speech token, or `cfg.max_new`). `text` is the **already
    /// rendered** prompt input — normalize it with [`crate::text::normalize_text`] at the frontend.
    ///
    /// Prefills the prompt once, then decodes incrementally via the AR's KV / conv / SSM caches
    /// ([`FalconH1::prefill`](crate::FalconH1::prefill) + `decode_step`) — O(T) total, vs the old
    /// re-prefill-every-step O(T²).
    pub fn generate_tokens_with(&self, text: &str, cfg: &GenConfig) -> anyhow::Result<Vec<i64>> {
        let ids = self.tok.encode_prompt(text)?;
        let mut sampler = Sampler::new(cfg);
        let (mut logits, mut cache) = self.ar.prefill(&ids)?;
        let mut speech = Vec::new();
        for _ in 0..cfg.max_new {
            let next = sampler.sample(&logits)?;
            if MioTokenizer::is_eos(next) {
                break;
            }
            match MioTokenizer::speech_index(next) {
                Some(idx) => speech.push(idx),
                None => break, // unexpected non-speech token
            }
            logits = self.ar.decode_step(next, &mut cache)?;
        }
        Ok(speech)
    }

    /// Decode content-token indices to a waveform using a voice `global` embedding `(128,)`.
    pub fn decode_speech(&self, speech: &[i64], global: &Tensor) -> anyhow::Result<Tensor> {
        if speech.is_empty() {
            anyhow::bail!("no speech tokens to decode");
        }
        let n = speech.len();
        let indices = Tensor::from_vec(speech.to_vec(), (n,), &self.device)?;
        // 25 Hz tokens → 24 kHz: 960 samples/token (gives stft_length = 2·n, matching ×2 upsample).
        let target = n * (self.codec.sample_rate() / 25);
        Ok(self.codec.decode(&indices, global, target)?)
    }

    /// Full pipeline (greedy): `text` + voice `global` `(128,)` → waveform `(samples,)` @ 24 kHz.
    pub fn generate(&self, text: &str, global: &Tensor, max_new: usize) -> anyhow::Result<Tensor> {
        self.generate_with(text, global, &GenConfig::greedy(max_new))
    }

    /// Full pipeline under `cfg` (greedy or temperature / top-p sampling): `text` + voice `global`
    /// `(128,)` → waveform `(samples,)` @ 24 kHz. Normalize `text` at the frontend
    /// ([`crate::text::normalize_text`]); the model layer takes the prompt input verbatim.
    pub fn generate_with(&self, text: &str, global: &Tensor, cfg: &GenConfig) -> anyhow::Result<Tensor> {
        let speech = self.generate_tokens_with(text, cfg)?;
        self.decode_speech(&speech, global)
    }

    /// Clone a voice from a **24 kHz mono** waveform (`(samples,)` or `(1, samples)`) → 128-d `global`
    /// embedding `(128,)` ready for [`generate`](Self::generate). Requires [`load_voice_encoder`].
    pub fn encode_ref(&self, wav24k: &Tensor) -> anyhow::Result<Tensor> {
        let ve = self.voice_enc.as_ref().context("voice encoder not loaded; call load_voice_encoder first")?;
        Ok(ve.encode_ref(wav24k)?.squeeze(0)?) // (1, 128) -> (128,)
    }

    /// Clone a voice from a WAV file (any rate/channels) → 128-d `global` embedding `(128,)`. Reads,
    /// down-mixes to mono, and resamples to 24 kHz before [`encode_ref`](Self::encode_ref).
    pub fn encode_ref_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<Tensor> {
        let (audio, sr) = tts_core::audio::read_wav(path)?; // (channels, samples)
        let mono = audio.mean(0)?.unsqueeze(0)?; // (1, samples)
        let wav24k = tts_core::audio::resample(&mono, sr, 24_000)?;
        self.encode_ref(&wav24k.to_device(&self.device)?)
    }
}
