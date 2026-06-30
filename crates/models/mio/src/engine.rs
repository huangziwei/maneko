//! End-to-end MioTTS engine: text → ChatML → Falcon-H1 AR (speech tokens) → MioCodec → 24 kHz wav.
//!
//! Generation is greedy with re-prefill each step (simple + correct; KV/conv/SSM caching for O(T)
//! decode is M6 perf work). The reference is **not** in the AR prompt — speaker identity is the
//! 128-d `global` embedding handed to the codec. Clone it on-device from any WAV with the WavLM
//! [`VoiceEncoder`] ([`Mio::load_voice_encoder`] + [`Mio::encode_ref_file`]).

use crate::codec::MioCodec;
use crate::encoder::VoiceEncoder;
use crate::falcon::FalconH1;
use crate::text::MioTokenizer;
use anyhow::Context;
use candle_core::{Device, IndexOp, Result, Tensor};

pub struct Mio {
    ar: FalconH1,
    codec: MioCodec,
    tok: MioTokenizer,
    voice_enc: Option<VoiceEncoder>,
    device: Device,
}

impl Mio {
    pub fn from_hf(device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            ar: FalconH1::from_hf(device)?,
            codec: MioCodec::from_hf(device)?,
            tok: MioTokenizer::from_hf()?,
            voice_enc: None,
            device: device.clone(),
        })
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

    /// Last-position logits `(vocab,)` for a token sequence (full re-prefill).
    fn last_logits(&self, ids: &[u32]) -> Result<Tensor> {
        let t = ids.len();
        let ids_t = Tensor::from_vec(ids.to_vec(), (1, t), &self.device)?;
        let (_hidden, logits) = self.ar.forward(&ids_t)?; // (1, T, vocab)
        logits.i((0, t - 1))
    }

    /// Greedy-generate the MioCodec content-token indices for `text` (stops on EOS or `max_new`).
    pub fn generate_tokens(&self, text: &str, max_new: usize) -> anyhow::Result<Vec<i64>> {
        let mut ids = self.tok.encode_prompt(text)?;
        let mut speech = Vec::new();
        for _ in 0..max_new {
            let next = self.last_logits(&ids)?.argmax(0)?.to_scalar::<u32>()?;
            if MioTokenizer::is_eos(next) {
                break;
            }
            match MioTokenizer::speech_index(next) {
                Some(idx) => {
                    speech.push(idx);
                    ids.push(next);
                }
                None => break, // unexpected non-speech token
            }
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

    /// Full pipeline: `text` + voice `global` `(128,)` → waveform `(samples,)` @ 24 kHz.
    pub fn generate(&self, text: &str, global: &Tensor, max_new: usize) -> anyhow::Result<Tensor> {
        let speech = self.generate_tokens(text, max_new)?;
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
