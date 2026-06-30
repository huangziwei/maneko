//! # mio — fast Japanese TTS engine (MioTTS-0.1B), port in progress
//!
//! A ChatML codec-LM (Falcon-H1 backbone) emits MioCodec **content tokens**, which the
//! **MioCodec-25Hz-24kHz** wave decoder turns into a 24 kHz waveform (FSQ dequant → Llama-3-style
//! transformers → iSTFT head). Built on [`tts_core`]. Ports the upstream `Aratako/MioCodec` +
//! `Aratako/MioTTS-0.1B` (codec MIT; AR Falcon-LLM License).
//!
//! Milestones:
//! - **M1 codec decode** — [`codec`]: content-token indices + 128-d global embedding → 24 kHz wav. ✅
//! - **M2 Falcon-H1 AR backbone** — [`falcon`]: token ids → hidden + logits (Mamba-2 ‖ GQA). ✅
//! - **M3 E2E** — [`engine`]: text → ChatML → AR greedy → `<|s_N|>` → codec → 24 kHz wav. ✅
//! - **M4 native voice cloning** — [`encoder`]: reference WAV → 128-d global (truncated WavLM +
//!   GlobalEncoder + torchaudio-exact resample). [`Mio::encode_ref_file`] clones any voice on-device. ✅
//! - **M5 text frontend + sampling + frontends** — [`text::normalize_text`], [`sampler`] (temperature /
//!   top-p, [`Mio::generate_with`]); wired into the `tts --engine mio` CLI and `maneko.Mio()`. ✅
//! - **M6 q8 + caching** — Falcon-H1 AR Linear weights → `Q8_0` GGUF ([`Mio::from_gguf`]); the fast
//!   Intel-CPU path (needs an AVX2 build). Incremental decode via KV / conv / SSM caches
//!   ([`FalconH1::prefill`] + `decode_step`) — O(T) vs the old O(T²) re-prefill (~4.6× on f32 CPU).
//!   Quantizer: `artifacts/mio-q8`. 🚧 (Intel RTF go/no-go bench pending)

// On x86_64, candle's Q8_0 GEMV (the whole point of the q8 path) falls back to a ~5× slower scalar
// loop without AVX2 — fail loudly so a slow build can't ship silently. Mirrors pocket.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
compile_error!(
    "mio-tts: building for x86_64 without AVX2 — candle's Q8_0 matmul would fall back to a ~5x \
     slower scalar path. Rebuild with RUSTFLAGS=\"-C target-cpu=native\" (or \
     \"-C target-feature=+avx2,+fma,+f16c\")."
);

pub mod codec;
pub mod config;
pub mod encoder;
pub mod engine;
pub mod falcon;
pub mod sampler;
pub mod text;
pub mod weights;

pub use codec::MioCodec;
pub use config::{CodecConfig, FalconH1Config};
pub use encoder::VoiceEncoder;
pub use engine::Mio;
pub use falcon::FalconH1;
pub use sampler::GenConfig;
pub use text::{normalize_text, MioTokenizer};
