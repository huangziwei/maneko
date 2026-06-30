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

pub mod codec;
pub mod config;
pub mod encoder;
pub mod engine;
pub mod falcon;
pub mod text;
pub mod weights;

pub use codec::MioCodec;
pub use config::{CodecConfig, FalconH1Config};
pub use encoder::VoiceEncoder;
pub use engine::Mio;
pub use falcon::FalconH1;
pub use text::MioTokenizer;
