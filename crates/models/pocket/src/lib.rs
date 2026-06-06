// On x86_64, candle's Q8_0 CPU kernels are compile-time gated on `target_feature = "avx2"`
// (no runtime detection) and silently fall back to a scalar path that is ~5x slower. Fail the
// build loudly instead of shipping a crippled wheel. Build with AVX2, e.g.
//   RUSTFLAGS="-C target-cpu=native"   (or "-C target-feature=+avx2,+fma,+f16c")
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
compile_error!(
    "pocket: building for x86_64 without AVX2 — candle's Q8_0 matmul would fall back to a ~5x \
     slower scalar path. Rebuild with RUSTFLAGS=\"-C target-cpu=native\" (or \
     \"-C target-feature=+avx2,+fma,+f16c\")."
);

pub mod audio;
pub mod conditioners;
pub mod config;
pub mod engine;
pub mod models;
pub mod modules;
pub mod pause;
pub mod quantize;
pub mod tts_model;
pub mod voice;
pub mod voice_state;
pub mod weights;

pub use engine::{Engine, GenParams};
pub use pause::{ParsedText, PauseMarker, parse_text_with_pauses};
pub use quantize::{QuantizeConfig, QuantizedTensor};
// q8 quant moved to the shared core (works for both engines); re-export under the old `qweights`
// name so pocket's call sites (`crate::qweights::…`) and public API are unchanged.
pub use tts_core::quant::{self as qweights, QLinear, Vb};
pub use tts_model::TTSModel;
pub use voice::{
    PREDEFINED_VOICES, predefined_voice_hf_path, resolve_voice, resolve_voice_spec, voice_cache_key,
};
pub use voice_state::ModelState;
