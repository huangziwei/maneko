//! M5 validation: the full E2E pipeline (tokenize → sample → decode → silence-trim) vs an MLX-CPU
//! golden, with the reference latent and init noise injected (so this isolates the integration glue;
//! ref-audio encode and resample are validated/approximated elsewhere).
//!
//!   source .cache/mlxgolden/bin/activate
//!   HF_HOME=$HOME/.cache/huggingface SDKROOT=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk|tail -1) \
//!     DEVELOPER_DIR=/Library/Developer/CommandLineTools python ref/tools/dump_golden_e2e.py
//!   HF_HOME=$HOME/.cache/huggingface cargo test -p irodori --features accelerate \
//!     e2e -- --ignored --nocapture

use candle_core::Device;
use irodori::{Irodori, SamplerConfig};

// Must match TEXT in ref/tools/dump_golden_e2e.py.
const TEXT: &str = "こんにちは。今日はいい天気ですね。";

#[test]
#[ignore = "needs weights + golden (run ref/tools/dump_golden_e2e.py first)"]
fn e2e_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/e2e.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_e2e.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;

    let iro = Irodori::from_hf(&dev)?;
    let cfg = SamplerConfig { num_steps: 8, ..SamplerConfig::default() };
    let audio = iro.run_from_latent(
        TEXT,
        g.get("ref_latent").unwrap(),
        g.get("ref_mask").unwrap(),
        g.get("x_init").unwrap(),
        &cfg,
    )?;

    let golden = g.get("audio").unwrap();
    assert_eq!(audio.dims(), golden.dims(), "audio length mismatch (silence trim diverged?)");
    let diff = (&audio - golden)?.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("E2E audio diff: {diff:.3e}  samples {:?}", audio.dims());
    assert!(diff < 5e-3, "E2E waveform diverges from MLX golden: {diff}");
    Ok(())
}
