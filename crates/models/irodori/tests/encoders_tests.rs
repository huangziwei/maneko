//! M2 validation: text + speaker encoder parity vs an MLX-CPU golden.
//!
//!   source .cache/mlxgolden/bin/activate
//!   HF_HOME=$PWD/.cache/huggingface SDKROOT=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk|tail -1) \
//!     DEVELOPER_DIR=/Library/Developer/CommandLineTools python ref/tools/dump_golden_encoders.py
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p irodori --features accelerate \
//!     encoders -- --ignored --nocapture

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use irodori::weights::hf_file;
use irodori::{DitConfig, Encoders};

const DIT_REPO: &str = "Aratako/Irodori-TTS-500M-v2";

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
}

#[test]
#[ignore = "needs DiT weights + golden (run ref/tools/dump_golden_encoders.py first)"]
fn encoders_match_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/encoders.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_encoders.py");
    let golden = candle_core::safetensors::load(&golden_path, &dev)?;

    let dit_path = hf_file(DIT_REPO, "model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, &dev)? };
    let enc = Encoders::load(vb, &DitConfig::v2(), 4096)?;

    let input_ids = golden.get("input_ids").unwrap().clone(); // (1,S) u32
    let text_mask = golden.get("text_mask").unwrap().clone(); // (1,S) f32
    let ref_latent = golden.get("ref_latent").unwrap().clone(); // (1,T,32) f32
    let ref_mask = golden.get("ref_mask").unwrap().clone(); // (1,T) f32

    let text_state = enc.encode_text(&input_ids, &text_mask)?;
    let dt = max_abs_diff(&text_state, golden.get("text_state").unwrap())?;
    eprintln!("text_state diff: {dt:.3e}  shape {:?}", text_state.dims());
    assert!(dt < 1e-3, "text_state diverges from MLX golden: {dt}");

    let speaker_state = enc.encode_speaker(&ref_latent, &ref_mask)?;
    let ds = max_abs_diff(&speaker_state, golden.get("speaker_state").unwrap())?;
    eprintln!("speaker_state diff: {ds:.3e}  shape {:?}", speaker_state.dims());
    assert!(ds < 1e-3, "speaker_state diverges from MLX golden: {ds}");
    Ok(())
}
