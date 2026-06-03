//! M3 validation: one-step DiT velocity (`forward_with_conditions`) parity vs an MLX-CPU golden.
//!
//!   source .cache/mlxgolden/bin/activate
//!   HF_HOME=$PWD/.cache/huggingface SDKROOT=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk|tail -1) \
//!     DEVELOPER_DIR=/Library/Developer/CommandLineTools python ref/tools/dump_golden_dit.py
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p irodori --features accelerate \
//!     dit_vpred -- --ignored --nocapture

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use irodori::weights::hf_file;
use irodori::{DitConfig, IrodoriDiT};

const DIT_REPO: &str = "Aratako/Irodori-TTS-500M-v2";

#[test]
#[ignore = "needs DiT weights + golden (run ref/tools/dump_golden_dit.py first)"]
fn dit_vpred_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/dit_vpred.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_dit.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;

    let dit_path = hf_file(DIT_REPO, "model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, &dev)? };
    let dit = IrodoriDiT::load(vb, DitConfig::v2(), 4096)?;

    let text_mask = g.get("text_mask").unwrap().clone();
    let ref_mask = g.get("ref_mask").unwrap().clone();
    let (text_state, speaker_state) = dit.encode_conditions(
        g.get("input_ids").unwrap(),
        &text_mask,
        g.get("ref_latent").unwrap(),
        &ref_mask,
    )?;
    let kv = dit.build_kv_cache(&text_state, &speaker_state)?;
    let v_pred = dit.forward_with_conditions(
        g.get("x_t").unwrap(),
        g.get("t").unwrap(),
        &text_mask,
        &ref_mask,
        &kv,
        0,
    )?;

    let golden = g.get("v_pred").unwrap();
    assert_eq!(v_pred.dims(), golden.dims(), "v_pred shape");
    let diff = (&v_pred - golden)?.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("v_pred diff: {diff:.3e}  shape {:?}", v_pred.dims());
    assert!(diff < 1e-3, "v_pred diverges from MLX golden: {diff}");
    Ok(())
}
