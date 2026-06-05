//! v3 validation: duration-predictor frame count parity vs an MLX-CPU golden.
//!
//! The v3 checkpoint adds `duration_predictor.*`; this checks Rust's
//! `IrodoriDiT::predict_duration_frames` (encode_conditions → token-sum SwiGLU → masked softplus
//! sum) reproduces mlx-audio's `predict_duration_log_frames` for both the speaker and the
//! null-speaker path, on the same fixed conditions as the DiT golden.
//!
//!   source .cache/mlxgolden/bin/activate
//!   SDKROOT=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk|tail -1) \
//!     DEVELOPER_DIR=/Library/Developer/CommandLineTools python ref/tools/dump_golden_duration.py
//!   HF_HOME=$HOME/.cache/huggingface cargo test -p irodori --features accelerate \
//!     duration -- --ignored --nocapture

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use irodori::weights::hf_file;
use irodori::{DitConfig, IrodoriDiT};

const DIT_REPO_V3: &str = "Aratako/Irodori-TTS-500M-v3";

#[test]
#[ignore = "needs v3 DiT weights + golden (run ref/tools/dump_golden_duration.py first)"]
fn duration_frames_match_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/duration.safetensors");
    assert!(
        golden_path.exists(),
        "missing golden at {golden_path:?}; run dump_golden_duration.py"
    );
    let g = candle_core::safetensors::load(&golden_path, &dev)?;

    let dit_path = hf_file(DIT_REPO_V3, "model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, &dev)? };
    let dit = IrodoriDiT::load(vb, DitConfig::v3(), 4096)?;
    assert!(dit.has_duration_predictor(), "v3 DiT must carry the duration predictor");

    let ids = g.get("input_ids").unwrap();
    let text_mask = g.get("text_mask").unwrap();
    let ref_latent = g.get("ref_latent").unwrap();
    let ref_mask = g.get("ref_mask").unwrap();

    let golden_spk = g.get("frames_spk").unwrap().to_vec1::<f32>()?[0] as f64;
    let golden_nospk = g.get("frames_nospk").unwrap().to_vec1::<f32>()?[0] as f64;

    let frames_spk = dit
        .predict_duration_frames(ids, text_mask, ref_latent, ref_mask, true)?
        .expect("v3 predicts a duration");
    let frames_nospk = dit
        .predict_duration_frames(ids, text_mask, ref_latent, ref_mask, false)?
        .expect("v3 predicts a duration");

    let rel = |r: f64, golden: f64| (r - golden).abs() / golden.abs().max(1.0);
    eprintln!(
        "duration frames  spk: rust={frames_spk:.4} golden={golden_spk:.4} (rel {:.2e})  \
         nospk: rust={frames_nospk:.4} golden={golden_nospk:.4} (rel {:.2e})",
        rel(frames_spk, golden_spk),
        rel(frames_nospk, golden_nospk),
    );
    assert!(rel(frames_spk, golden_spk) < 1e-3, "speaker-path frames diverge from MLX golden");
    assert!(rel(frames_nospk, golden_nospk) < 1e-3, "null-speaker frames diverge from MLX golden");
    Ok(())
}
