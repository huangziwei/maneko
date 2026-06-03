//! M4 validation: RF Euler/CFG sampler parity vs an MLX-CPU golden (init noise injected).
//!
//!   source .cache/mlxgolden/bin/activate
//!   HF_HOME=$HOME/.cache/huggingface SDKROOT=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk|tail -1) \
//!     DEVELOPER_DIR=/Library/Developer/CommandLineTools python ref/tools/dump_golden_sampler.py
//!   HF_HOME=$HOME/.cache/huggingface cargo test -p irodori --features accelerate \
//!     sampler -- --ignored --nocapture

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use irodori::weights::hf_file;
use irodori::{sample_euler_cfg, DitConfig, IrodoriDiT, SamplerConfig};

const DIT_REPO: &str = "Aratako/Irodori-TTS-500M-v2";

#[test]
#[ignore = "needs DiT weights + golden (run ref/tools/dump_golden_sampler.py first)"]
fn sampler_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/sampler_latent.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_sampler.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;

    let dit_path = hf_file(DIT_REPO, "model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dit_path], DType::F32, &dev)? };
    let dit = IrodoriDiT::load(vb, DitConfig::v2(), 4096)?;

    // num_steps must match the golden dump (8).
    let cfg = SamplerConfig {
        num_steps: 8,
        cfg_scale_text: 3.0,
        cfg_scale_speaker: 5.0,
        cfg_min_t: 0.5,
        cfg_max_t: 1.0,
    };
    let latent = sample_euler_cfg(
        &dit,
        g.get("input_ids").unwrap(),
        g.get("text_mask").unwrap(),
        g.get("ref_latent").unwrap(),
        g.get("ref_mask").unwrap(),
        g.get("x_init").unwrap(),
        &cfg,
    )?;

    let golden = g.get("latent").unwrap();
    assert_eq!(latent.dims(), golden.dims(), "latent shape");
    let diff = (&latent - golden)?.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("sampler latent diff: {diff:.3e}  shape {:?}", latent.dims());
    assert!(diff < 1e-3, "sampler latent diverges from MLX golden: {diff}");
    Ok(())
}
