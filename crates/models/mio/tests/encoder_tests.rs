//! M4 stage parity: WavLM voice-encoder stages vs the torch golden.
//!
//! Needs `.cache/mio_wavlm.safetensors` (export) + the golden:
//!   HF_HOME=$PWD/.cache/huggingface .cache/miogolden/bin/python ref/tools/dump_golden_mio_encode.py "voices/ja/堺雅人.wav"
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p mio-tts --test encoder_tests -- --ignored --nocapture

use candle_core::{Device, Result, Tensor};
use mio_tts::VoiceEncoder;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.flatten_all()? - b.flatten_all()?)?.abs()?.max_all()?.to_scalar::<f32>()
}

#[test]
#[ignore = "needs .cache/mio_wavlm.safetensors + golden (dump_golden_mio_encode.py)"]
fn encoder_stages_match_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let weights = root.join(".cache/mio_wavlm.safetensors");
    let golden = root.join(".cache/golden/mio_encode.safetensors");
    assert!(weights.exists(), "missing {weights:?}");
    assert!(golden.exists(), "missing {golden:?}; run dump_golden_mio_encode.py");

    let g = candle_core::safetensors::load(&golden, &dev)?;
    let wavlm_input = g.get("wavlm_input").unwrap(); // (1, samples) resampled+padded 16 kHz

    let codec = mio_tts::weights::hf_file("Aratako/MioCodec-25Hz-24kHz", "model.safetensors")?;
    let ve = VoiceEncoder::from_safetensors(&weights, &codec, &dev)?;

    let conv = ve.conv_features(wavlm_input)?;
    let d_conv = max_abs_diff(&conv, g.get("conv_out").unwrap())?;
    eprintln!("conv_out   max_abs_diff = {d_conv:.3e}  shape {:?}", conv.dims());
    assert!(d_conv < 1e-3, "conv extractor diverges: {d_conv}");

    let proj = ve.projected(wavlm_input)?;
    let d_proj = max_abs_diff(&proj, g.get("feat_proj").unwrap())?;
    eprintln!("feat_proj  max_abs_diff = {d_proj:.3e}  shape {:?}", proj.dims());
    assert!(d_proj < 1e-3, "feature projection diverges: {d_proj}");

    let layers = ve.transformer_layers(wavlm_input)?;
    for (i, key) in ["tlayer0", "tlayer1"].iter().enumerate() {
        let d = max_abs_diff(&layers[i], g.get(*key).unwrap())?;
        eprintln!("{key}     max_abs_diff = {d:.3e}  shape {:?}", layers[i].dims());
        assert!(d < 2e-3, "transformer {key} diverges: {d}");
    }

    let global = ve.encode_global(wavlm_input)?; // (1, 128)
    let d_global = max_abs_diff(&global, g.get("global_enc").unwrap())?;
    eprintln!("global     max_abs_diff = {d_global:.3e}  shape {:?}", global.dims());
    assert!(d_global < 2e-3, "global embedding diverges: {d_global}");

    // Full clone path from the raw 24 kHz waveform (stage 6 + everything above).
    let ref_global = ve.encode_ref(g.get("wav24k").unwrap())?; // (1, 128)
    let d_ref = max_abs_diff(&ref_global, g.get("global_enc").unwrap())?;
    eprintln!("encode_ref max_abs_diff = {d_ref:.3e}  shape {:?}", ref_global.dims());
    assert!(d_ref < 2e-3, "encode_ref diverges: {d_ref}");
    Ok(())
}
