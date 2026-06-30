//! M1 parity: MioCodec-25Hz-24kHz wave decode vs the torch golden.
//!
//! Needs the codec weights (HF cache) + the golden. Generate the golden first:
//!   HF_HOME=$PWD/.cache/huggingface .cache/miogolden/bin/python ref/tools/dump_golden_mio_codec.py
//! then:
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p mio-tts --release -- --ignored --nocapture

use candle_core::{Device, Result, Tensor};
use mio_tts::MioCodec;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.flatten_all()? - b.flatten_all()?)?.abs()?.max_all()?.to_scalar::<f32>()
}

#[test]
#[ignore = "needs codec weights + golden (run ref/tools/dump_golden_mio_codec.py)"]
fn codec_decode_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/mio_codec_decode.safetensors");
    assert!(
        golden_path.exists(),
        "missing golden at {golden_path:?}; run ref/tools/dump_golden_mio_codec.py"
    );
    let g = candle_core::safetensors::load(&golden_path, &dev)?;

    let indices = g.get("indices").unwrap();
    let global = g.get("global").unwrap();
    let target = g.get("wav").unwrap().elem_count(); // 48000 samples

    let codec = MioCodec::from_hf(&dev)?;
    assert_eq!(codec.sample_rate(), 24_000);
    let s = codec.decode_stages(indices, global, target)?;

    // (name, ours, golden, tolerance) — intermediates tight; the iSTFT-reconstructed wav looser.
    let checks: [(&str, &Tensor, &Tensor, f32); 5] = [
        ("content", &s.content, g.get("content_emb").unwrap(), 2e-3),
        ("prenet", &s.prenet, g.get("stage_wave_prenet").unwrap(), 2e-3),
        ("decoder", &s.decoder, g.get("stage_wave_decoder").unwrap(), 5e-3),
        ("istft_in", &s.istft_in, g.get("stage_istft_head_in").unwrap(), 5e-3),
        ("wav", &s.wav, g.get("wav").unwrap(), 1e-2),
    ];
    let mut failed = Vec::new();
    for (name, got, want, tol) in checks {
        assert_eq!(got.elem_count(), want.elem_count(), "{name} length mismatch");
        let d = max_abs_diff(got, want)?;
        eprintln!("{name:10} max_abs_diff = {d:.3e}  (tol {tol:.0e})");
        if !(d < tol) {
            failed.push(format!("{name}: {d:.3e} >= {tol:.0e}"));
        }
    }
    assert!(failed.is_empty(), "stage parity failures: {failed:?}");
    Ok(())
}
