//! M2 parity: Falcon-H1 (MioTTS-0.1B AR) forward vs the torch golden.
//!
//! Needs AR weights (HF cache) + the golden. Generate the golden first:
//!   HF_HOME=$PWD/.cache/huggingface .cache/miogolden/bin/python ref/tools/dump_golden_mio_ar.py
//! then:
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p mio-tts --test ar_tests -- --ignored --nocapture

use candle_core::{Device, IndexOp, Result, Tensor};
use mio_tts::FalconH1;

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.flatten_all()? - b.flatten_all()?)?.abs()?.max_all()?.to_scalar::<f32>()
}

#[test]
#[ignore = "needs AR weights + golden (run ref/tools/dump_golden_mio_ar.py)"]
fn ar_forward_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/mio_ar_forward.safetensors");
    assert!(
        golden_path.exists(),
        "missing golden at {golden_path:?}; run ref/tools/dump_golden_mio_ar.py"
    );
    let g = candle_core::safetensors::load(&golden_path, &dev)?;
    let ids = g.get("input_ids").unwrap();

    let model = FalconH1::from_hf(&dev)?;
    let s = model.forward_stages(ids)?;

    let checks: [(&str, &Tensor, &Tensor, f32); 5] = [
        ("layer0_mamba", &s.layer0_mamba, g.get("stage_layer0_mamba").unwrap(), 2e-3),
        ("layer0_attn", &s.layer0_attn, g.get("stage_layer0_attn").unwrap(), 2e-3),
        ("layer0_out", &s.layer0_out, g.get("stage_layer0_out").unwrap(), 2e-3),
        ("hidden", &s.hidden, g.get("hidden").unwrap(), 5e-3),
        ("logits", &s.logits, g.get("logits").unwrap(), 1e-2),
    ];
    let mut failed = Vec::new();
    for (name, got, want, tol) in checks {
        assert_eq!(got.elem_count(), want.elem_count(), "{name} length mismatch");
        let d = max_abs_diff(got, want)?;
        eprintln!("{name:14} max_abs_diff = {d:.3e}  (tol {tol:.0e})");
        if !(d < tol) {
            failed.push(format!("{name}: {d:.3e} >= {tol:.0e}"));
        }
    }

    // Greedy next-token must match the reference argmax.
    let logits_last = s.logits.i((0, s.logits.dim(1)? - 1))?; // (vocab,)
    let argmax = logits_last.argmax(0)?.to_scalar::<u32>()? as i64;
    let want_argmax = g.get("argmax_last").unwrap().to_vec1::<i64>()?[0];
    eprintln!("argmax last: ours={argmax} golden={want_argmax}");
    assert_eq!(argmax, want_argmax, "greedy next-token mismatch");

    assert!(failed.is_empty(), "stage parity failures: {failed:?}");
    Ok(())
}

/// The incremental cached decode (`prefill` + `decode_step`, which use the hand-rolled depthwise
/// conv) must match the reference full `forward` (candle `Conv1d`) at the logit level, in pure f32.
/// This isolates the cached path's numerics from q8 rounding — a large divergence here is a conv /
/// recurrence bug; ~1e-5 is the expected f32-accumulation floor.
#[test]
#[ignore = "needs f32 AR weights (Aratako/MioTTS-0.1B model.safetensors)"]
fn cached_decode_matches_full_forward() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let model = FalconH1::from_hf(&dev)?;
    // A long fixed pseudo-sequence (deterministic, < vocab 78336, with speech-range ≥65536 tokens)
    // so the cached attention is exercised at large Tc — a hand-rolled-attention bug that only
    // shows up for long contexts would surface here, not in a handful of tokens.
    let n = 192usize;
    let ids: Vec<u32> = (0..n).map(|i| ((i * 1009 + 17) % 78336) as u32).collect();

    // Reference: full forward over the whole sequence (logits at every position).
    let ids_t = Tensor::from_vec(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), (1, n), &dev)?;
    let (_, logits_full) = model.forward(&ids_t)?;

    // prefill(all) last-position logits must match forward's last position.
    let (pref_last, _) = model.prefill(&ids)?;
    let d_prefill = max_abs_diff(&logits_full.i((0, n - 1))?, &pref_last)?;
    eprintln!("prefill(all) vs forward last:        max_abs_diff = {d_prefill:.3e}");

    // Incremental decode: prefill the first token, then step the rest, comparing the logits after
    // feeding token i (predict i+1) to forward's position i — across the whole growing context.
    let (mut step_logits, mut cache) = model.prefill(&ids[..1])?;
    let mut worst = max_abs_diff(&logits_full.i((0, 0))?, &step_logits)?;
    for i in 1..n {
        step_logits = model.decode_step(ids[i], &mut cache)?;
        let d = max_abs_diff(&logits_full.i((0, i))?, &step_logits)?;
        if d > worst {
            worst = d;
        }
    }
    eprintln!("decode_step vs forward (all {n} pos):  max_abs_diff = {worst:.3e}");

    assert!(d_prefill < 1e-2, "prefill diverges from forward: {d_prefill:.3e}");
    assert!(worst < 1e-2, "decode_step diverges from forward at some position: {worst:.3e}");
    Ok(())
}
