//! M3 end-to-end: text → ChatML → Falcon-H1 greedy → `<|s_N|>` → MioCodec → 24 kHz wav.
//!
//! Needs AR + codec weights (HF cache) and two goldens:
//!   HF_HOME=$PWD/.cache/huggingface .cache/miogolden/bin/python ref/tools/dump_golden_mio_gen.py
//!   HF_HOME=$PWD/.cache/huggingface .cache/miogolden/bin/python ref/tools/dump_golden_mio_preset.py "voices/ja/堺雅人.wav"
//! then:
//!   HF_HOME=$PWD/.cache/huggingface cargo test -p mio-tts --test e2e_tests -- --ignored --nocapture

use candle_core::Device;
use mio_tts::Mio;

fn golden(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.cache/golden").join(name)
}

#[test]
#[ignore = "needs weights + goldens (dump_golden_mio_gen.py + dump_golden_mio_preset.py)"]
fn generate_matches_golden_and_produces_audio() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let gen_path = golden("mio_gen.safetensors");
    let preset_path = golden("mio_preset_ref.safetensors");
    assert!(gen_path.exists(), "missing {gen_path:?}; run dump_golden_mio_gen.py");
    assert!(preset_path.exists(), "missing {preset_path:?}; run dump_golden_mio_preset.py");

    let g = candle_core::safetensors::load(&gen_path, &dev)?;
    let prompt_ids = g.get("prompt_ids").unwrap().to_vec1::<i64>()?;
    let golden_speech = g.get("speech_tokens").unwrap().to_vec1::<i64>()?;

    let mio = Mio::from_hf(&dev)?;

    // 1. Tokenizer + ChatML parity.
    let ids: Vec<i64> = mio.tokenizer().encode_prompt("こんにちは。")?.iter().map(|&x| x as i64).collect();
    assert_eq!(ids, prompt_ids, "ChatML tokenization mismatch");
    eprintln!("tokenizer: {} prompt ids match", ids.len());

    // 2. Greedy generation parity.
    let speech = mio.generate_tokens("こんにちは。", 256)?;
    eprintln!("generated {} speech tokens (golden {})", speech.len(), golden_speech.len());
    assert_eq!(speech, golden_speech, "greedy speech-token mismatch");

    // 3. Produce audio with the real voice preset.
    let global = candle_core::safetensors::load(&preset_path, &dev)?.get("global").unwrap().clone();
    let wav = mio.decode_speech(&speech, &global)?; // (samples,)
    let n = wav.elem_count();
    eprintln!("wav: {n} samples = {:.2}s @ {} Hz", n as f64 / mio.sample_rate() as f64, mio.sample_rate());

    let flat = wav.flatten_all()?.to_vec1::<f32>()?;
    assert!(flat.iter().all(|v| v.is_finite()), "non-finite samples");
    let peak = flat.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak > 1e-3 && peak <= 1.5, "implausible peak {peak}");
    assert_eq!(n, speech.len() * (mio.sample_rate() / 25), "audio length");

    // 4. Numeric parity vs the Python end-to-end waveform (run dump_golden_mio_e2e.py).
    let e2e_path = golden("mio_e2e.safetensors");
    if e2e_path.exists() {
        let want = candle_core::safetensors::load(&e2e_path, &dev)?.get("wav").unwrap().clone();
        let diff = (wav.flatten_all()? - want.flatten_all()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        eprintln!("e2e wav vs Python: max_abs_diff = {diff:.3e}");
        assert!(diff < 1e-3, "e2e waveform diverges from Python: {diff}");
    }

    let out = golden("mio_e2e_out.wav");
    tts_core::audio::write_wav(&out, &wav, mio.sample_rate() as u32)?;
    eprintln!("✓ wrote {out:?}  (peak {peak:.3})");
    Ok(())
}

/// M4 engine path: clone a voice from a WAV on-device and generate with it. Needs the WavLM bundle
/// (`.cache/mio_wavlm.safetensors`) and the encode golden (`dump_golden_mio_encode.py`).
#[test]
#[ignore = "needs weights + .cache/mio_wavlm.safetensors + mio_encode golden"]
fn clone_from_wav_and_generate() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let wavlm = root.join(".cache/mio_wavlm.safetensors");
    let ref_wav = root.join("voices/ja/堺雅人.wav");
    let enc_golden = golden("mio_encode.safetensors");
    assert!(wavlm.exists(), "missing {wavlm:?}");
    assert!(enc_golden.exists(), "missing {enc_golden:?}; run dump_golden_mio_encode.py");

    let mut mio = Mio::from_hf(&dev)?;
    mio.load_voice_encoder(&wavlm)?;

    // Clone the voice straight from the WAV file (read → mono → resample → WavLM → GlobalEncoder).
    let global = mio.encode_ref_file(&ref_wav)?; // (128,)
    assert_eq!(global.dims(), [128]);

    // The source is already 24 kHz mono, so this must match the Python global embedding.
    let want = candle_core::safetensors::load(&enc_golden, &dev)?.get("global").unwrap().clone();
    let diff = (global.flatten_all()? - want.flatten_all()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("cloned global vs Python: max_abs_diff = {diff:.3e}");
    assert!(diff < 2e-3, "cloned embedding diverges: {diff}");

    // And it drives generation end-to-end.
    let wav = mio.generate("こんにちは。", &global, 256)?;
    let flat = wav.flatten_all()?.to_vec1::<f32>()?;
    let peak = flat.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(flat.iter().all(|v| v.is_finite()) && peak > 1e-3 && peak <= 1.5, "implausible audio (peak {peak})");
    eprintln!("✓ cloned voice generated {:.2}s @ {} Hz (peak {peak:.3})", flat.len() as f64 / mio.sample_rate() as f64, mio.sample_rate());
    Ok(())
}
