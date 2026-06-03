//! M1 validation for the DACVAE decoder.
//!
//! These need the local weights (~430 MB torch `.pth` + the mlx-community safetensors), so they
//! are `#[ignore]`d. Run with `HF_HOME` pointing at the cache that holds the Irodori repos:
//!
//!   HF_HOME=$HOME/.cache/huggingface \
//!     cargo test -p irodori --release --features accelerate -- --ignored --nocapture
//!
//! `fold_parity_vs_mlx` is the key numeric check: it confirms our torch weight-norm fold +
//! name-map + conv layout match the independently-converted mlx-audio DACVAE — **without** needing
//! an MLX runtime (which can't run in the agent shell). `decode_smoke` confirms the full decode
//! path runs on real weights and yields bounded audio of the expected length.

use candle_core::{Device, Result, Tensor};
use irodori::weights::{hf_file, Weights};
use irodori::{Dacvae, DacvaeConfig};

const DIT_REV: &str = "Aratako/Irodori-TTS-500M-v2";
const DACVAE_REPO: &str = "Aratako/Semantic-DACVAE-Japanese-32dim";
const MLX_REPO: &str = "mlx-community/Irodori-TTS-500M-v2-fp16";

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
}

#[test]
#[ignore = "needs local DACVAE weights"]
fn decode_smoke() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let dac = Dacvae::from_hf(&dev)?;
    assert_eq!(dac.sample_rate(), 48_000);
    assert_eq!(dac.hop_length(), 1920);

    // A small VAE latent (B=1, codebook_dim=32, T=16): a low-amplitude sine per channel.
    let t = 16usize;
    let cb = 32usize;
    let mut data = Vec::with_capacity(cb * t);
    for c in 0..cb {
        for i in 0..t {
            data.push(0.1 * ((i as f32 * 0.3) + c as f32).sin());
        }
    }
    let latent = Tensor::from_vec(data, (1, cb, t), &dev)?;

    let audio = dac.decode(&latent)?;
    assert_eq!(audio.dims(), &[1, 1, t * 1920], "decode output length");

    let flat = audio.flatten_all()?.to_vec1::<f32>()?;
    assert!(flat.iter().all(|v| v.is_finite()), "decode produced non-finite samples");
    let peak = flat.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(peak <= 1.0 + 1e-4, "tanh output should be in [-1,1], peak={peak}");
    eprintln!("decode_smoke: {} samples, peak {:.4}", flat.len(), peak);
    Ok(())
}

#[test]
#[ignore = "needs local weights + golden (run ref/tools/dump_golden_dacvae.py first)"]
fn decode_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/dacvae_decode.safetensors");
    if !golden_path.exists() {
        panic!(
            "missing golden at {golden_path:?}; run:\n  \
             source .cache/mlxgolden/bin/activate && \
             HF_HOME=$HOME/.cache/huggingface python ref/tools/dump_golden_dacvae.py"
        );
    }
    let golden = candle_core::safetensors::load(&golden_path, &dev)?;
    let latent = golden.get("latent").expect("golden latent").clone(); // (1,32,T)
    let wav_golden = golden.get("wav").expect("golden wav").clone(); // (L,)

    let dac = Dacvae::from_hf(&dev)?;
    let wav_rust = dac.decode(&latent)?.flatten_all()?; // (L,)

    assert_eq!(
        wav_rust.elem_count(),
        wav_golden.elem_count(),
        "decode length mismatch"
    );
    let diff = max_abs_diff(&wav_rust, &wav_golden)?;
    let gpeak = wav_golden.abs()?.max_all()?.to_scalar::<f32>()?;
    eprintln!("decode vs MLX golden: max_abs_diff={diff:.3e}  (golden peak {gpeak:.4})");
    // Weights are bit-identical (~1e-8) to MLX's, so the forward math should match tightly.
    assert!(diff < 1e-3, "decode diverges from MLX golden: {diff}");
    Ok(())
}

#[test]
#[ignore = "needs local weights + golden (run ref/tools/dump_golden_decode_chunked.py first)"]
fn decode_chunked_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/dacvae_decode_chunked.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_decode_chunked.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;
    let dac = Dacvae::from_hf(&dev)?;
    let wav = dac.decode_chunked(g.get("latent").unwrap(), 50, 4)?.flatten_all()?;
    let golden = g.get("wav").unwrap();
    assert_eq!(wav.elem_count(), golden.elem_count(), "chunked decode length");
    let diff = max_abs_diff(&wav, golden)?;
    eprintln!("chunked decode vs MLX golden: max_abs_diff={diff:.3e}");
    assert!(diff < 1e-3, "chunked decode diverges from MLX golden: {diff}");
    Ok(())
}

#[test]
#[ignore = "needs local weights + golden (run ref/tools/dump_golden_encode.py first)"]
fn encode_matches_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/dacvae_encode.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_encode.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;
    let audio = g.get("audio").unwrap(); // (1, L)
    let (b, l) = audio.dims2()?;
    let audio = audio.reshape((b, 1, l))?; // (1,1,L)

    let dac = Dacvae::from_hf(&dev)?;
    let ref_latent = dac.encode(&audio)?; // (1, T, codebook_dim)
    let golden = g.get("ref_latent").unwrap();
    assert_eq!(ref_latent.dims(), golden.dims(), "ref_latent shape");
    let diff = max_abs_diff(&ref_latent, golden)?;
    eprintln!("encode ref_latent diff: {diff:.3e}  shape {:?}", ref_latent.dims());
    assert!(diff < 1e-3, "encode diverges from MLX golden: {diff}");
    Ok(())
}

#[test]
#[ignore = "needs local DACVAE weights"]
fn fold_parity_vs_mlx() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let _ = DIT_REV; // (kept for reference; DiT parity is M2+)

    // torch .pth (Candle layout: conv1d (out,in,k), convT (in,out,k); folded by our loader)
    let torch = Weights::from_pth(
        &hf_file(DACVAE_REPO, "weights.pth")?,
        Some("state_dict"),
        &dev,
    )?;
    // mlx-audio converted DACVAE (MLX NLC layout (out,k,in); stores weight_g/weight_v unfolded)
    let mlx = Weights::from_safetensors(&hf_file(MLX_REPO, "dacvae/model.safetensors")?, &dev)?;

    // --- Conv1d: conv_in (decoder.model.0 ↔ decoder.conv_in) ---
    let torch_w = tts_core::fold_weight_norm(
        torch.get("decoder.model.0.weight_g")?,
        torch.get("decoder.model.0.weight_v")?,
    )?; // (out,in,k)
    let mv = mlx.get("decoder.conv_in.weight_v")?; // (out,k,in)
    let mg = mlx.get("decoder.conv_in.weight_g")?; // (out,1,1)
    let mnorm = mv.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let mlx_w = mv
        .broadcast_div(&mnorm)?
        .broadcast_mul(mg)?
        .transpose(1, 2)? // (out,k,in) → (out,in,k)
        .contiguous()?;
    assert_eq!(torch_w.dims(), mlx_w.dims(), "conv_in folded shape");
    let d = max_abs_diff(&torch_w, &mlx_w)?;
    eprintln!("conv_in fold diff: {d:.3e}");
    assert!(d < 1e-4, "conv_in fold mismatch: {d}");

    // --- ConvTranspose1d: first upsample (decoder.model.1.block.1 ↔ decoder.blocks.0.block_1) ---
    let torch_t = tts_core::fold_weight_norm(
        torch.get("decoder.model.1.block.1.weight_g")?,
        torch.get("decoder.model.1.block.1.weight_v")?,
    )?; // (in,out,k)
    let tv = mlx.get("decoder.blocks.0.block_1.weight_v")?; // (out,k,in)
    let tg = mlx.get("decoder.blocks.0.block_1.weight_g")?; // (1,1,in)
    // MLX convT fold keeps the `in` axis (dim 2): norm over dims (0,1).
    let tnorm = tv.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?; // (1,1,in)
    let mlx_t = tv
        .broadcast_div(&tnorm)?
        .broadcast_mul(tg)?
        .permute((2, 0, 1))? // (out,k,in) → (in,out,k)
        .contiguous()?;
    assert_eq!(torch_t.dims(), mlx_t.dims(), "convT folded shape");
    let dt = max_abs_diff(&torch_t, &mlx_t)?;
    eprintln!("convT fold diff: {dt:.3e}");
    assert!(dt < 1e-4, "convT fold mismatch: {dt}");

    // --- quantizer_out_proj (quantizer.out_proj ↔ quantizer_out_proj) ---
    let torch_q = tts_core::fold_weight_norm(
        torch.get("quantizer.out_proj.weight_g")?,
        torch.get("quantizer.out_proj.weight_v")?,
    )?; // (out=1024,in=32,k=1)
    let qv = mlx.get("quantizer_out_proj.weight_v")?; // (1024,1,32)
    let qg = mlx.get("quantizer_out_proj.weight_g")?; // (1024,1,1)
    let qnorm = qv.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    let mlx_q = qv
        .broadcast_div(&qnorm)?
        .broadcast_mul(qg)?
        .transpose(1, 2)? // (out,k,in) → (out,in,k)
        .contiguous()?;
    let dq = max_abs_diff(&torch_q, &mlx_q)?;
    eprintln!("quantizer_out_proj fold diff: {dq:.3e}");
    assert!(dq < 1e-4, "quantizer_out_proj fold mismatch: {dq}");

    // Sanity: Snake α should be identical (1,C,1) torch vs (1,1,C) mlx after reshape.
    let ta = torch.get("decoder.wm_model.encoder_block.pre.0.alpha")?; // (1,96,1)
    let ma = mlx.get("decoder.snake_out.alpha")?.reshape(ta.dims())?; // (1,1,96)→(1,96,1)
    let da = max_abs_diff(ta, &ma)?;
    eprintln!("snake_out alpha diff: {da:.3e}");
    assert!(da < 1e-4, "snake_out alpha mismatch: {da}");

    let _ = DacvaeConfig::v2();
    Ok(())
}
