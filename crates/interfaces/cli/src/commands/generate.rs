//! `tts generate` — unified text → WAV over the maneko engines.
//!
//! `--engine pocket` (multilingual, 24 kHz, via [`pocket::Engine`]), `--engine irodori`
//! (Japanese, 48 kHz, via [`irodori::Irodori`]), or `--engine mio` (Japanese, 24 kHz, codec-LM via
//! [`mio_tts::Mio`], voice cloned on-device from a reference WAV). Shared flags (text, voice, output,
//! device) plus engine-specific ones. Weights resolve from `HF_HOME` — point it at the project-local
//! `.cache/huggingface` (the engines' repos live there).

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::{Parser, ValueEnum};
use owo_colors::OwoColorize;
use std::path::PathBuf;
use std::time::Instant;

/// Which TTS engine to run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// pocket-tts: multilingual (en/de/es/fr/it/pt), 24 kHz, autoregressive.
    Pocket,
    /// Irodori: Japanese, 48 kHz, flow-matching diffusion.
    Irodori,
    /// MioTTS: Japanese, 24 kHz, Falcon-H1 codec-LM; clones the voice on-device from a reference WAV.
    Mio,
}

/// Default text (English) shown when `--text` is omitted; pair with `--engine pocket`.
pub const DEFAULT_TEXT: &str = "Hello world! I am maneko, a native Rust text to speech engine.";

/// Japanese fallback substituted for [`DEFAULT_TEXT`] when `--engine mio` runs without `--text`
/// (the English default is a pocket placeholder; mio is Japanese-only).
pub const MIO_DEFAULT_TEXT: &str = "こんにちは。今日はいい天気ですね。";

#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Text to synthesize.
    #[arg(short, long, default_value = DEFAULT_TEXT)]
    pub text: String,

    /// Engine: pocket (multilingual, 24 kHz) or irodori (Japanese, 48 kHz).
    #[arg(short, long, value_enum, default_value_t = EngineKind::Pocket)]
    pub engine: EngineKind,

    /// Voice. pocket: a predefined name / .wav / .safetensors / hf:// / base64.
    /// irodori / mio: a reference .wav to clone (mio also accepts a stem under voices/ja/).
    /// Default: the engine's stock voice.
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Output WAV path.
    #[arg(short, long, default_value = "output.wav")]
    pub output: PathBuf,

    /// [pocket] Language / config stem: english_2026-04, german, french_24l, …
    #[arg(short, long, default_value = "english_2026-04")]
    pub language: String,

    /// [irodori] Target duration in seconds (default: model fallback, trimmed to silence).
    #[arg(long)]
    pub seconds: Option<f64>,

    /// [irodori] Diffusion sampling steps. v3 holds intelligibility down to ~8 (its duration
    /// predictor sizes each clip); more = better prosody/fidelity, slower. 4 transcribes but sounds rough.
    #[arg(long, default_value_t = 8)]
    pub steps: usize,

    /// [irodori/mio] Load the model from a local q8 GGUF (irodori: DiT; mio: the AR) instead of
    /// resolving it from the maneko HF repo.
    #[arg(long)]
    pub gguf: Option<PathBuf>,

    /// [pocket/mio] Sampling temperature (default: pocket 0.7, mio 0.8).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// [mio] Nucleus top-p (1.0 = full distribution).
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,

    /// [mio] Max speech tokens (~25/s; 700 ≈ 28 s).
    #[arg(long, default_value_t = 700)]
    pub max_tokens: usize,

    /// [mio] RNG seed for reproducible sampling (default: entropy).
    #[arg(long)]
    pub seed: Option<u64>,

    /// [pocket] LSD decode steps.
    #[arg(long, default_value_t = 1)]
    pub lsd_decode_steps: usize,

    /// [pocket] EOS threshold (more negative = longer audio).
    #[arg(long, default_value_t = -4.0)]
    pub eos_threshold: f32,

    /// [pocket] Noise clamp.
    #[arg(long)]
    pub noise_clamp: Option<f32>,

    /// Use the Metal GPU (macOS; requires building with --features metal).
    #[arg(long)]
    pub metal: bool,

    /// Suppress progress output.
    #[arg(short, long)]
    pub quiet: bool,
}

fn select_device(metal: bool) -> Result<Device> {
    if metal {
        #[cfg(feature = "metal")]
        {
            return Ok(Device::new_metal(0)?);
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("--metal requires building with --features metal");
        }
    }
    Ok(Device::Cpu)
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let device = select_device(args.metal)?;
    if !args.quiet {
        println!(
            "{} maneko — {} engine on {:?}",
            "▶".cyan(),
            format!("{:?}", args.engine).to_lowercase().yellow(),
            device
        );
    }

    let t = Instant::now();
    let (audio, sample_rate) = match args.engine {
        EngineKind::Pocket => generate_pocket(&args, device)?,
        EngineKind::Irodori => generate_irodori(&args, device)?,
        EngineKind::Mio => generate_mio(&args, device)?,
    };
    let gen_s = t.elapsed().as_secs_f64();

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    tts_core::audio::write_wav(&args.output, &audio, sample_rate as u32)?;

    if !args.quiet {
        let samples = *audio.dims().last().unwrap_or(&0);
        let secs = samples as f64 / sample_rate as f64;
        println!(
            "{} {:.2}s @ {} Hz in {:.2}s (RTF {:.2}x) → {}",
            "✓".green(),
            secs,
            sample_rate,
            gen_s,
            gen_s / secs.max(1e-6),
            args.output.display().cyan()
        );
    }
    Ok(())
}

/// pocket-tts via the cached multi-model [`pocket::Engine`].
fn generate_pocket(args: &GenerateArgs, device: Device) -> Result<(Tensor, usize)> {
    let params = pocket::GenParams {
        temp: args.temperature.unwrap_or(0.7),
        lsd_decode_steps: args.lsd_decode_steps,
        eos_threshold: args.eos_threshold,
        noise_clamp: args.noise_clamp,
    };
    let mut engine = pocket::Engine::new(device).with_params(params);
    let audio = engine.generate(&args.text, &args.language, args.voice.as_deref())?;
    let sr = engine.sample_rate(&args.language)?;
    Ok((audio, sr))
}

/// Irodori via [`irodori::Irodori`].
fn generate_irodori(args: &GenerateArgs, device: Device) -> Result<(Tensor, usize)> {
    let iro = match &args.gguf {
        Some(path) => irodori::Irodori::from_gguf(&device, path)?,
        None => irodori::Irodori::from_hf(&device)?,
    };
    let opts = irodori::GenerateOptions {
        seconds: args.seconds,
        sampler: irodori::SamplerConfig {
            num_steps: args.steps,
            ..irodori::SamplerConfig::default()
        },
        ..Default::default()
    };
    let audio = iro.generate(&args.text, args.voice.as_deref(), &opts)?;
    let sr = iro.sample_rate();
    Ok((audio, sr))
}

/// MioTTS via [`mio_tts::Mio`]: AR (q8 on x86_64, f32 on arm64; clone `--voice` on-device via
/// WavLM), then text → 24 kHz wav. `--gguf` forces a local q8 GGUF; q8 otherwise resolves from
/// the maneko HF repo.
fn generate_mio(args: &GenerateArgs, device: Device) -> Result<(Tensor, usize)> {
    let mut mio = mio_tts::Mio::load_default(&device, args.gguf.as_deref())?;
    mio.load_voice_encoder(mio_tts::weights::resolve_wavlm(None)?)?;
    let global = mio.encode_ref_file(resolve_mio_ref(args.voice.as_deref())?)?;

    // The English DEFAULT_TEXT is a pocket placeholder; mio is Japanese — fall back to its own.
    let raw = if args.text == DEFAULT_TEXT { MIO_DEFAULT_TEXT } else { args.text.as_str() };
    let text = mio_tts::normalize_text(raw);
    let cfg = mio_tts::GenConfig {
        max_new: args.max_tokens,
        temperature: args.temperature.unwrap_or(0.8),
        top_p: args.top_p,
        seed: args.seed,
    };
    let wav = mio.generate_with(&text, &global, &cfg)?;
    Ok((wav, mio.sample_rate()))
}

/// Resolve `--voice` for mio: an existing `.wav` path, else a stem under `voices/ja/`, else error.
/// Defaults to a stock Japanese reference when omitted.
fn resolve_mio_ref(voice: Option<&str>) -> Result<PathBuf> {
    let name = voice.unwrap_or("voices/ja/堺雅人.wav");
    let direct = PathBuf::from(name);
    if direct.exists() {
        return Ok(direct);
    }
    let stem = PathBuf::from("voices/ja").join(format!("{name}.wav"));
    if stem.exists() {
        return Ok(stem);
    }
    anyhow::bail!("mio voice {name:?} not found — pass a .wav path or a stem under voices/ja/");
}
