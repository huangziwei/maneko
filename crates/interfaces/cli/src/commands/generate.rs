//! `tts generate` — unified text → WAV over both maneko engines.
//!
//! `--engine pocket` (multilingual, 24 kHz, via [`pocket::Engine`]) or `--engine irodori`
//! (Japanese, 48 kHz, via [`irodori::Irodori`]). Shared flags (text, voice, output, device) plus
//! engine-specific ones. Weights resolve from `HF_HOME` — point it at the project-local
//! `.cache/huggingface` (both engines' repos live there).

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
}

/// Default text (English) shown when `--text` is omitted; pair with `--engine pocket`.
pub const DEFAULT_TEXT: &str = "Hello world! I am maneko, a native Rust text to speech engine.";

#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Text to synthesize.
    #[arg(short, long, default_value = DEFAULT_TEXT)]
    pub text: String,

    /// Engine: pocket (multilingual, 24 kHz) or irodori (Japanese, 48 kHz).
    #[arg(short, long, value_enum, default_value_t = EngineKind::Pocket)]
    pub engine: EngineKind,

    /// Voice. pocket: a predefined name / .wav / .safetensors / hf:// / base64.
    /// irodori: a reference .wav to clone. Default: the engine's stock voice.
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

    /// [irodori] Load the DiT from a local q8 GGUF (Vb::from_gguf) instead of the f32 HF weights.
    #[arg(long)]
    pub gguf: Option<PathBuf>,

    /// [pocket] Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,

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
        temp: args.temperature,
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
