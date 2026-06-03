//! maneko CLI — the pocket-tts engine: text -> wav, with optional voice cloning.

use anyhow::Result;
use clap::Parser;

use tts_cli::commands;

/// maneko / pocket-tts — high-quality text-to-speech, fast on CPU.
#[derive(Parser)]
#[command(
    name = "tts",
    author,
    version,
    about = "maneko / pocket-tts - text to speech",
    long_about = "A Rust/Candle TTS engine (pocket-tts). Generate natural speech from text, \
                  with voice cloning from audio samples."
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Generate audio from text and save it to a WAV file.
    Generate(commands::generate::GenerateArgs),
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Commands::Generate(cmd_args) => commands::generate::run(cmd_args),
    }
}
