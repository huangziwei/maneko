//! maneko CLI — text → wav over both engines (pocket-tts + Irodori), with voice cloning.

use anyhow::Result;
use clap::Parser;

use tts_cli::commands;

/// maneko — native Rust/Candle text-to-speech.
#[derive(Parser)]
#[command(
    name = "tts",
    author,
    version,
    about = "maneko — text to speech (pocket-tts + Irodori)",
    long_about = "A Rust/Candle TTS engine. Generate speech from text with voice cloning, using \
                  either the pocket-tts (multilingual, 24 kHz) or Irodori (Japanese, 48 kHz) \
                  engine via --engine."
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
