//! maneko CLI — text → wav over the engines (pocket-tts + Irodori + MioTTS), with voice cloning.

use anyhow::Result;
use clap::Parser;

use tts_cli::commands;

/// maneko — native Rust/Candle text-to-speech.
#[derive(Parser)]
#[command(
    name = "tts",
    author,
    version,
    about = "maneko — text to speech (pocket-tts + Irodori + MioTTS)",
    long_about = "A Rust/Candle TTS engine. Generate speech from text with voice cloning, using the \
                  pocket-tts (multilingual, 24 kHz), Irodori (Japanese, 48 kHz), or MioTTS \
                  (Japanese, 24 kHz) engine via --engine."
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
