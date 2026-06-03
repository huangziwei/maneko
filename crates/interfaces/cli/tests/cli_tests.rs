use assert_cmd::Command;
use std::path::Path;

#[test]
fn cli_help_lists_engine_flag() {
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tts").unwrap();
    let out = cmd.arg("generate").arg("--help").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(stdout.contains("--engine"), "generate --help should advertise --engine");
}

#[test]
#[ignore = "needs pocket weights + HF_HOME=$PWD/.cache/huggingface"]
fn cli_generate_pocket() {
    let out = "test_cli_pocket.wav";
    let _ = std::fs::remove_file(out);
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tts").unwrap();
    cmd.args(["generate", "--engine", "pocket", "--language", "english_2026-04", "--text", "Hello from the CLI.", "--output", out])
        .assert()
        .success();
    assert!(Path::new(out).exists());
    assert!(hound::WavReader::open(out).unwrap().duration() > 0);
    std::fs::remove_file(out).unwrap();
}

#[test]
#[ignore = "needs Irodori weights + HF_HOME=$PWD/.cache/huggingface"]
fn cli_generate_irodori() {
    let out = "test_cli_irodori.wav";
    let _ = std::fs::remove_file(out);
    #[allow(deprecated)]
    let mut cmd = Command::cargo_bin("tts").unwrap();
    cmd.args([
        "generate", "--engine", "irodori", "--text", "こんにちは。", "--voice", "voices/ja/花澤香菜.wav",
        "--seconds", "2", "--steps", "16", "--output", out,
    ])
    .assert()
    .success();
    assert!(Path::new(out).exists());
    let r = hound::WavReader::open(out).unwrap();
    assert_eq!(r.spec().sample_rate, 48000);
    std::fs::remove_file(out).unwrap();
}
