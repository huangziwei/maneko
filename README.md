# maneko

**maneko** = 真似 (*mane*, "to imitate") + 猫 (*neko*, "cat") — a pun on 招き猫 (*maneki-neko*).
One native **Rust/Candle** TTS engine hosting two voice-cloning model families, with no
podman/MLX/torch in the runtime path:

- **pocket-tts v2** — multilingual (en/fr/de/it/es/pt), 24 kHz, autoregressive (Mimi + FlowLM).
- **Irodori** — Japanese, 48 kHz, flow-matching DiT + DACVAE codec.

One codebase runs on **Apple Silicon** (CPU / Accelerate / Metal) and **Intel** (CPU / MKL). See
`.claude/plans/maneko.md` for the full plan and status, and `NOTICE` for attribution.

## Status

**Both engines generate natively, behind one frozen surface** (CLI + Python + Rust). Irodori was
ported stage-by-stage and parity-checked against mlx-audio (≤1.3e-4 vs CPU golden tensors at every
stage); its output is confirmed intelligible by a Whisper round-trip. pocket-tts does multilingual
synthesis with per-language model switching and voice cloning.

Deferred polish: int8/quantized perf pass, Irodori chunked long-form decode, Intel cross-platform
validation, streaming.

## Layout

```
crates/
  core/        (tts-core)          # shared math: ops, conv, attention, RoPE, weight-norm, audio
  models/
    pocket/    (pocket)            # Mimi + FlowLM + per-language Engine + voice cloning (24 kHz)
    irodori/   (irodori)           # DiT + RF/CFG sampler + DACVAE + JP frontend (48 kHz)
  interfaces/
    cli/       (tts-cli, bin tts)  # `tts generate --engine pocket|irodori …`
    python/    (maneko-py)         # PyO3 wheel — `import maneko` (Pocket + Irodori)
ref/                               # vendored upstreams + golden-dump tools (gitignored)
.claude/plans/maneko.md            # the living plan + status
```

## Usage

Weights load from a HuggingFace cache — point `HF_HOME` at the cache holding that engine's repos
(pocket: the project-local `.cache/huggingface`; Irodori: `~/.cache/huggingface`). Always build
`--release` (debug Candle is ~40× slower); `--features accelerate` on Apple Silicon CPU,
`--features metal` for the GPU.

**CLI** (`tts`):

```sh
# pocket-tts (multilingual)
HF_HOME=$PWD/.cache/huggingface \
  cargo run --release --features accelerate -p tts-cli -- \
  generate --engine pocket --language german --voice voices/de/nathan.wav -o de.wav --text "Hallo Welt."

# Irodori (Japanese, voice cloning)
HF_HOME=$HOME/.cache/huggingface \
  cargo run --release --features accelerate -p tts-cli -- \
  generate --engine irodori --voice voices/ja/foo.wav --seconds 5 --steps 40 -o ja.wav --text "こんにちは。"
```

**Python** (`maneko`): build with `maturin develop --features accelerate` (in `crates/interfaces/python`), then:

```python
import maneko
p = maneko.Pocket()
maneko.save_wav("out.wav", p.generate("Hello.", language="b6369a24"), p.sample_rate("b6369a24"))

i = maneko.Irodori()
maneko.save_wav("ja.wav", i.generate("こんにちは。", voice="ref.wav", seconds=4, steps=40), i.sample_rate)
```

**Rust**: depend on `pocket` / `irodori`; the frozen entry points are `pocket::Engine`
(`generate(text, language, voice)`) and `irodori::Irodori` (`generate(text, ref_wav, opts)`).

## License

**AGPL-3.0-or-later** (see `LICENSE`). maneko incorporates and ports MIT-licensed upstreams
(babybirdprd/pocket-tts, mlx-audio); their notices are retained in `NOTICE` (MIT/Apache are
AGPL-compatible). Model weights are separate artifacts under their own terms
(Kyutai/Aratako/llm-jp), including no non-consensual voice cloning.
