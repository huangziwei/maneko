# maneko

**maneko** = 真似 (*mane*, "to imitate") + 猫 (*neko*, "cat") — a pun on 招き猫 (*maneki-neko*).
One native **Rust/Candle** TTS engine hosting two voice-cloning model families, so we can
**delete podman** (and, later, MLX/torch):

- **pocket-tts v2** (multilingual: en/fr/de/it/es/pt, 24 kHz) — for `neb`
- **Irodori** (Japanese, flow-matching DiT + DACVAE, 48 kHz) — for `nik`

One binary runs natively on **Intel** (CPU / MKL) and **Apple Silicon** (CPU / Metal). See
`.claude/plans/maneko.md` for the full plan and status, and `NOTICE` for attribution.

## Status

**P0 reached** — babybirdprd's Candle pocket-tts is forked into this workspace, bumped to
candle 0.10.2, and generates a 24 kHz wav natively (no podman / MLX / torch). Next: extract
`tts-core`, add pocket v2 multilingual (P2), then Irodori (P3).

## Layout

```
crates/
  pocket-tts/          # pocket-tts library (FlowLM + Mimi + voice cloning) — the seed
  pocket-tts-cli/      # `generate` / `serve` CLI (build with --no-default-features for now)
  pocket-tts-bindings/ # optional PyO3 bindings (excluded from the default build)
assets/                # local reference wavs + fixtures (gitignored; repopulate from ref/)
ref/                   # vendored upstream sources for porting (gitignored; see ref/README.md)
.claude/plans/maneko.md   # the plan (verified + revised)
```

Target layout (rename pending): `crates/{tts-core,pocket,irodori,tts-cli,bindings}` + `tools/`.

## Quickstart

Weights load from a HuggingFace cache (`HF_HOME`). The v1-English model is ~225 MB and may
already be cached locally (e.g. `neb/.cache/huggingface`); P0 copies what it needs into a
maneko-local `.cache/huggingface`. The example clones its voice from `assets/ref.wav` (local,
gitignored) — pass `VOICE=/path/to.wav` for your own, or copy one from `ref/pocket-tts-rs/assets/`.

```sh
# Generate a wav (ALWAYS build --release — debug Candle is ~40x slower)
HF_HOME=$PWD/.cache/huggingface \
  cargo run --release -p pocket-tts --example maneko_generate -- "Hello from maneko."

# Intel MKL BLAS (same kernels torch uses) — fastest on x86_64:
HF_HOME=$PWD/.cache/huggingface \
  cargo run --release --features mkl -p pocket-tts --example maneko_generate -- "Hello."
```

Features: `mkl` (Intel), `accelerate` (macOS CPU), `metal` (Apple GPU), `quantized`, `wasm`.

## License

**AGPL-3.0-or-later** (see `LICENSE`). maneko incorporates and ports MIT-licensed upstreams
(babybirdprd/pocket-tts, mlx-audio); their notices are retained in `NOTICE` (MIT/Apache are
AGPL-compatible). Running maneko as a network service (the `serve` command) triggers AGPL §13
(offer source to remote users); offline CLI/library use does not. Model weights are separate
artifacts under their own terms (Kyutai/Aratako/llm-jp), incl. no non-consensual voice cloning.
