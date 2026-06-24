# maneko 真似猫

This project exists purely because I am too cheap to upgrade my (still working perfectly fine) Intel Macbook Pro to Apple Silicon. It's a native Rust/Candle TTS engine that can run pocket-tts v2 and irodori-tts v3 on those really dated Intel mac, slowly. 

And it turns out that only the pocket-tts is worth running and can get sub-1x real time audio output, while irodori-tts is 5~10x slower than real time. It's still a win and a good way to waste tokens.

## Usage

Weights load from a HuggingFace cache — point `HF_HOME` at the cache holding that engine's repos
(both engines use the project-local `.cache/huggingface`). Always build
`--release` (debug Candle is ~40× slower); `--features accelerate` on Apple Silicon CPU,
`--features metal` for the GPU.

**CLI** (`tts`):

```sh
# pocket-tts (multilingual)
HF_HOME=$PWD/.cache/huggingface \
  cargo run --release --features accelerate -p tts-cli -- \
  generate --engine pocket --language german --voice voices/de/nathan.wav -o de.wav --text "Hallo Welt."

# Irodori v3 (Japanese, voice cloning) — duration is auto-predicted; omit --seconds, or pass it to override
HF_HOME=$PWD/.cache/huggingface \
  cargo run --release --features accelerate -p tts-cli -- \
  generate --engine irodori --voice voices/ja/ref.wav --steps 8 -o ja.wav --text "こんにちは。"
```

**Python** (`maneko`): build with `maturin develop --features accelerate` (in `crates/interfaces/python`), then:

```python
import maneko
p = maneko.Pocket()
maneko.save_wav("out.wav", p.generate("Hello.", language="english_2026-04", voice="alba"), p.sample_rate("english_2026-04"))

i = maneko.Irodori()
maneko.save_wav("ja.wav", i.generate("こんにちは。", voice="ref.wav", seconds=4, steps=8), i.sample_rate)
```

**Rust**: depend on `pocket` / `irodori`; the frozen entry points are `pocket::Engine`
(`generate(text, language, voice)`) and `irodori::Irodori` (`generate(text, ref_wav, opts)`).

## License

**AGPL-3.0-or-later** (see `LICENSE`). maneko incorporates and ports MIT-licensed upstreams
(babybirdprd/pocket-tts, mlx-audio); their notices are retained in `NOTICE` (MIT/Apache are
AGPL-compatible). Model weights are separate artifacts under their own terms
(Kyutai/Aratako/llm-jp), including no non-consensual voice cloning.
