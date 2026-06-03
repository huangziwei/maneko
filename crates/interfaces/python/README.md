# maneko — Python bindings

Native Rust/Candle TTS with no PyTorch/MLX in the runtime, exposed as the `maneko` module
(PyO3 + [Candle](https://github.com/huggingface/candle)). Two engines:

- **`maneko.Pocket`** — pocket-tts: multilingual (en/de/es/fr/it/pt), 24 kHz.
- **`maneko.Irodori`** — Irodori: Japanese, 48 kHz, with reference-voice cloning.

## Build

Needs [Rust](https://www.rust-lang.org/) and [maturin](https://github.com/PyO3/maturin).

```bash
pip install maturin
# from crates/interfaces/python, inside a venv:
maturin develop --features accelerate     # or --features metal on Apple GPU
```

## Use

```python
import maneko

# pocket-tts (multilingual, 24 kHz)
p = maneko.Pocket()
audio = p.generate("Hello world.", language="german", voice="nathan.wav")
maneko.save_wav("out.wav", audio, p.sample_rate("german"))

# Irodori (Japanese, 48 kHz, voice cloning)
i = maneko.Irodori()
jp = i.generate("こんにちは。", voice="ref.wav", seconds=4, steps=40)
maneko.save_wav("jp.wav", jp, i.sample_rate)
```

`generate(...)` returns a mono `list[float]`. Weights resolve from `HF_HOME` — point it at the
project-local `.cache/huggingface` (both engines' repos live there). See `test_bindings.py`.
