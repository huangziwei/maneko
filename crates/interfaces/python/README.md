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
maturin develop --features accelerate,metal   # dev install: fast CPU + Apple GPU
# …or a distributable wheel:
maturin build --release --features accelerate,metal
```

`accelerate` = Apple CPU BLAS, `metal` = Apple GPU, `mkl` = Intel CPU BLAS. Build both
`accelerate,metal` for one wheel that does fast CPU **and** GPU, selected at runtime via `device=`.

## Use

```python
import maneko

# pocket-tts (multilingual, 24 kHz). device="cpu" (default) or "metal".
p = maneko.Pocket()
audio = p.generate("Hello world.", language="german", voice="nathan.wav")
maneko.save_wav("out.wav", audio, p.sample_rate("german"))

# Irodori (Japanese, 48 kHz, voice cloning). steps=8 default; duration auto-predicted.
i = maneko.Irodori(device="metal")   # GPU; omit for CPU
jp = i.generate("こんにちは。", voice="ref.wav")
maneko.save_wav("jp.wav", jp, i.sample_rate)

# Book narration: encode the narrator ONCE, reuse across chunks
# (skips the per-call voice encode; keeps timbre consistent).
ref = i.encode_ref("ref.wav")
for k, chunk in enumerate(chunks):
    maneko.save_wav(f"{k}.wav", i.generate_with_ref(chunk, ref), i.sample_rate)
```

`generate(...)` returns a mono `list[float]`. Weights resolve from `HF_HOME` — point it at the
project-local `.cache/huggingface` (both engines' repos live there). See `test_bindings.py`.
