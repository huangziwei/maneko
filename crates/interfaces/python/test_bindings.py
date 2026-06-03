"""Smoke test for the `maneko` Python bindings.

Build + install first:
    maturin develop --features accelerate     # in this dir, inside a venv

Then run (point HF_HOME at the cache holding the engine's weights):
    HF_HOME=$PWD/.cache/huggingface python test_bindings.py pocket
    HF_HOME=$HOME/.cache/huggingface  python test_bindings.py irodori voices/ja/foo.wav
"""

import sys

import maneko

print("imported maneko:", [n for n in dir(maneko) if not n.startswith("_")])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "pocket"
    if which == "pocket":
        p = maneko.Pocket()
        sr = p.sample_rate("b6369a24")
        audio = p.generate("Hello from maneko.", language="b6369a24")
        maneko.save_wav("maneko_pocket.py.wav", audio, sr)
        print(f"pocket: {len(audio)} samples @ {sr} Hz -> maneko_pocket.py.wav")
    elif which == "irodori":
        voice = sys.argv[2] if len(sys.argv) > 2 else None
        i = maneko.Irodori()
        audio = i.generate("こんにちは。", voice=voice, seconds=3, steps=40)
        maneko.save_wav("maneko_irodori.py.wav", audio, i.sample_rate)
        print(f"irodori: {len(audio)} samples @ {i.sample_rate} Hz -> maneko_irodori.py.wav")
    else:
        sys.exit(f"unknown engine: {which!r} (use 'pocket' or 'irodori')")


if __name__ == "__main__":
    main()
