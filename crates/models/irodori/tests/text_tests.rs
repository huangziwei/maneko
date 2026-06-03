//! M5 validation: JP frontend (normalize + tokenize) parity vs transformers + text.py golden.
//!
//!   source .cache/mlxgolden/bin/activate
//!   HF_HOME=$HOME/.cache/huggingface python ref/tools/dump_golden_text.py
//!   HF_HOME=$HOME/.cache/huggingface cargo test -p irodori --features accelerate \
//!     text_ids -- --ignored --nocapture

use candle_core::Device;
use irodori::IrodoriTokenizer;

// Must match SENTENCES in ref/tools/dump_golden_text.py.
const SENTENCES: &[&str] = &[
    "こんにちは、世界。",
    "（テスト）です！",
    "Ａ１２３ABC",
    "猫がかわいい。",
    "「引用」と……思う",
];

#[test]
#[ignore = "needs llm-jp tokenizer + golden (run ref/tools/dump_golden_text.py first)"]
fn text_ids_match_golden() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../.cache/golden/text_ids.safetensors");
    assert!(golden_path.exists(), "missing golden at {golden_path:?}; run dump_golden_text.py");
    let g = candle_core::safetensors::load(&golden_path, &dev)?;
    let ids_golden = g.get("ids").unwrap().to_vec2::<u32>()?;
    let mask_golden = g.get("mask").unwrap().to_vec2::<f32>()?;

    let tok = IrodoriTokenizer::v2(&dev)?;
    for (i, s) in SENTENCES.iter().enumerate() {
        let (ids, mask) = tok.encode(s)?;
        assert_eq!(ids, ids_golden[i], "token ids mismatch for {s:?}");
        assert_eq!(mask, mask_golden[i], "mask mismatch for {s:?}");
    }
    eprintln!("text frontend: {} sentences match golden ids+mask exactly", SENTENCES.len());
    Ok(())
}
