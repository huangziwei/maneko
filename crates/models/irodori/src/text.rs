//! Japanese text frontend: `normalize_text` + tokenization (llm-jp).
//!
//! Faithful port of `ref/mlx-audio/.../irodori_tts/text.py`. `normalize_text` removes noise
//! characters, folds fullwidth alphanumerics → halfwidth and halfwidth katakana → fullwidth,
//! collapses long ellipses, strips surrounding brackets and trailing 。/、. Tokenization uses the
//! llm-jp `tokenizer.json` (via the `tokenizers` crate) with **no** auto special tokens, a manually
//! prepended BOS, and right-padding to `max_length`.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

// Halfwidth → fullwidth katakana, position-by-position (matches text.py's _HW_KANA/_FW_KANA).
const HW_KANA: &str = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ";
const FW_KANA: &str = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";

/// Normalize Japanese text for TTS input (see module docs).
pub fn normalize_text(text: &str) -> String {
    // 1. _REPLACE_MAP, in source order. `\t` and `\[n\]` are multi-char/literal; the rest are
    //    single-char rules folded into one pass below.
    let text = text.replace('\t', "").replace("[n]", "");

    let kana: std::collections::HashMap<char, char> =
        HW_KANA.chars().zip(FW_KANA.chars()).collect();

    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // removals
            '\u{0020}' | '\u{3000}' => {}
            ';' | '▼' | '♀' | '♂' | '《' | '》' | '≪' | '≫' | '①' | '②' | '③' | '④' | '⑤'
            | '⑥' => {}
            '\u{02d7}' | '\u{2010}'..='\u{2015}' | '\u{2043}' | '\u{2212}' | '\u{23af}'
            | '\u{23e4}' | '\u{2500}' | '\u{2501}' | '\u{2e3a}' | '\u{2e3b}' => {}
            // substitutions
            '\u{ff5e}' | '\u{301c}' => out.push('ー'),
            '？' => out.push('?'),
            '！' => out.push('!'),
            '●' | '◯' | '〇' => out.push('○'),
            '♥' => out.push('♡'),
            // fullwidth alnum → halfwidth
            '\u{ff21}'..='\u{ff3a}' => out.push(char::from_u32(c as u32 - 0xff21 + b'A' as u32).unwrap()),
            '\u{ff41}'..='\u{ff5a}' => out.push(char::from_u32(c as u32 - 0xff41 + b'a' as u32).unwrap()),
            '\u{ff10}'..='\u{ff19}' => out.push(char::from_u32(c as u32 - 0xff10 + b'0' as u32).unwrap()),
            // halfwidth → fullwidth katakana, else passthrough
            _ => out.push(*kana.get(&c).unwrap_or(&c)),
        }
    }

    // 2. Collapse runs of 3+ ellipsis (U+2026) to two.
    let collapsed = collapse_ellipsis(&out);

    // 3. Strip surrounding bracket pairs (sequentially — nested pairs peel off in order).
    let mut s = collapsed;
    for (open, close) in [('「', '」'), ('『', '』'), ('（', '）'), ('【', '】'), ('(', ')')] {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() >= 2 && chars[0] == open && *chars.last().unwrap() == close {
            s = chars[1..chars.len() - 1].iter().collect();
        }
    }

    // 4. Strip trailing 。/、.
    s.trim_end_matches(['。', '、']).to_string()
}

fn collapse_ellipsis(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '…' {
            let mut n = 1;
            while chars.peek() == Some(&'…') {
                chars.next();
                n += 1;
            }
            out.push('…');
            if n >= 2 {
                out.push('…'); // 2 stays 2; 3+ collapses to 2
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// llm-jp tokenizer wrapper: normalize → encode (no special tokens) → prepend BOS → right-pad.
pub struct IrodoriTokenizer {
    tokenizer: Tokenizer,
    bos_id: u32,
    pad_id: u32,
    max_length: usize,
}

impl IrodoriTokenizer {
    /// Load from a `tokenizer.json`. v2 uses `bos_id=1`, `pad_id=4`, `max_length=256`.
    pub fn load(tokenizer_json: &std::path::Path, bos_id: u32, pad_id: u32, max_length: usize) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        Ok(Self { tokenizer, bos_id, pad_id, max_length })
    }

    /// The v2 llm-jp tokenizer from the HF cache.
    pub fn v2(device: &Device) -> Result<Self> {
        let _ = device;
        let path = crate::weights::hf_file("llm-jp/llm-jp-3-150m", "tokenizer.json")?;
        Self::load(&path, 1, 4, 256)
    }

    /// Normalize + tokenize → padded `input_ids` and a validity `mask`, both length `max_length`.
    pub fn encode(&self, text: &str) -> Result<(Vec<u32>, Vec<f32>)> {
        let norm = normalize_text(text);
        let enc = self
            .tokenizer
            .encode(norm, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;

        let mut ids = Vec::with_capacity(self.max_length);
        ids.push(self.bos_id);
        ids.extend_from_slice(enc.get_ids());
        ids.truncate(self.max_length);
        let n = ids.len();
        ids.resize(self.max_length, self.pad_id);
        let mask = (0..self.max_length).map(|i| if i < n { 1.0 } else { 0.0 }).collect();
        Ok((ids, mask))
    }

    /// Encode to `(input_ids (1,L) u32, mask (1,L) f32)` tensors.
    pub fn encode_tensor(&self, text: &str, device: &Device) -> Result<(Tensor, Tensor)> {
        let (ids, mask) = self.encode(text)?;
        let l = self.max_length;
        let ids = Tensor::from_vec(ids, (1, l), device).context("ids tensor")?;
        let mask = Tensor::from_vec(mask, (1, l), device).context("mask tensor")?;
        Ok((ids, mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_matches_reference_cases() {
        // Cases dumped from text.py::normalize_text.
        assert_eq!(normalize_text("こんにちは、世界。"), "こんにちは、世界");
        assert_eq!(normalize_text("（テスト）"), "テスト");
        assert_eq!(normalize_text("Ａ１２３ｶﾞ"), "A123カﾞ");
        assert_eq!(normalize_text("ﾊﾛｰ‥‥‥"), "ハロー‥‥‥");
        assert_eq!(normalize_text("「引用」"), "引用");
    }

    #[test]
    fn ellipsis_collapse() {
        assert_eq!(collapse_ellipsis("a……b"), "a……b"); // 2 stays 2
        assert_eq!(collapse_ellipsis("a………b"), "a……b"); // 3 → 2
        assert_eq!(collapse_ellipsis("a…b"), "a…b"); // 1 stays 1
    }

    #[test]
    fn normalize_keeps_annotation_emojis() {
        // v3 emoji style/emotion/SFX control: the annotation emojis are outside normalize_text's
        // removal set, so they must pass through untouched to reach the tokenizer.
        for (input, emoji) in [
            ("こんにちは😊", '😊'),
            ("テスト🎵", '🎵'),
            ("わあ💦", '💦'),
            ("おっと⏩", '⏩'),
            ("ねえ🐱", '🐱'),
        ] {
            let out = normalize_text(input);
            assert!(out.contains(emoji), "normalize_text stripped {emoji:?} from {input:?} → {out:?}");
        }
    }
}
