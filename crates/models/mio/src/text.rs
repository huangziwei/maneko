//! Text frontend: the Falcon-H1 multilingual BPE tokenizer + ChatML prompt rendering, the
//! speech-token (`<|s_N|>`) constants, and [`normalize_text`]. The AR emits speech tokens whose ids
//! are contiguous, so the MioCodec content-token index is simply `token_id - SPEECH_BASE`.

use crate::weights::hf_file;
use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

/// Half-width katakana → full-width, char-for-char (copied verbatim from the upstream
/// `text.py` `str.maketrans` source; the two strings must stay the same length and aligned).
const HALFWIDTH_KATAKANA: &str = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ";
const FULLWIDTH_KATAKANA: &str = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン";

/// Normalize input text the way the MioTTS server does **before** ChatML rendering — a port of
/// `miotts_server.text.normalize_text`. The model layer ([`MioTokenizer::encode_prompt`]) stays raw;
/// call this at the frontend boundary (CLI / Python), mirroring upstream `api.py`, which normalizes
/// before `apply_chat_template`. Pipeline (order matters):
/// 1. drop tabs, the literal `[n]`, half/full-width spaces, a fixed junk-symbol set, and a dash set;
/// 2. fold full-width ASCII letters & digits to half-width, half-width katakana to full-width;
/// 3. collapse `…{3,}` → `……`; 4. strip one matched outer bracket pair; 5. drop trailing `。`/`、`.
pub fn normalize_text(text: &str) -> String {
    // REPLACE_MAP — every entry is character-level except the literal "[n]".
    let stripped = text.replace('\t', "").replace("[n]", "");
    let mut s = String::with_capacity(stripped.len());
    for c in stripped.chars() {
        match c {
            // dropped: spaces (half/full-width)
            ' ' | '\u{3000}' => {}
            // dropped: junk/symbol set [;▼♀♂《》≪≫①②③④⑤⑥]
            ';' | '▼' | '♀' | '♂' | '《' | '》' | '≪' | '≫' | '①' | '②' | '③' | '④' | '⑤'
            | '⑥' => {}
            // dropped: dash/hyphen set
            '\u{02d7}' | '\u{2010}'..='\u{2015}' | '\u{2043}' | '\u{2212}' | '\u{23af}'
            | '\u{23e4}' | '\u{2500}' | '\u{2501}' | '\u{2e3a}' | '\u{2e3b}' => {}
            // replaced
            '\u{ff5e}' | '\u{301c}' => s.push('ー'), // full-width tilde / wave dash → ー
            '？' => s.push('?'),
            '！' => s.push('!'),
            '●' | '◯' | '〇' => s.push('○'),
            '♥' => s.push('♡'),
            // translate tables (disjoint from the above, so order vs REPLACE_MAP is irrelevant)
            'Ａ'..='Ｚ' => s.push(char::from_u32(c as u32 - 0xFF21 + 0x41).unwrap()),
            'ａ'..='ｚ' => s.push(char::from_u32(c as u32 - 0xFF41 + 0x61).unwrap()),
            '０'..='９' => s.push(char::from_u32(c as u32 - 0xFF10 + 0x30).unwrap()),
            _ => s.push(halfwidth_katakana_to_fullwidth(c).unwrap_or(c)),
        }
    }

    let mut s = collapse_ellipsis(&s);
    // Strip a single matched outer bracket pair (sequential, mirroring the upstream `if` chain).
    for (open, close) in [('「', '」'), ('『', '』'), ('（', '）'), ('【', '】'), ('(', ')')] {
        s = strip_outer_pair(s, open, close);
    }
    if s.ends_with(['。', '、']) {
        s = s.trim_end_matches(['。', '、']).to_string();
    }
    s
}

/// Half-width katakana → full-width via the verbatim upstream table (incl. `ｰ` → `ー`).
fn halfwidth_katakana_to_fullwidth(c: char) -> Option<char> {
    HALFWIDTH_KATAKANA.chars().position(|h| h == c).and_then(|i| FULLWIDTH_KATAKANA.chars().nth(i))
}

/// Collapse runs of 3+ `…` (U+2026) to exactly two — `re.sub(r"…{3,}", "……", …)`.
fn collapse_ellipsis(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut run = 0usize;
    let flush = |out: &mut String, run: usize| match run {
        0 => {}
        n if n >= 3 => out.push_str("……"),
        n => (0..n).for_each(|_| out.push('…')),
    };
    for c in s.chars() {
        if c == '…' {
            run += 1;
        } else {
            flush(&mut out, run);
            run = 0;
            out.push(c);
        }
    }
    flush(&mut out, run);
    out
}

/// If `s` both starts with `open` and ends with `close` (≥2 chars), drop those outer chars —
/// upstream `text[1:-1]` (by code point, so a 2-char pair collapses to "").
fn strip_outer_pair(s: String, open: char, close: char) -> String {
    let n = s.chars().count();
    if n >= 2 && s.starts_with(open) && s.ends_with(close) {
        s.chars().skip(1).take(n - 2).collect()
    } else {
        s
    }
}

/// First speech-token id: `id(<|s_0|>) = 65536`, contiguous through `<|s_12799|> = 78335`.
pub const SPEECH_BASE: u32 = 65536;
pub const SPEECH_COUNT: u32 = 12800;
/// Stop tokens (generation_config `eos_token_id`): `<|im_end|>` (229) and 11.
pub const EOS_IDS: [u32; 2] = [11, 229];

pub struct MioTokenizer {
    tok: Tokenizer,
}

impl MioTokenizer {
    pub fn from_hf() -> Result<Self> {
        Self::from_file(hf_file("Aratako/MioTTS-0.1B", "tokenizer.json")?)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = Tokenizer::from_file(path.as_ref()).map_err(|e| anyhow!("load tokenizer: {e}"))?;
        Ok(Self { tok })
    }

    /// Render the ChatML prompt for `text` and tokenize it (no auto special tokens — the markers are
    /// in the rendered string; `bos_token_id` is null).
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let s = format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let enc = self.tok.encode(s, false).map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Map a generated token id to a speech-content index, if it is a speech token.
    pub fn speech_index(id: u32) -> Option<i64> {
        (SPEECH_BASE..SPEECH_BASE + SPEECH_COUNT).contains(&id).then(|| (id - SPEECH_BASE) as i64)
    }

    pub fn is_eos(id: u32) -> bool {
        EOS_IDS.contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn katakana_tables_aligned() {
        assert_eq!(HALFWIDTH_KATAKANA.chars().count(), FULLWIDTH_KATAKANA.chars().count());
    }

    #[test]
    fn normalize_matches_upstream_rules() {
        // trailing 。/、 (rstrip the whole run)
        assert_eq!(normalize_text("こんにちは。"), "こんにちは");
        assert_eq!(normalize_text("あ。。、"), "あ");
        // full-width ASCII letters + digits → half-width
        assert_eq!(normalize_text("ＡＢＣａｚ"), "ABCaz");
        assert_eq!(normalize_text("１２３"), "123");
        // half-width katakana → full-width (incl. ｰ → ー)
        assert_eq!(normalize_text("ｶﾀｶﾅ"), "カタカナ");
        assert_eq!(normalize_text("ｱｲｳｰ"), "アイウー");
        // dropped: spaces (half/full-width), junk symbols, dash set, tab, literal [n]
        assert_eq!(normalize_text("あ い　う"), "あいう");
        assert_eq!(normalize_text("①あ▼い《x》"), "あいx");
        assert_eq!(normalize_text("あ―い—う"), "あいう"); // U+2015, U+2014
        assert_eq!(normalize_text("あ\t[n]い"), "あい");
        // replaced: tilde/wave→ー, ？！→?!, ●◯〇→○, ♥→♡
        assert_eq!(normalize_text("あ～い〜"), "あーいー"); // U+FF5E, U+301C
        assert_eq!(normalize_text("なに？すごい！"), "なに?すごい!");
        assert_eq!(normalize_text("〇●◯"), "○○○");
        assert_eq!(normalize_text("♥"), "♡");
        // …{3,} → ……  (1 and 2 are left alone)
        assert_eq!(normalize_text("あ…………"), "あ……"); // 6 → 2
        assert_eq!(normalize_text("あ…"), "あ…");
        assert_eq!(normalize_text("あ……"), "あ……");
        // single matched outer bracket pair, then trailing strip
        assert_eq!(normalize_text("「こんにちは。」"), "こんにちは");
        assert_eq!(normalize_text("『本』"), "本");
        assert_eq!(normalize_text("（注）"), "注");
        assert_eq!(normalize_text("【重要】"), "重要");
        assert_eq!(normalize_text("(x)"), "x");
        assert_eq!(normalize_text("（「x」）"), "「x」"); // only the outer （）, sequentially
        assert_eq!(normalize_text("「」"), "");
        // disjoint rules compose in one pass
        assert_eq!(normalize_text("Ａ１ｱ"), "A1ア");
    }
}
