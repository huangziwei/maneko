//! Text frontend: the Falcon-H1 multilingual BPE tokenizer + ChatML prompt rendering, and the
//! speech-token (`<|s_N|>`) constants. The AR emits speech tokens whose ids are contiguous, so the
//! MioCodec content-token index is simply `token_id - SPEECH_BASE`.

use crate::weights::hf_file;
use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

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
