//! Text and reference-latent (speaker) encoders, plus their final norms.
//!
//! Port of `TextEncoder` / `ReferenceLatentEncoder` from `model.py`. Both embed/project their
//! input, run N pre-norm Transformer blocks, and **mask-zero** fully-padded positions before and
//! after every block (so they contribute nothing downstream). The speaker encoder scales its
//! in-projection by `1/6` (a real, load-bearing constant). `Encoders::encode_*` returns the
//! `*_norm`-applied state — i.e. exactly `encode_conditions`'s `text_state` / `speaker_state`.

use crate::blocks::{additive_key_mask, RmsNorm, TextBlock};
use crate::config::DitConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module};
use tts_core::{QLinear, RotaryEmbedding, Vb};

/// Run blocks with mask-zeroing: `x *= mask` before, after each block, and at the end.
fn run_masked(
    mut x: Tensor,
    blocks: &[TextBlock],
    mask: &Tensor,
    rope: &RotaryEmbedding,
) -> Result<Tensor> {
    let (b, s) = mask.dims2()?;
    let mask_f = mask.reshape((b, s, 1))?;
    let add_mask = additive_key_mask(mask)?;
    x = x.broadcast_mul(&mask_f)?;
    for block in blocks {
        x = block.forward(&x, Some(&add_mask), rope)?;
        x = x.broadcast_mul(&mask_f)?;
    }
    Ok(x)
}

struct TextEncoder {
    embedding: Embedding,
    blocks: Vec<TextBlock>,
}

impl TextEncoder {
    fn load(vb: Vb, cfg: &DitConfig) -> Result<Self> {
        let mlp_hidden = (cfg.text_dim as f64 * cfg.text_mlp_ratio) as usize;
        // Embedding weight isn't quantized; Vb::get fetches it (dequantizing if ever stored q).
        let embedding = candle_nn::Embedding::new(
            vb.pp("text_embedding")
                .get((cfg.text_vocab_size, cfg.text_dim), "weight")?,
            cfg.text_dim,
        );
        let blocks = (0..cfg.text_layers)
            .map(|i| {
                TextBlock::load(
                    vb.pp("blocks").pp(i),
                    cfg.text_dim,
                    cfg.text_heads,
                    mlp_hidden,
                    cfg.norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { embedding, blocks })
    }

    fn forward(&self, input_ids: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let x = self.embedding.forward(input_ids)?; // (B,S,text_dim)
        run_masked(x, &self.blocks, mask, rope)
    }
}

struct ReferenceLatentEncoder {
    in_proj: QLinear,
    blocks: Vec<TextBlock>,
}

impl ReferenceLatentEncoder {
    fn load(vb: Vb, cfg: &DitConfig) -> Result<Self> {
        let mlp_hidden = (cfg.speaker_dim as f64 * cfg.speaker_mlp_ratio) as usize;
        let in_proj = vb
            .pp("in_proj")
            .qlinear(cfg.speaker_in_dim(), cfg.speaker_dim, true)?;
        let blocks = (0..cfg.speaker_layers)
            .map(|i| {
                TextBlock::load(
                    vb.pp("blocks").pp(i),
                    cfg.speaker_dim,
                    cfg.speaker_heads,
                    mlp_hidden,
                    cfg.norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { in_proj, blocks })
    }

    fn forward(&self, latent: &Tensor, mask: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        let x = (self.in_proj.forward(latent)? / 6.0)?; // the /6 scalar is load-bearing
        run_masked(x, &self.blocks, mask, rope)
    }
}

/// The two condition encoders + their final RMSNorms (speaker-conditioned v2).
pub struct Encoders {
    text_encoder: TextEncoder,
    text_norm: RmsNorm,
    speaker_encoder: ReferenceLatentEncoder,
    speaker_norm: RmsNorm,
    rope: RotaryEmbedding,
}

impl Encoders {
    /// Load from a DiT [`Vb`] (root, i.e. keys `text_encoder.*`, `speaker_encoder.*`,
    /// `text_norm.*`, `speaker_norm.*`). `rope_max_seq` bounds the RoPE table (head_dim 64).
    pub fn load(vb: Vb, cfg: &DitConfig, rope_max_seq: usize) -> Result<Self> {
        let rope = RotaryEmbedding::new(cfg.head_dim(), rope_max_seq, 10000.0, vb.device())?;
        Ok(Self {
            text_encoder: TextEncoder::load(vb.pp("text_encoder"), cfg)?,
            text_norm: RmsNorm::load(vb.pp("text_norm"), cfg.text_dim, cfg.norm_eps)?,
            speaker_encoder: ReferenceLatentEncoder::load(vb.pp("speaker_encoder"), cfg)?,
            speaker_norm: RmsNorm::load(vb.pp("speaker_norm"), cfg.speaker_dim, cfg.norm_eps)?,
            rope,
        })
    }

    /// `text_state = text_norm(text_encoder(input_ids, mask))`. `input_ids`: `(B,S)` u32,
    /// `mask`: `(B,S)` f32 (1.0 valid / 0.0 pad). Returns `(B,S,text_dim)`.
    pub fn encode_text(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let state = self.text_encoder.forward(input_ids, mask, &self.rope)?;
        self.text_norm.forward(&state)
    }

    /// `speaker_state = speaker_norm(speaker_encoder(ref_latent, mask))`. `ref_latent`:
    /// `(B,T,speaker_in_dim)` (speaker_patch_size=1 ⇒ = latent_dim), `mask`: `(B,T)` f32.
    pub fn encode_speaker(&self, ref_latent: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let state = self.speaker_encoder.forward(ref_latent, mask, &self.rope)?;
        self.speaker_norm.forward(&state)
    }

    pub fn rope(&self) -> &RotaryEmbedding {
        &self.rope
    }
}
