//! GlobalEncoder (M4 stage 5): averaged WavLM layer-1/2 features `(B, T, 768)` → 128-d speaker
//! embedding. A ConvNeXt backbone (embed conv + 4 blocks) feeds an attentive-statistics pool.
//! Weights live in the **codec** checkpoint (`global_encoder.*`), not the WavLM bundle.

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

const CN_EPS: f64 = 1e-6; // ConvNeXt LayerNorms
const POOL_EPS: f64 = 1e-5; // pooling output LayerNorm

/// ConvNeXt block (1-D): depthwise conv → LayerNorm → pointwise 384→1152→384 (erf-GELU) → γ-scale,
/// residual. Operates on `(B, C, T)`.
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pw1: Linear,
    pw2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(dim: usize, inter: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 3, groups: dim, ..Default::default() };
        Ok(Self {
            dwconv: candle_nn::conv1d(dim, dim, 7, cfg, vb.pp("dwconv"))?,
            norm: candle_nn::layer_norm(dim, CN_EPS, vb.pp("norm"))?,
            pw1: candle_nn::linear(dim, inter, vb.pp("pwconv1"))?,
            pw2: candle_nn::linear(inter, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dwconv.forward(x)?.transpose(1, 2)?.contiguous()?; // (B, T, C)
        let h = self.pw2.forward(&self.pw1.forward(&self.norm.forward(&h)?)?.gelu_erf()?)?;
        let h = h.broadcast_mul(&self.gamma)?.transpose(1, 2)?.contiguous()?; // γ-scale, (B, C, T)
        x + h
    }
}

/// ConvNeXt backbone: embed conv (768→384) + LayerNorm + 4 blocks + final LayerNorm. `(B, T, 768)` →
/// `(B, T, 384)` (`proj_out` is identity here).
struct ConvNextBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_ln: LayerNorm,
}

impl ConvNextBackbone {
    fn load(in_ch: usize, dim: usize, inter: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 3, ..Default::default() };
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            blocks.push(ConvNeXtBlock::load(dim, inter, vb.pp(format!("convnext.{i}")))?);
        }
        Ok(Self {
            embed: candle_nn::conv1d(in_ch, dim, 7, cfg, vb.pp("embed"))?,
            norm: candle_nn::layer_norm(dim, CN_EPS, vb.pp("norm"))?,
            blocks,
            final_ln: candle_nn::layer_norm(dim, CN_EPS, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(&x.transpose(1, 2)?.contiguous()?)?; // (B, 384, T)
        let x = self.norm.forward(&x.transpose(1, 2)?.contiguous()?)?; // LN on (B, T, 384)
        let mut x = x.transpose(1, 2)?.contiguous()?; // (B, 384, T)
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        self.final_ln.forward(&x.transpose(1, 2)?.contiguous()?) // (B, T, 384)
    }
}

/// Attentive statistics pooling: per-channel softmax attention over time → weighted mean+std →
/// concat `(B, 2C)` → Linear → LayerNorm. `(B, C, T)` → `(B, out)`.
struct AttentiveStatsPool {
    attn1: Conv1d, // C→128
    attn2: Conv1d, // 128→C
    proj: Linear,  // 2C→out
    norm: LayerNorm,
}

impl AttentiveStatsPool {
    fn load(dim: usize, attn_ch: usize, out: usize, vb: VarBuilder) -> Result<Self> {
        let k1 = Conv1dConfig::default();
        Ok(Self {
            attn1: candle_nn::conv1d(dim, attn_ch, 1, k1, vb.pp("attn.0"))?,
            attn2: candle_nn::conv1d(attn_ch, dim, 1, k1, vb.pp("attn.2"))?,
            proj: candle_nn::linear(dim * 2, out, vb.pp("proj"))?,
            norm: candle_nn::layer_norm(out, POOL_EPS, vb.pp("norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.attn2.forward(&self.attn1.forward(x)?.tanh()?)?; // (B, C, T)
        let alpha = candle_nn::ops::softmax(&a, 2)?; // softmax over time
        let mean = alpha.mul(x)?.sum(2)?; // (B, C)
        let resid = (alpha.mul(&x.sqr()?)?.sum(2)? - mean.sqr()?)?;
        let std = resid.clamp(1e-4, 1e4)?.sqrt()?;
        let stats = Tensor::cat(&[&mean, &std], 1)?; // (B, 2C)
        self.norm.forward(&self.proj.forward(&stats)?)
    }
}

/// SSL features `(B, T, 768)` → 128-d speaker embedding `(B, 128)`.
pub struct GlobalEncoder {
    backbone: ConvNextBackbone,
    pooling: AttentiveStatsPool,
}

impl GlobalEncoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            backbone: ConvNextBackbone::load(768, 384, 1152, 4, vb.pp("backbone"))?,
            pooling: AttentiveStatsPool::load(384, 128, 128, vb.pp("pooling"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feats = self.backbone.forward(x)?.transpose(1, 2)?.contiguous()?; // (B, 384, T)
        self.pooling.forward(&feats)
    }
}
