//! ResNet stack used as `wave_prior_net` / `wave_post_net`. Port of `Aratako/MioCodec`'s
//! `ResNetBlock`/`ResNetStack` (`module/istft_head.py`): `GroupNorm → SiLU → Conv1d` twice, residual.
//! Operates in `(B, C, L)`.

use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, Module, VarBuilder};

struct ResNetBlock {
    norm1: GroupNorm,
    conv1: Conv1d,
    norm2: GroupNorm,
    conv2: Conv1d,
}

impl ResNetBlock {
    fn load(channels: usize, kernel: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: (kernel - 1) / 2, ..Default::default() };
        Ok(Self {
            norm1: candle_nn::group_norm(groups, channels, 1e-6, vb.pp("norm1"))?,
            conv1: candle_nn::conv1d(channels, channels, kernel, cfg, vb.pp("conv1"))?,
            norm2: candle_nn::group_norm(groups, channels, 1e-6, vb.pp("norm2"))?,
            conv2: candle_nn::conv1d(channels, channels, kernel, cfg, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?; // dropout is identity at inference
        x + h
    }
}

pub struct ResNetStack {
    blocks: Vec<ResNetBlock>,
}

impl ResNetStack {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        num_blocks: usize,
        kernel: usize,
        groups: usize,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(ResNetBlock::load(channels, kernel, groups, vb.pp(format!("blocks.{i}")))?);
        }
        Ok(Self { blocks })
    }

    /// `x`: `(B, C, L)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        Ok(x)
    }
}
