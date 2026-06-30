//! Rotate-half (NeoX) RoPE for Falcon-H1 attention. `emb = cat(freqs, freqs)`,
//! `x_out = x·cos + rotate_half(x)·sin`, `rotate_half([a,b]) = [-b, a]`. Distinct from the codec's
//! interleaved RoPE, so it lives here rather than in `tts_core`.

use candle_core::{DType, Device, Result, Tensor, D};

pub struct Rope {
    cos: Tensor, // (max_seq, head_dim)
    sin: Tensor,
}

impl Rope {
    pub fn new(head_dim: usize, max_seq: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / head_dim as f64)) as f32)
            .collect();
        let inv = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.broadcast_mul(&inv)?; // (max_seq, half)
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?; // (max_seq, head_dim)
        Ok(Self { cos: emb.cos()?, sin: emb.sin()? })
    }

    /// Apply to `x` of shape `(B, H, T, Dh)` at sequence `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, _h, t, d) = x.dims4()?;
        let cos = self.cos.narrow(0, offset, t)?.reshape((1, 1, t, d))?;
        let sin = self.sin.narrow(0, offset, t)?.reshape((1, 1, t, d))?;
        let half = d / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?; // [-x2, x1]
        x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
    }
}
