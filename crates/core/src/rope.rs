//! Interleaved Rotary Position Embedding (the GPT-J / "rope_i" convention).
//!
//! Irodori rotates **adjacent even/odd pairs** (`x[..., 0::2]` with `x[..., 1::2]`), not the
//! rotate-half (NeoX) convention. Pair `j` is rotated by angle `pos · θ^(-2j/dim)`. This matches
//! the reference `apply_rotary_emb` exactly; getting the convention wrong is a top-3 port risk
//! (`ref/port-irodori-to-rust.md` §6), so it lives here with a unit test.

use candle_core::{DType, Device, Result, Tensor};

/// Precomputed `cos`/`sin` tables for interleaved RoPE.
pub struct RotaryEmbedding {
    cos: Tensor, // (max_seq_len, dim/2)
    sin: Tensor, // (max_seq_len, dim/2)
}

impl RotaryEmbedding {
    /// `dim` = head dimension (must be even). `theta` is the base (Irodori uses 10000).
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.broadcast_mul(&inv_freq)?; // (max_seq_len, half)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Apply interleaved RoPE to `x` of shape `(B, H, T, Dh)`, starting at position `offset`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        let half = d / 2;
        let cos = self.cos.narrow(0, offset, t)?.reshape((1, 1, t, half))?;
        let sin = self.sin.narrow(0, offset, t)?.reshape((1, 1, t, half))?;
        // Split adjacent pairs: xr[..., j, 0] = x[..., 2j], xr[..., j, 1] = x[..., 2j+1].
        let xr = x.reshape((b, h, t, half, 2))?;
        let x1 = xr.narrow(4, 0, 1)?.squeeze(4)?; // even
        let x2 = xr.narrow(4, 1, 1)?.squeeze(4)?; // odd
        let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let o2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        // Re-interleave: [o1_0, o2_0, o1_1, o2_1, ...].
        Tensor::stack(&[o1, o2], 4)?.reshape((b, h, t, d))
    }

    /// Apply RoPE to the **first half of the heads** only (JointAttention's `_apply_rotary_half`),
    /// leaving the second half untouched. `x`: `(B, H, T, Dh)`.
    pub fn apply_half_heads(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, h, _t, _d) = x.dims4()?;
        let half = h / 2;
        if half == 0 {
            return self.apply(x, offset);
        }
        let first = x.narrow(1, 0, half)?.contiguous()?;
        let rest = x.narrow(1, half, h - half)?;
        let first = self.apply(&first, offset)?;
        Tensor::cat(&[first, rest], 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_rope_pos0_is_identity() -> Result<()> {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 8, 10000.0, &dev)?;
        // At position 0 the angle is 0 → cos=1, sin=0 → identity.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &dev)?;
        let y = rope.apply(&x, 0)?.flatten_all()?.to_vec1::<f32>()?;
        for (yi, xi) in y.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            assert!((yi - xi).abs() < 1e-5, "pos-0 rope not identity: {yi} vs {xi}");
        }
        Ok(())
    }

    #[test]
    fn interleaved_rope_pos1_rotates_pairs() -> Result<()> {
        let dev = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 8, 10000.0, &dev)?;
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 1.0, 0.0], (1, 1, 1, 4), &dev)?;
        // pos=1, pair 0 freq=1.0 → angle 1 rad; pair 1 freq=10000^-1 → tiny angle.
        let y = rope.apply(&x, 1)?.flatten_all()?.to_vec1::<f32>()?;
        // Pair 0: (x1=1,x2=0) → (cos1, sin1).
        assert!((y[0] - 1.0f32.cos()).abs() < 1e-5);
        assert!((y[1] - 1.0f32.sin()).abs() < 1e-5);
        // Pair 1: angle ≈ 0.01 → ≈ (cos, sin).
        let a1 = (1.0f64 / 10000f64.powf(2.0 / 4.0)) as f32; // freq for pair 1, pos 1
        assert!((y[2] - a1.cos()).abs() < 1e-4);
        assert!((y[3] - a1.sin()).abs() < 1e-4);
        Ok(())
    }
}
