//! Weight-normalization fold for convolutional weights loaded from PyTorch.
//!
//! PyTorch stores a weight-normed conv as two tensors: a direction `weight_v` and a per-(dim 0)
//! magnitude `weight_g`. The effective weight is `g · v / ‖v‖`, where the norm is taken over every
//! dim **except dim 0** (keepdim). This holds for both `Conv1d` (`v`: `(out, in, k)`, `g`: `(out,1,1)`)
//! and `ConvTranspose1d` (`v`: `(in, out, k)`, `g`: `(in,1,1)`) because torch aligns `weight_g` to
//! dim 0 in both cases. Folding once at load time keeps the runtime conv a plain conv.

use candle_core::{DType, Result, Tensor};

/// Fold PyTorch weight-norm into a single conv weight: `g · v / ‖v‖`.
///
/// Both inputs are 3-D conv weights. The norm is computed over dims 1 and 2 (keepdim) in f32.
pub fn fold_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    if v.rank() != 3 {
        candle_core::bail!("fold_weight_norm expects a 3-D conv weight, got rank {}", v.rank());
    }
    let v32 = v.to_dtype(DType::F32)?;
    let norm = v32.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?; // (d0, 1, 1)
    let dir = v32.broadcast_div(&norm)?;
    dir.broadcast_mul(&g.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn fold_recovers_unit_direction() -> Result<()> {
        let dev = Device::Cpu;
        // v has rows with known norms; g rescales each output channel to magnitude g.
        // out=2, in=1, k=2: row0=[3,4] (‖‖=5), row1=[0,1] (‖‖=1).
        let v = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 1.0], (2, 1, 2), &dev)?;
        let g = Tensor::from_vec(vec![5.0f32, 2.0], (2, 1, 1), &dev)?;
        let w = fold_weight_norm(&g, &v)?.flatten_all()?.to_vec1::<f32>()?;
        // row0: g=5, dir=[3/5,4/5] → [3,4]; row1: g=2, dir=[0,1] → [0,2]
        assert!((w[0] - 3.0).abs() < 1e-6);
        assert!((w[1] - 4.0).abs() < 1e-6);
        assert!((w[2] - 0.0).abs() < 1e-6);
        assert!((w[3] - 2.0).abs() < 1e-6);
        Ok(())
    }
}
