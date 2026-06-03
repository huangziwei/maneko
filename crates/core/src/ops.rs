//! Model-agnostic tensor ops shared by maneko's engines.
//!
//! All numerically-sensitive ops (Snake, RMSNorm) accumulate in **f32** regardless of the
//! input dtype, then cast back — matching the reference MLX/torch implementations, where a
//! near-zero `alpha` in fp16 (`1/(alpha+eps)`) or a tiny RMS would otherwise blow up.

use candle_core::{DType, Result, Tensor, D};

/// Snake activation: `x + sin²(αx) / (α + 1e-9)`, per-channel `α`.
///
/// `alpha` is broadcast against `x` (e.g. `x` is `(B, C, L)` and `alpha` is `(1, C, 1)`).
/// Computed in f32 (α≈0 in fp16 makes the reciprocal explode).
pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let a = alpha.to_dtype(DType::F32)?;
    let recip = (&a + 1e-9)?.recip()?;
    let s = a.broadcast_mul(&x)?.sin()?.sqr()?;
    let out = (&x + recip.broadcast_mul(&s)?)?;
    out.to_dtype(dt)
}

/// RMSNorm over the last dim: `x * rsqrt(mean(x²) + eps) * weight`, accumulated in f32.
///
/// `weight` must be broadcastable against `x`'s last dim. For per-head norms (q/k) where
/// `weight` is `(heads, head_dim)` and `x` is `(B, heads, T, head_dim)`, reshape `weight` to
/// `(heads, 1, head_dim)` at the call site so broadcasting lines the head axes up.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps)?.recip()?.sqrt()?; // 1/sqrt(var+eps)
    let normed = x.broadcast_mul(&inv)?;
    normed
        .broadcast_mul(&weight.to_dtype(DType::F32)?)?
        .to_dtype(dt)
}

/// Scaled dot-product attention with an optional **additive** mask.
///
/// `q`: `(B, H, T, Dh)`, `k`/`v`: `(B, H, S, Dh)`. `scale` is applied to the logits
/// (typically `1/sqrt(Dh)`). `mask`, if given, is broadcast-added to the `(B, H, T, S)`
/// logits before softmax (use `-inf`/`-1e9` for masked positions). Returns `(B, H, T, Dh)`.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, mask: Option<&Tensor>) -> Result<Tensor> {
    let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let mut att = (q.contiguous()?.matmul(&k_t)? * scale)?;
    if let Some(m) = mask {
        att = att.broadcast_add(m)?;
    }
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn snake_matches_formula() -> Result<()> {
        let dev = Device::Cpu;
        // x = [0.5, -1.0] on one channel, alpha = 2.0
        let x = Tensor::from_vec(vec![0.5f32, -1.0], (1, 1, 2), &dev)?;
        let alpha = Tensor::from_vec(vec![2.0f32], (1, 1, 1), &dev)?;
        let y = snake(&x, &alpha)?.flatten_all()?.to_vec1::<f32>()?;
        for (xi, yi) in [0.5f32, -1.0].iter().zip(y) {
            let expect = xi + (1.0 / (2.0 + 1e-9)) * (2.0 * xi).sin().powi(2);
            assert!((expect - yi).abs() < 1e-5, "snake mismatch {expect} vs {yi}");
        }
        Ok(())
    }

    #[test]
    fn rms_norm_unit_weight() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &dev)?;
        let w = Tensor::ones((2,), DType::F32, &dev)?;
        let y = rms_norm(&x, &w, 0.0)?.flatten_all()?.to_vec1::<f32>()?;
        // rms = sqrt((9+16)/2) = sqrt(12.5); normalized = x/rms
        let rms = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-5);
        assert!((y[1] - 4.0 / rms).abs() < 1e-5);
        Ok(())
    }
}
