//! Finite Scalar Quantizer decode: content-token index → continuous embedding.
//! Port of `Aratako/MioCodec`'s `module/fsq.py` (`FiniteScalarQuantizer.decode`): unpack the index
//! into per-axis codes, recenter to `[-1, 1]`, then `proj_out` to the model dim. No learned codebook.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct Fsq {
    proj_out: Linear, // len(levels) -> output_dim
    levels: Vec<i64>,
    basis: Vec<i64>, // cumprod([1, levels[:-1]])
    half: Vec<f32>,  // levels // 2
}

impl Fsq {
    pub fn load(vb: VarBuilder, levels: &[u32], output_dim: usize) -> Result<Self> {
        let proj_out = candle_nn::linear(levels.len(), output_dim, vb.pp("proj_out"))?;
        let levels: Vec<i64> = levels.iter().map(|&l| l as i64).collect();
        let mut basis = vec![1i64; levels.len()];
        for j in 1..levels.len() {
            basis[j] = basis[j - 1] * levels[j - 1];
        }
        let half = levels.iter().map(|&l| (l / 2) as f32).collect();
        Ok(Self { proj_out, levels, basis, half })
    }

    /// `indices`: `(T,)` int64 → content embedding `(T, output_dim)`.
    pub fn decode(&self, indices: &Tensor, device: &Device) -> Result<Tensor> {
        let idx = indices.to_dtype(candle_core::DType::I64)?.to_vec1::<i64>()?;
        let d = self.levels.len();
        let t = idx.len();
        let mut z = vec![0f32; t * d];
        for (i, &id) in idx.iter().enumerate() {
            for j in 0..d {
                // indices_to_codes: codes = (idx // basis) % levels; then (codes - half)/half.
                let code = (id / self.basis[j]) % self.levels[j];
                z[i * d + j] = (code as f32 - self.half[j]) / self.half[j];
            }
        }
        let z = Tensor::from_vec(z, (t, d), device)?;
        self.proj_out.forward(&z)
    }
}
