//! ISTFT head: predict log-magnitude + phase, then inverse-STFT to a waveform. Port of
//! `Aratako/MioCodec`'s `ISTFTHead`/`ISTFT` (`module/istft_head.py`, "same" padding). The inverse
//! real FFT uses `realfft` (unnormalized → divided by `n_fft` to match torch's `norm="backward"`).

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;
use std::f32::consts::PI;

pub struct IstftHead {
    out: Linear, // dim -> n_fft + 2  (mag[n_fft/2+1] ++ phase[n_fft/2+1])
    n_fft: usize,
    hop: usize,
    window: Vec<f32>, // periodic Hann, length n_fft (= win_length)
}

impl IstftHead {
    pub fn load(dim: usize, n_fft: usize, hop: usize, vb: VarBuilder) -> Result<Self> {
        let out = candle_nn::linear(dim, n_fft + 2, vb.pp("out"))?;
        let window = (0..n_fft)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / n_fft as f32).cos())
            .collect();
        Ok(Self { out, n_fft, hop, window })
    }

    /// `x`: `(B, L, dim)` → waveform `(B, samples)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.out.forward(x)?.transpose(1, 2)?.contiguous()?; // (B, n_fft+2, L)
        let freq = self.n_fft / 2 + 1;
        let mag = h.narrow(1, 0, freq)?.exp()?.clamp(0f32, 100f32)?; // log-mag → mag, clamped
        let phase = h.narrow(1, freq, freq)?;
        let real = (&mag * phase.cos()?)?; // (B, freq, L)
        let imag = (&mag * phase.sin()?)?;
        self.istft(&real, &imag)
    }

    /// Inverse STFT with "same" padding (overlap-add of windowed irfft frames, envelope-normalized).
    fn istft(&self, real: &Tensor, imag: &Tensor) -> Result<Tensor> {
        let (b, freq, l) = real.dims3()?;
        let n = self.n_fft;
        let pad = (n - self.hop) / 2; // win_length == n_fft
        let out_size = (l - 1) * self.hop + n;
        let final_len = out_size - 2 * pad;

        let re = real.to_dtype(candle_core::DType::F32)?.to_vec3::<f32>()?;
        let im = imag.to_dtype(candle_core::DType::F32)?.to_vec3::<f32>()?;

        let mut planner = RealFftPlanner::<f32>::new();
        let c2r = planner.plan_fft_inverse(n);
        let mut spectrum = c2r.make_input_vec(); // freq complex
        let mut frame = c2r.make_output_vec(); // n real

        // Window envelope (∑ window² over the overlap) is signal-independent → compute once.
        let mut env = vec![0f32; out_size];
        for start in (0..).map(|f| f * self.hop).take(l) {
            for k in 0..n {
                env[start + k] += self.window[k] * self.window[k];
            }
        }

        let mut out = vec![0f32; b * final_len];
        for bi in 0..b {
            let mut acc = vec![0f32; out_size];
            for li in 0..l {
                for f in 0..freq {
                    spectrum[f] = Complex::new(re[bi][f][li], im[bi][f][li]);
                }
                // Hermitian: DC and Nyquist are real (matches torch.fft.irfft).
                spectrum[0].im = 0.0;
                spectrum[freq - 1].im = 0.0;
                c2r.process(&mut spectrum, &mut frame).expect("irfft");
                let start = li * self.hop;
                for k in 0..n {
                    // realfft inverse is unnormalized → /n for torch "backward" norm; then × window.
                    acc[start + k] += (frame[k] / n as f32) * self.window[k];
                }
            }
            for j in 0..final_len {
                out[bi * final_len + j] = acc[pad + j] / env[pad + j];
            }
        }
        Tensor::from_vec(out, (b, final_len), real.device())
    }
}
