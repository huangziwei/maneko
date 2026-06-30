//! torchaudio-compatible sinc resampling 24 kHz → 16 kHz (the rate the WavLM SSL model expects) plus
//! the encoder's input padding. Ports `torchaudio.functional.resample` (hann sinc-interp, the default)
//! and `MioCodecModel._calculate_waveform_padding`. With `gcd(24000,16000)=8000` the ratio reduces to
//! orig=3 / new=2, so the kernel is a `(2, 1, 23)` stride-3 polyphase filter.

use candle_core::{Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Module};

const ORIG: usize = 3; // 24000 / gcd
const NEW: usize = 2; // 16000 / gcd
const LPW: f64 = 6.0; // lowpass_filter_width
const ROLLOFF: f64 = 0.99;

/// WavLM conv stack `(kernel, stride)` — used to size the encoder's input padding.
const CONV_CFG: [(i64, i64); 7] = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)];

/// Minimum 16 kHz input length that yields `out` feature frames (reverses the conv stack).
fn min_input_length(out: i64) -> i64 {
    let mut l = out;
    for &(k, s) in CONV_CFG.iter().rev() {
        l = (l - 1) * s + k;
    }
    l
}

/// `MioCodecModel._calculate_waveform_padding`: zeros added each side of the 24 kHz waveform so the
/// resampled+convolved length lands exactly on a whole number of SSL frames.
pub fn ssl_padding(length_24k: usize) -> usize {
    let after_resample = length_24k as f64 / 24_000.0 * 16_000.0;
    let expected_out = (after_resample / 320.0).ceil() as i64; // hop_size = 320
    let min_input_24k = min_input_length(expected_out) as f64 / 16_000.0 * 24_000.0;
    ((min_input_24k - length_24k as f64) / 2.0).ceil() as usize
}

/// `_get_sinc_resample_kernel` (hann): row-major `(NEW, kernel_width)` filter taps + the edge `width`.
/// Computed in f64 (as torch does) then stored f32.
fn sinc_kernel() -> (Vec<f32>, usize, usize) {
    let base_freq = (ORIG.min(NEW) as f64) * ROLLOFF; // 1.98
    let width = (LPW * ORIG as f64 / base_freq).ceil() as usize; // 10
    let kw = 2 * width + ORIG; // 23
    let scale = base_freq / ORIG as f64;
    let pi = std::f64::consts::PI;
    let mut k = Vec::with_capacity(NEW * kw);
    for j in 0..NEW {
        let toff = -(j as f64) / NEW as f64; // arange(0, -new, -1) / new
        for i in 0..kw {
            let idx = (i as i64 - width as i64) as f64 / ORIG as f64; // arange(-width, width+orig) / orig
            let mut t = ((toff + idx) * base_freq).clamp(-LPW, LPW);
            let window = (t * pi / LPW / 2.0).cos().powi(2);
            t *= pi;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            k.push((sinc * window * scale) as f32);
        }
    }
    (k, width, kw)
}

/// Resample a `(1, L)` 24 kHz waveform to 16 kHz, matching `torchaudio.transforms.Resample`.
pub fn resample_24k_to_16k(wav: &Tensor) -> Result<Tensor> {
    let length = wav.dim(D::Minus1)?;
    let (k, width, kw) = sinc_kernel();
    let kernel = Tensor::from_vec(k, (NEW, 1, kw), wav.device())?;
    let padded = wav.pad_with_zeros(D::Minus1, width, width + ORIG)?.unsqueeze(1)?; // (1, 1, L')
    let conv = Conv1d::new(kernel, None, Conv1dConfig { stride: ORIG, ..Default::default() });
    let r = conv.forward(&padded)?; // (1, NEW, frames)
    let frames = r.dim(2)?;
    let r = r.transpose(1, 2)?.reshape((1, frames * NEW))?; // interleave the NEW phases over time
    let target = (NEW * length).div_ceil(ORIG);
    r.narrow(D::Minus1, 0, target)
}
