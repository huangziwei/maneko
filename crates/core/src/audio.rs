//! Minimal WAV I/O + resampling shared across engines.

use candle_core::{DType, Device, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

/// Read a WAV into `(channels, samples)` f32 in `[-1, 1]`, plus its sample rate.
pub fn read_wav<P: AsRef<Path>>(path: P) -> anyhow::Result<(Tensor, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
        SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
    };
    let n = samples.len() / channels.max(1);
    // De-interleave into (channels, samples).
    let mut data = vec![0f32; channels * n];
    for (i, s) in samples.iter().enumerate() {
        let (c, t) = (i % channels, i / channels);
        data[c * n + t] = *s;
    }
    let t = Tensor::from_vec(data, (channels, n), &Device::Cpu)?;
    Ok((t, spec.sample_rate))
}

/// Resample `(channels, samples)` audio from `from` to `to` Hz (high-quality polynomial).
pub fn resample(audio: &Tensor, from: u32, to: u32) -> anyhow::Result<Tensor> {
    if from == to {
        return Ok(audio.clone());
    }
    use rubato::{FastFixedIn, PolynomialDegree, Resampler};
    let (channels, n) = audio.dims2()?;
    let input = audio.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut resampler =
        FastFixedIn::<f32>::new(to as f64 / from as f64, 1.0, PolynomialDegree::Septic, n, channels)?;
    let out = resampler.process(&input, None)?;
    let m = out[0].len();
    let flat: Vec<f32> = out.into_iter().flatten().collect();
    Ok(Tensor::from_vec(flat, (channels, m), audio.device())?)
}

/// Write a 16-bit PCM WAV. `audio` is `(samples,)` mono or `(channels, samples)`; values are
/// clamped to `[-1, 1]` then scaled to i16.
pub fn write_wav<P: AsRef<Path>>(path: P, audio: &Tensor, sample_rate: u32) -> anyhow::Result<()> {
    let audio = match audio.rank() {
        1 => audio.reshape((1, audio.elem_count()))?,
        2 => audio.clone(),
        n => anyhow::bail!("write_wav expects 1-D or 2-D [channels, samples], got {n}-D"),
    };
    let (channels, samples) = audio.dims2()?;
    let data = audio.to_dtype(DType::F32)?.to_vec2::<f32>()?;

    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for i in 0..samples {
        for ch in data.iter() {
            let v = (ch[i].clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(v)?;
        }
    }
    writer.finalize()?;
    Ok(())
}
