//! Minimal WAV output shared across engines.

use candle_core::{DType, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::Path;

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
