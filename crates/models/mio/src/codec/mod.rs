//! MioCodec-25Hz-24kHz wave decoder: FSQ content-token indices + a 128-d global embedding →
//! 24 kHz waveform. Port of `Aratako/MioCodec`'s `MioCodecModel.forward_wave` (the
//! `use_wave_decoder: true` path; no external vocoder). Decode is a single non-autoregressive pass.

mod fsq;
mod istft;
mod resnet;
mod transformer;

use crate::config::CodecConfig;
use crate::weights::hf_file;
use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{conv_transpose1d, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use fsq::Fsq;
use istft::IstftHead;
use resnet::ResNetStack;
use transformer::Transformer;

pub(crate) const CODEC_REPO: &str = "Aratako/MioCodec-25Hz-24kHz";

/// Intermediate decode tensors, for stage-by-stage parity checks against the Python golden.
pub struct Stages {
    pub content: Tensor,  // (T, 768)   FSQ-decoded content embedding
    pub prenet: Tensor,   // (1, T, 512) wave_prenet output
    pub decoder: Tensor,  // (1, L, 512) wave_decoder output
    pub istft_in: Tensor, // (1, L, 512) wave_post_net output (ISTFT head input)
    pub wav: Tensor,      // (samples,)
}

/// The MioCodec wave decoder.
pub struct MioCodec {
    fsq: Fsq,
    prenet: Transformer,
    conv_upsample: ConvTranspose1d,
    prior_net: ResNetStack,
    decoder: Transformer,
    post_net: ResNetStack,
    istft_head: IstftHead,
    cfg: CodecConfig,
    device: Device,
}

impl MioCodec {
    /// Load the 24 kHz codec from the HF cache / hub (honours `HF_HOME`).
    pub fn from_hf(device: &Device) -> anyhow::Result<Self> {
        let path = hf_file(CODEC_REPO, "model.safetensors")?;
        Ok(Self::from_safetensors(path, device)?)
    }

    pub fn from_safetensors(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let cfg = CodecConfig::miocodec_24khz();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path.as_ref().to_path_buf()], DType::F32, device)?
        };
        let dim = cfg.wave_decoder_dim;
        let up_cfg = ConvTranspose1dConfig { stride: cfg.wave_upsample_factor, ..Default::default() };
        Ok(Self {
            fsq: Fsq::load(vb.pp("local_quantizer"), &cfg.fsq_levels, cfg.fsq_output_dim)?,
            prenet: Transformer::load(vb.pp("wave_prenet"), &cfg.prenet, device)?,
            conv_upsample: conv_transpose1d(
                dim,
                dim,
                cfg.wave_upsample_factor,
                up_cfg,
                vb.pp("wave_conv_upsample"),
            )?,
            prior_net: ResNetStack::load(
                vb.pp("wave_prior_net"),
                dim,
                cfg.wave_resnet_num_blocks,
                cfg.wave_resnet_kernel_size,
                cfg.wave_resnet_num_groups,
            )?,
            decoder: Transformer::load(vb.pp("wave_decoder"), &cfg.decoder, device)?,
            post_net: ResNetStack::load(
                vb.pp("wave_post_net"),
                dim,
                cfg.wave_resnet_num_blocks,
                cfg.wave_resnet_kernel_size,
                cfg.wave_resnet_num_groups,
            )?,
            istft_head: IstftHead::load(dim, cfg.n_fft, cfg.hop_length, vb.pp("istft_head"))?,
            cfg,
            device: device.clone(),
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.sample_rate
    }

    /// Target STFT frame count for a given audio length (samples), "same" padding, no upsampler.
    fn stft_length(&self, target_audio_len: usize) -> usize {
        target_audio_len / self.cfg.hop_length
    }

    /// Decode content-token `indices` `(T,)` + `global` `(128,)` to a `(samples,)` waveform of the
    /// requested length.
    pub fn decode(&self, indices: &Tensor, global: &Tensor, target_audio_len: usize) -> Result<Tensor> {
        Ok(self.decode_stages(indices, global, target_audio_len)?.wav)
    }

    /// Like [`decode`](Self::decode) but also returns the stage intermediates (for parity tests).
    pub fn decode_stages(
        &self,
        indices: &Tensor,
        global: &Tensor,
        target_audio_len: usize,
    ) -> Result<Stages> {
        let content = self.fsq.decode(indices, &self.device)?; // (T, 768)
        let global = global.reshape((1, 1, self.cfg.global_dim))?; // condition

        let x = content.unsqueeze(0)?; // (1, T, 768)
        let prenet = self.prenet.forward(&x, None)?; // (1, T, 512)

        // Conv-transpose ×2 upsample, then linear-interpolate to the exact STFT frame count.
        let x = self.conv_upsample.forward(&prenet.transpose(1, 2)?)?; // (1, 512, 2T)
        let stft_len = self.stft_length(target_audio_len);
        let x = interp1d_linear(&x, stft_len)?; // (1, 512, L)

        let x = self.prior_net.forward(&x)?; // (1, 512, L)
        let x = x.transpose(1, 2)?.contiguous()?; // (1, L, 512)
        let decoder = self.decoder.forward(&x, Some(&global))?; // (1, L, 512)

        let x = self.post_net.forward(&decoder.transpose(1, 2)?.contiguous()?)?; // (1, 512, L)
        let istft_in = x.transpose(1, 2)?.contiguous()?; // (1, L, 512)
        let wav = self.istft_head.forward(&istft_in)?.squeeze(0)?; // (samples,)

        Ok(Stages { content, prenet, decoder, istft_in, wav })
    }
}

/// 1-D linear interpolation along the last axis, matching `F.interpolate(mode="linear",
/// align_corners=False)`. `x`: `(B, C, L_in)` → `(B, C, out_len)`.
fn interp1d_linear(x: &Tensor, out_len: usize) -> Result<Tensor> {
    let (b, c, lin) = x.dims3()?;
    if lin == out_len {
        return Ok(x.clone());
    }
    let scale = lin as f64 / out_len as f64;
    let mut w = vec![0f32; out_len * lin];
    for i in 0..out_len {
        let src = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(lin - 1);
        let frac = (src - i0 as f64) as f32;
        w[i * lin + i0] += 1.0 - frac;
        w[i * lin + i1] += frac;
    }
    let w = Tensor::from_vec(w, (out_len, lin), x.device())?; // (out, lin)
    let o = x.reshape((b * c, lin))?.matmul(&w.t()?)?; // (B·C, out)
    o.reshape((b, c, out_len))
}
