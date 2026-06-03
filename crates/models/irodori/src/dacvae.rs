//! DACVAE decoder (synthesis path): 32-dim VAE latent → 48 kHz waveform.
//!
//! A DAC/SEANet conv vocoder with **Snake** activations (no iSTFT, no LSTM, no ELU on the main
//! path). Port of `ref/mlx-audio/.../codec/models/dacvae/codec.py`, decode path only. We work in
//! Candle's native **NCL** layout `(B, C, L)` throughout — the torch `.pth` conv weights are
//! already `(out,in,k)` / `(in,out,k)` and Snake α is `(1, C, 1)`, so no permutation is needed.
//!
//! Per latent frame the decoder upsamples ×∏`decoder_rates` = ×1920 (strides 12·10·8·2) at 48 kHz.
//! Watermarker / LSTM / message paths in the checkpoint are unused for synthesis and skipped.

use crate::config::DacvaeConfig;
use crate::weights::Weights;
use candle_core::{Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module};
use tts_core::{fold_weight_norm, snake};

/// Build a plain Conv1d from a weight-normed torch conv (`{base}.weight_g/.weight_v/.bias`).
fn wn_conv1d(w: &Weights, base: &str, padding: usize, stride: usize, dilation: usize) -> Result<Conv1d> {
    let weight = fold_weight_norm(
        w.get(&format!("{base}.weight_g"))?,
        w.get(&format!("{base}.weight_v"))?,
    )?;
    let bias = w.get(&format!("{base}.bias"))?.clone();
    let cfg = Conv1dConfig {
        padding,
        stride,
        dilation,
        groups: 1,
        ..Default::default()
    };
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// Build a plain ConvTranspose1d for the `pad_mode="none"` upsample: `padding=(stride+1)/2`,
/// `output_padding=0`, `kernel=2·stride` → exact ×stride upsample (no `_unpad` since pad_mode none).
fn wn_convtr1d(w: &Weights, base: &str, stride: usize) -> Result<ConvTranspose1d> {
    let weight = fold_weight_norm(
        w.get(&format!("{base}.weight_g"))?,
        w.get(&format!("{base}.weight_v"))?,
    )?;
    let bias = w.get(&format!("{base}.bias"))?.clone();
    let cfg = ConvTranspose1dConfig {
        padding: (stride + 1) / 2,
        output_padding: 0,
        stride,
        dilation: 1,
        groups: 1,
    };
    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

/// Center-crop `x` (NCL) along length to match `len`, matching codec.py's residual alignment.
fn center_crop_to(x: &Tensor, len: usize) -> Result<Tensor> {
    let cur = x.dim(D::Minus1)?;
    if cur == len {
        return Ok(x.clone());
    }
    let pad = (cur - len) / 2;
    x.narrow(D::Minus1, pad, len)
}

/// Snake → WNConv1d(k=7, dilation=d) → Snake → WNConv1d(k=1), residual.
struct ResidualUnit {
    alpha1: Tensor,
    conv1: Conv1d,
    alpha2: Tensor,
    conv2: Conv1d,
}

impl ResidualUnit {
    /// `base` points at the torch `block` Sequential (sub-indices 0=Snake,1=conv k7,2=Snake,3=conv k1).
    fn load(w: &Weights, base: &str, dilation: usize) -> Result<Self> {
        let pad = (7 - 1) * dilation / 2; // pad_mode="none" → 3·dilation, preserves length
        Ok(Self {
            alpha1: w.get(&format!("{base}.0.alpha"))?.clone(),
            conv1: wn_conv1d(w, &format!("{base}.1"), pad, 1, dilation)?,
            alpha2: w.get(&format!("{base}.2.alpha"))?.clone(),
            conv2: wn_conv1d(w, &format!("{base}.3"), 0, 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = snake(x, &self.alpha1)?;
        let y = self.conv1.forward(&y)?;
        let y = snake(&y, &self.alpha2)?;
        let y = self.conv2.forward(&y)?;
        let x = center_crop_to(x, y.dim(D::Minus1)?)?;
        x + y
    }
}

/// Decoder main path: Snake → ConvTranspose(up) → ResidualUnit(d=1,3,9).
struct DecoderBlock {
    alpha0: Tensor,
    convt: ConvTranspose1d,
    res1: ResidualUnit,
    res3: ResidualUnit,
    res9: ResidualUnit,
}

impl DecoderBlock {
    /// `base` is the torch `decoder.model.{i+1}` prefix (its `.block` holds the sub-modules).
    fn load(w: &Weights, base: &str, stride: usize) -> Result<Self> {
        Ok(Self {
            alpha0: w.get(&format!("{base}.block.0.alpha"))?.clone(),
            convt: wn_convtr1d(w, &format!("{base}.block.1"), stride)?,
            res1: ResidualUnit::load(w, &format!("{base}.block.4.block"), 1)?,
            res3: ResidualUnit::load(w, &format!("{base}.block.5.block"), 3)?,
            res9: ResidualUnit::load(w, &format!("{base}.block.8.block"), 9)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = snake(x, &self.alpha0)?;
        let x = self.convt.forward(&x)?;
        let x = self.res1.forward(&x)?;
        let x = self.res3.forward(&x)?;
        self.res9.forward(&x)
    }
}

/// Encoder downsampling block: ResidualUnit(d=1,3,9) → Snake → strided WNConv1d.
struct EncoderBlock {
    res1: ResidualUnit,
    res3: ResidualUnit,
    res9: ResidualUnit,
    snake: Tensor,
    conv: Conv1d,
}

impl EncoderBlock {
    /// `base` is the torch `encoder.block.{i+1}` prefix (its `.block` Sequential holds res×3,
    /// Snake, downsample conv at sub-indices 0,1,2,3,4).
    fn load(w: &Weights, base: &str, stride: usize) -> Result<Self> {
        Ok(Self {
            res1: ResidualUnit::load(w, &format!("{base}.block.0.block"), 1)?,
            res3: ResidualUnit::load(w, &format!("{base}.block.1.block"), 3)?,
            res9: ResidualUnit::load(w, &format!("{base}.block.2.block"), 9)?,
            snake: w.get(&format!("{base}.block.3.alpha"))?.clone(),
            // downsample: k=2·stride, stride, padding=ceil(stride/2).
            conv: wn_conv1d(w, &format!("{base}.block.4"), stride.div_ceil(2), stride, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.res1.forward(x)?;
        let x = self.res3.forward(&x)?;
        let x = self.res9.forward(&x)?;
        let x = snake(&x, &self.snake)?;
        self.conv.forward(&x)
    }
}

/// DACVAE encoder: `conv_in` → N `EncoderBlock`s → `snake_out` → `conv_out`.
struct Encoder {
    conv_in: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake_out: Tensor,
    conv_out: Conv1d,
}

impl Encoder {
    fn load(w: &Weights, cfg: &DacvaeConfig) -> Result<Self> {
        let conv_in = wn_conv1d(w, "encoder.block.0", 3, 1, 1)?; // 1 → encoder_dim, k=7
        let blocks = cfg
            .encoder_rates
            .iter()
            .enumerate()
            .map(|(i, &stride)| EncoderBlock::load(w, &format!("encoder.block.{}", i + 1), stride))
            .collect::<Result<Vec<_>>>()?;
        let n = cfg.encoder_rates.len();
        let snake_out = w.get(&format!("encoder.block.{}.alpha", n + 1))?.clone();
        let conv_out = wn_conv1d(w, &format!("encoder.block.{}", n + 2), 1, 1, 1)?; // →latent_dim, k=3
        Ok(Self {
            conv_in,
            blocks,
            snake_out,
            conv_out,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = snake(&x, &self.snake_out)?;
        self.conv_out.forward(&x)
    }
}

/// The DACVAE: encode (ref audio → latent) and the synthesis stack (`quantizer_out_proj` →
/// `conv_in` → N `DecoderBlock`s → `snake_out` → `conv_out` → tanh).
pub struct Dacvae {
    encoder: Encoder,
    quantizer_in_proj: Conv1d,
    quantizer_out_proj: Conv1d,
    conv_in: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake_out: Tensor,
    conv_out: Conv1d,
    cfg: DacvaeConfig,
}

impl Dacvae {
    pub fn load(w: &Weights, cfg: DacvaeConfig) -> Result<Self> {
        // Encoder (ref-audio → latent) + VAE input projection (latent_dim → 2·codebook_dim, k=1).
        let encoder = Encoder::load(w, &cfg)?;
        let quantizer_in_proj = wn_conv1d(w, "quantizer.in_proj", 0, 1, 1)?;
        // codebook_dim (32) → latent_dim (1024), k=1.
        let quantizer_out_proj = wn_conv1d(w, "quantizer.out_proj", 0, 1, 1)?;
        // latent_dim → decoder_dim, k=7.
        let conv_in = wn_conv1d(w, "decoder.model.0", 3, 1, 1)?;

        let mut blocks = Vec::with_capacity(cfg.decoder_rates.len());
        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            // torch decoder is an nn.Sequential `model`: 0 = conv_in, 1.. = DecoderBlocks.
            blocks.push(DecoderBlock::load(w, &format!("decoder.model.{}", i + 1), stride)?);
        }

        // snake_out / conv_out are shared into the (unused) watermark encoder block in the
        // checkpoint: `decoder.wm_model.encoder_block.pre.{0=Snake, 1=conv}`.
        let snake_out = w
            .get("decoder.wm_model.encoder_block.pre.0.alpha")?
            .clone();
        let conv_out = wn_conv1d(w, "decoder.wm_model.encoder_block.pre.1", 3, 1, 1)?;

        Ok(Self {
            encoder,
            quantizer_in_proj,
            quantizer_out_proj,
            conv_in,
            blocks,
            snake_out,
            conv_out,
            cfg,
        })
    }

    /// Right-pad a waveform `(B,1,L)` so `L` is a multiple of `hop_length`.
    fn pad_to_hop(&self, x: &Tensor) -> Result<Tensor> {
        let l = x.dim(D::Minus1)?;
        let hop = self.cfg.hop_length();
        let rem = l % hop;
        if rem == 0 {
            Ok(x.clone())
        } else {
            x.pad_with_zeros(D::Minus1, 0, hop - rem)
        }
    }

    /// Encode a mono waveform `(B, 1, L)` (48 kHz) to a VAE-mean latent `(B, T, codebook_dim)`,
    /// the reference latent for voice cloning. `T = ceil(L / hop_length)`.
    pub fn encode(&self, waveform: &Tensor) -> Result<Tensor> {
        let w = self.pad_to_hop(waveform)?;
        let z = self.encoder.forward(&w)?; // (B, latent_dim, T)
        let proj = self.quantizer_in_proj.forward(&z)?; // (B, 2·codebook_dim, T)
        let mean = proj.narrow(1, 0, self.cfg.codebook_dim)?; // VAE mean = first half: (B, codebook_dim, T)
        mean.transpose(1, 2)?.contiguous() // (B, T, codebook_dim)
    }

    pub fn sample_rate(&self) -> usize {
        self.cfg.sample_rate
    }

    pub fn hop_length(&self) -> usize {
        self.cfg.hop_length()
    }

    /// Load the v2 Japanese DACVAE from the HF cache (`Aratako/Semantic-DACVAE-Japanese-32dim`,
    /// a torch `.pth`). Honors `HF_HOME`.
    pub fn from_hf(device: &candle_core::Device) -> anyhow::Result<Self> {
        let path = crate::weights::hf_file("Aratako/Semantic-DACVAE-Japanese-32dim", "weights.pth")?;
        let w = Weights::from_pth(&path, Some("state_dict"), device)?;
        Ok(Self::load(&w, DacvaeConfig::v2())?)
    }

    /// Decode a VAE latent `(B, codebook_dim, T)` → waveform `(B, 1, T·hop)` in `[-1, 1]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let emb = self.quantizer_out_proj.forward(latent)?; // (B, latent_dim, T)
        let mut x = self.conv_in.forward(&emb)?; // (B, decoder_dim, T)
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = snake(&x, &self.snake_out)?;
        let x = self.conv_out.forward(&x)?; // (B, 1, L)
        x.tanh()
    }
}
