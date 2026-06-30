//! MioCodec-25Hz-24kHz decode config (from the upstream `config.yaml`, `use_wave_decoder: true`).

/// One MioCodec transformer stack (`wave_prenet` / `wave_decoder`): a Llama-3-style block with
/// interleaved RoPE, SwiGLU FFN, banded windowed (non-causal) attention, and either plain
/// LayerNorm or AdaLN-Zero conditioning.
#[derive(Clone, Debug)]
pub struct TfConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// Odd local-attention window (band half-width = `window/2`); `None` = full attention.
    pub window_size: Option<usize>,
    pub rope_theta: f64,
    pub max_seq_len: usize,
    /// Final `output_proj` target dim (`wave_prenet` projects 768→512); `None` = no projection.
    pub output_dim: Option<usize>,
    /// AdaLN-Zero conditioning dim (the 128-d global embedding); `None` = plain LayerNorm.
    pub adanorm_condition_dim: Option<usize>,
    pub use_adaln_zero: bool,
    pub multiple_of: usize,
    pub norm_eps: f64,
}

/// The full MioCodec-25Hz-24kHz wave-decode configuration.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub fsq_levels: Vec<u32>,
    pub fsq_output_dim: usize,
    pub wave_upsample_factor: usize,
    pub wave_decoder_dim: usize,
    pub wave_resnet_num_blocks: usize,
    pub wave_resnet_kernel_size: usize,
    pub wave_resnet_num_groups: usize,
    pub global_dim: usize,
    pub prenet: TfConfig,
    pub decoder: TfConfig,
}

impl CodecConfig {
    /// `Aratako/MioCodec-25Hz-24kHz` — the values from its `config.yaml`.
    pub fn miocodec_24khz() -> Self {
        Self {
            sample_rate: 24_000,
            n_fft: 1920,
            hop_length: 480,
            fsq_levels: vec![8, 8, 8, 5, 5], // 12800 codes
            fsq_output_dim: 768,
            wave_upsample_factor: 2,
            wave_decoder_dim: 512,
            wave_resnet_num_blocks: 2,
            wave_resnet_kernel_size: 3,
            wave_resnet_num_groups: 32,
            global_dim: 128,
            prenet: TfConfig {
                dim: 768,
                n_layers: 6,
                n_heads: 12,
                window_size: Some(65),
                rope_theta: 10_000.0,
                max_seq_len: 512,
                output_dim: Some(512),
                adanorm_condition_dim: None,
                use_adaln_zero: false,
                multiple_of: 256,
                norm_eps: 1e-5,
            },
            decoder: TfConfig {
                dim: 512,
                n_layers: 8,
                n_heads: 8,
                window_size: Some(65),
                rope_theta: 10_000.0,
                max_seq_len: 512,
                output_dim: None,
                adanorm_condition_dim: Some(128),
                use_adaln_zero: true,
                multiple_of: 256,
                norm_eps: 1e-5,
            },
        }
    }
}

/// MioTTS-0.1B AR backbone (`FalconH1ForCausalLM`) — from its `config.json`. Every layer runs a
/// Mamba-2 mixer and GQA attention *in parallel* off a shared RMSNorm, summed into the residual.
/// All in/out/key/ssm/mlp multipliers are 1.0; only `embedding_multiplier` and `lm_head_multiplier`
/// are non-trivial.
#[derive(Clone, Debug)]
pub struct FalconH1Config {
    pub hidden_size: usize,        // 512
    pub num_layers: usize,         // 24
    pub vocab_size: usize,         // 78336
    pub intermediate_size: usize,  // 768 (MLP)
    pub num_heads: usize,          // 8
    pub num_kv_heads: usize,       // 2
    pub head_dim: usize,           // 64
    pub rope_theta: f64,           // 1e11
    pub rms_eps: f64,              // 1e-5
    pub embedding_multiplier: f64, // 0.123046875
    pub lm_head_multiplier: f64,   // 0.078125
    pub mamba_d_ssm: usize,        // 768
    pub mamba_d_state: usize,      // 64
    pub mamba_d_conv: usize,       // 4
    pub mamba_n_heads: usize,      // 24
    pub mamba_d_head: usize,       // 32
    pub mamba_n_groups: usize,     // 1
}

impl FalconH1Config {
    pub fn miotts_0_1b() -> Self {
        Self {
            hidden_size: 512,
            num_layers: 24,
            vocab_size: 78_336,
            intermediate_size: 768,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            rope_theta: 100_000_000_000.0,
            rms_eps: 1e-5,
            embedding_multiplier: 0.123_046_875,
            lm_head_multiplier: 0.078_125,
            mamba_d_ssm: 768,
            mamba_d_state: 64,
            mamba_d_conv: 4,
            mamba_n_heads: 24,
            mamba_d_head: 32,
            mamba_n_groups: 1,
        }
    }

    /// Mamba conv1d channel count = `d_ssm + 2·n_groups·d_state`.
    pub fn conv_dim(&self) -> usize {
        self.mamba_d_ssm + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}
