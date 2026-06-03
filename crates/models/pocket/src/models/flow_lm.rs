use crate::ModelState;
use crate::models::transformer::StreamingTransformer;
use crate::modules::mlp::{LayerNorm, ModulationParams, SimpleMLPAdaLN};
use crate::qweights::{QLinear, Vb};
use candle_core::{Result, Tensor};

pub fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    modulations: &[Vec<ModulationParams>],
    x_0: &Tensor,
) -> Result<Tensor> {
    let mut current = x_0.clone();
    let num_steps = modulations.len();

    let step_factor = 1.0 / num_steps as f64;
    for step_mod in modulations {
        // Use forward_step_cached with pre-computed modulation batch for this ODE step
        let flow_dir = flow_net.forward_step_cached(&current, step_mod)?;
        current = (current + flow_dir.affine(step_factor, 0.0)?)?;
    }
    Ok(current)
}

#[derive(Clone)]
pub struct FlowLMModel {
    pub flow_net: SimpleMLPAdaLN,
    pub transformer: StreamingTransformer,
    pub input_linear: QLinear,
    pub out_norm: LayerNorm,
    pub out_eos: QLinear,
    pub bos_emb: Tensor,
    pub emb_mean: Tensor,
    pub emb_std: Tensor,
    pub ldim: usize,
    pub dim: usize,
    pub noise_clamp: Option<f32>,
    /// v2 only: learnable `[1, 1, dim]` token prepended before the voice conditioning. None for v1.
    pub bos_before_voice: Option<Tensor>,
}

fn sample_noise(
    device: &candle_core::Device,
    shape: (usize, usize),
    temp: f32,
    clamp: Option<f32>,
) -> Result<Tensor> {
    let std = temp.sqrt();
    match clamp {
        None => Tensor::randn(0.0f32, std, shape, device),
        Some(limit) => {
            // Rejection sampling for truncated normal
            let count = shape.0 * shape.1;
            let mut data = Vec::with_capacity(count);
            let mut rng = rand::thread_rng();
            let dist = rand_distr::Normal::new(0.0f32, std)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            while data.len() < count {
                let v = rand_distr::Distribution::sample(&dist, &mut rng);
                if v.abs() <= limit {
                    data.push(v);
                }
            }
            Tensor::from_vec(data, shape, device)
        }
    }
}

impl FlowLMModel {
    pub fn new(
        flow_net: SimpleMLPAdaLN,
        transformer: StreamingTransformer,
        ldim: usize,
        dim: usize,
        insert_bos_before_voice: bool,
        vb: Vb,
    ) -> Result<Self> {
        let input_linear = vb.pp("input_linear").qlinear(ldim, dim, false)?;
        let out_norm = LayerNorm::new(dim, 1e-5, true, vb.pp("out_norm"))?;
        let out_eos = vb.pp("out_eos").qlinear(dim, 1, true)?;
        let bos_emb = vb.get(ldim, "bos_emb")?;
        let emb_mean = vb.get(ldim, "emb_mean")?;
        let emb_std = vb.get(ldim, "emb_std")?;
        // v2: learnable token prepended before the voice conditioning (shape [1, 1, dim]).
        let bos_before_voice = if insert_bos_before_voice {
            Some(vb.get((1, 1, dim), "bos_before_voice")?)
        } else {
            None
        };

        Ok(Self {
            flow_net,
            transformer,
            input_linear,
            out_norm,
            out_eos,
            bos_emb,
            emb_mean,
            emb_std,
            ldim,
            dim,
            noise_clamp: None, // Default to no clamp
            bos_before_voice,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        sequence: &Tensor,
        text_embeddings: &Tensor,
        model_state: &mut ModelState,
        time_embeddings: &Tensor,
        temp: f32,
        eos_threshold: f32,
        step: usize,
    ) -> Result<(Tensor, bool)> {
        // sequence is [B, T, ldim]
        // text_embeddings is [B, S, dim]

        // Handle BOS (if NaN, use bos_emb) - simplistic check for NaN
        // In Candle we can use `Tensor::where_cond`
        // But for now let's assume sequence passed in doesn't have NaNs or handled upstream.
        // Original: sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)

        // Let's assume BOS is handled by caller for now or if sequence empty.

        let x = self.input_linear.forward(sequence)?;
        let s_len = text_embeddings.dims()[1];

        // Cat text embeddings and sequence embeddings only if text_embeddings is not empty
        let transformer_out_pre_norm = if s_len > 0 {
            let input = Tensor::cat(&[text_embeddings, &x], 1)?;
            let mut out = self.transformer.forward(&input, model_state, step)?;
            // Remove prefix (text embeddings length)
            out = out.narrow(1, s_len, out.dims()[1] - s_len)?;
            out
        } else {
            self.transformer.forward(&x, model_state, step)?
        };

        let transformer_out = self.out_norm.forward(&transformer_out_pre_norm)?;

        // Only use the last frame for generation
        let last_frame = transformer_out
            .narrow(1, transformer_out.dims()[1] - 1, 1)?
            .squeeze(1)?;

        let eos_score = self
            .out_eos
            .forward(&last_frame)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<f32>()?;
        let is_eos = eos_score > eos_threshold;

        // Generate noise with optional clamping
        let noise = sample_noise(
            last_frame.device(),
            (last_frame.dims()[0], self.ldim),
            temp,
            self.noise_clamp,
        )?;

        // Pre-compute all modulations for this frame's ODE steps (8 steps * N blocks) in batch
        let c_emb = self.flow_net.embed_condition(&last_frame)?;
        let modulations = self
            .flow_net
            .precompute_modulations(&c_emb, time_embeddings)?;

        let next_latent = lsd_decode(&self.flow_net, &modulations, &noise)?;

        Ok((next_latent, is_eos))
    }
}
