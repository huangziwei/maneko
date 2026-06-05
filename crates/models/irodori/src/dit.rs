//! Irodori DiT backbone: timestep conditioning + JointAttention + dual LowRankAdaLN → velocity.
//!
//! Port of `IrodoriDiT` / `DiffusionBlock` / `JointAttention` / `LowRankAdaLN` from `model.py`
//! (speaker-conditioned v2). Text and speaker KV are **constant across diffusion steps**, so they
//! are projected once per (state, block) and reused. Self-attention keys/queries get **half-by-
//! heads** interleaved RoPE (first H/2 heads only); cross-attention keys are unrotated.

use crate::blocks::{additive_key_mask, RmsNorm, SwiGlu};
use crate::config::DitConfig;
use crate::duration::DurationPredictor;
use crate::encoders::Encoders;
use candle_core::{DType, Result, Tensor, D};
use tts_core::{sdpa, QLinear, RotaryEmbedding, Vb};

/// Per-block, per-context key/value cache in `(B, H, S, Dh)` layout (`k` is RMS-normed).
type Kv = (Tensor, Tensor);

/// Sinusoidal timestep embedding: `concat(cos, sin)` over `1000 · exp(-ln(10000)·i/half)`.
fn timestep_embedding(t: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let base = 10000f64.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (1000.0 * (-base * i as f64 / half as f64).exp()) as f32)
        .collect();
    let freqs = Tensor::from_vec(freqs, (1, half), t.device())?;
    let args = t.reshape((t.elem_count(), 1))?.broadcast_mul(&freqs)?; // (B, half)
    Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1) // (B, dim)
}

/// Timestep → conditioning embedding: `Linear→SiLU→Linear→SiLU→Linear(→3·model_dim)`, no bias.
struct CondModule {
    l0: QLinear,
    l2: QLinear,
    l4: QLinear,
}

impl CondModule {
    fn load(vb: Vb, in_dim: usize, model_dim: usize) -> Result<Self> {
        Ok(Self {
            l0: vb.pp("0").qlinear(in_dim, model_dim, false)?,
            l2: vb.pp("2").qlinear(model_dim, model_dim, false)?,
            l4: vb.pp("4").qlinear(model_dim, model_dim * 3, false)?,
        })
    }

    /// `t_embed`: `(B, in_dim)` → `(B, 1, 3·model_dim)`.
    fn forward(&self, t_embed: &Tensor) -> Result<Tensor> {
        let h = candle_nn::ops::silu(&self.l0.forward(t_embed)?)?;
        let h = candle_nn::ops::silu(&self.l2.forward(&h)?)?;
        self.l4.forward(&h)?.unsqueeze(1)
    }
}

/// Low-rank adaptive LayerNorm: split `cond_embed` into shift/scale/gate, each `up(silu(down(·)))+·`
/// (residual); weightless RMS-norm of `x`, then `x·(1+scale)+shift`; `gate = tanh(gate)`.
struct LowRankAdaLN {
    shift_down: QLinear,
    scale_down: QLinear,
    gate_down: QLinear,
    shift_up: QLinear,
    scale_up: QLinear,
    gate_up: QLinear,
    eps: f64,
}

impl LowRankAdaLN {
    fn load(vb: Vb, model_dim: usize, rank: usize, eps: f64) -> Result<Self> {
        let rank = rank.clamp(1, model_dim);
        Ok(Self {
            shift_down: vb.pp("shift_down").qlinear(model_dim, rank, false)?,
            scale_down: vb.pp("scale_down").qlinear(model_dim, rank, false)?,
            gate_down: vb.pp("gate_down").qlinear(model_dim, rank, false)?,
            shift_up: vb.pp("shift_up").qlinear(rank, model_dim, true)?,
            scale_up: vb.pp("scale_up").qlinear(rank, model_dim, true)?,
            gate_up: vb.pp("gate_up").qlinear(rank, model_dim, true)?,
            eps,
        })
    }

    fn branch(&self, down: &QLinear, up: &QLinear, c: &Tensor) -> Result<Tensor> {
        up.forward(&down.forward(&candle_nn::ops::silu(c)?)?)? + c
    }

    /// `x`: `(B,S,D)`, `cond_embed`: `(B,1,3D)` → modulated `x` `(B,S,D)` and `gate` `(B,1,D)`.
    fn forward(&self, x: &Tensor, cond_embed: &Tensor) -> Result<(Tensor, Tensor)> {
        let parts = cond_embed.chunk(3, D::Minus1)?;
        let shift = self.branch(&self.shift_down, &self.shift_up, &parts[0])?;
        let scale = self.branch(&self.scale_down, &self.scale_up, &parts[1])?;
        let gate = self.branch(&self.gate_down, &self.gate_up, &parts[2])?;

        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let xn = xf.broadcast_div(&(var + self.eps)?.sqrt()?)?.to_dtype(dt)?;
        let x_mod = xn
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        Ok((x_mod, gate.tanh()?))
    }
}

/// JointAttention: latent self-attention + cross-attention to text and speaker contexts.
struct JointAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    gate: QLinear,
    wk_text: QLinear,
    wv_text: QLinear,
    wk_speaker: QLinear,
    wv_speaker: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    head_dim: usize,
}

impl JointAttention {
    fn load(vb: Vb, cfg: &DitConfig) -> Result<Self> {
        let dim = cfg.model_dim;
        let (h, hd) = (cfg.num_heads, cfg.head_dim());
        Ok(Self {
            wq: vb.pp("wq").qlinear(dim, dim, false)?,
            wk: vb.pp("wk").qlinear(dim, dim, false)?,
            wv: vb.pp("wv").qlinear(dim, dim, false)?,
            wo: vb.pp("wo").qlinear(dim, dim, false)?,
            gate: vb.pp("gate").qlinear(dim, dim, false)?,
            wk_text: vb.pp("wk_text").qlinear(cfg.text_dim, dim, false)?,
            wv_text: vb.pp("wv_text").qlinear(cfg.text_dim, dim, false)?,
            wk_speaker: vb.pp("wk_speaker").qlinear(cfg.speaker_dim, dim, false)?,
            wv_speaker: vb.pp("wv_speaker").qlinear(cfg.speaker_dim, dim, false)?,
            q_norm: RmsNorm::load(vb.pp("q_norm"), (h, hd), cfg.norm_eps)?,
            k_norm: RmsNorm::load(vb.pp("k_norm"), (h, hd), cfg.norm_eps)?,
            heads: h,
            head_dim: hd,
        })
    }

    /// Project a context `(B,S,ctx)` to `(k,v)` in `(B,H,S,Dh)` (k RMS-normed, no RoPE).
    fn project_kv(&self, state: &Tensor, wk: &QLinear, wv: &QLinear) -> Result<Kv> {
        let (b, s, _) = state.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let k = self.k_norm.forward(&wk.forward(state)?.reshape((b, s, h, hd))?)?;
        let v = wv.forward(state)?.reshape((b, s, h, hd))?;
        Ok((k.transpose(1, 2)?.contiguous()?, v.transpose(1, 2)?.contiguous()?))
    }

    fn kv_text(&self, text_state: &Tensor) -> Result<Kv> {
        self.project_kv(text_state, &self.wk_text, &self.wv_text)
    }

    fn kv_speaker(&self, speaker_state: &Tensor) -> Result<Kv> {
        self.project_kv(speaker_state, &self.wk_speaker, &self.wv_speaker)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        text_mask: &Tensor,
        ctx_mask: &Tensor,
        rope: &RotaryEmbedding,
        kv_text: &Kv,
        kv_ctx: &Kv,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (h, hd) = (self.heads, self.head_dim);
        let q = self.q_norm.forward(&self.wq.forward(x)?.reshape((b, s, h, hd))?)?;
        let k_self = self.k_norm.forward(&self.wk.forward(x)?.reshape((b, s, h, hd))?)?;
        let v_self = self.wv.forward(x)?.reshape((b, s, h, hd))?;
        let gate = self.gate.forward(x)?;

        // (B,S,H,Dh) → (B,H,S,Dh); RoPE on the first H/2 heads of q and self-k only.
        let q = rope.apply_half_heads(&q.transpose(1, 2)?.contiguous()?, start_pos)?;
        let k_self = rope.apply_half_heads(&k_self.transpose(1, 2)?.contiguous()?, start_pos)?;
        let v_self = v_self.transpose(1, 2)?.contiguous()?;

        let (k_text, v_text) = kv_text;
        let (k_ctx, v_ctx) = kv_ctx;
        let k = Tensor::cat(&[&k_self, k_text, k_ctx], 2)?; // (B,H,total,Dh)
        let v = Tensor::cat(&[&v_self, v_text, v_ctx], 2)?;

        let self_mask = Tensor::ones((b, s), DType::F32, x.device())?;
        let full = Tensor::cat(&[&self_mask, text_mask, ctx_mask], 1)?; // (B, total)
        let add_mask = additive_key_mask(&full)?;

        let scale = 1.0 / (hd as f64).sqrt();
        let out = sdpa(&q, &k, &v, scale, Some(&add_mask))?; // (B,H,S,Dh)
        let out = out.transpose(1, 2)?.reshape((b, s, h * hd))?;
        self.wo.forward(&(out * candle_nn::ops::sigmoid(&gate)?)?)
    }
}

/// One DiT block: `x += attn_gate · attn(adaln1(x))`, then `x += mlp_gate · mlp(adaln2(x))`.
struct DiffusionBlock {
    attention: JointAttention,
    mlp: SwiGlu,
    attention_adaln: LowRankAdaLN,
    mlp_adaln: LowRankAdaLN,
}

impl DiffusionBlock {
    fn load(vb: Vb, cfg: &DitConfig, mlp_hidden: usize) -> Result<Self> {
        Ok(Self {
            attention: JointAttention::load(vb.pp("attention"), cfg)?,
            mlp: SwiGlu::load(vb.pp("mlp"), cfg.model_dim, mlp_hidden)?,
            attention_adaln: LowRankAdaLN::load(vb.pp("attention_adaln"), cfg.model_dim, cfg.adaln_rank, cfg.norm_eps)?,
            mlp_adaln: LowRankAdaLN::load(vb.pp("mlp_adaln"), cfg.model_dim, cfg.adaln_rank, cfg.norm_eps)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cond_embed: &Tensor,
        text_mask: &Tensor,
        ctx_mask: &Tensor,
        rope: &RotaryEmbedding,
        kv_text: &Kv,
        kv_ctx: &Kv,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (x_norm, attn_gate) = self.attention_adaln.forward(x, cond_embed)?;
        let attn_out = self
            .attention
            .forward(&x_norm, text_mask, ctx_mask, rope, kv_text, kv_ctx, start_pos)?;
        let x = (x + attn_gate.broadcast_mul(&attn_out)?)?;
        let (x_norm, mlp_gate) = self.mlp_adaln.forward(&x, cond_embed)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        &x + mlp_gate.broadcast_mul(&mlp_out)?
    }
}

/// Per-layer KV caches for text and context, precomputed once for a set of conditions.
pub struct KvCaches {
    text: Vec<Kv>,
    context: Vec<Kv>,
}

impl KvCaches {
    /// Tile every cached K/V `n` times along the batch axis (for an `n`-way CFG batch, where each
    /// sub-batch reuses the same conditional KV and differs only in its attention mask).
    pub fn replicate_batch(&self, n: usize) -> Result<KvCaches> {
        let rep = |kvs: &[Kv]| -> Result<Vec<Kv>> {
            kvs.iter()
                .map(|(k, v)| {
                    let ks: Vec<Tensor> = (0..n).map(|_| k.clone()).collect();
                    let vs: Vec<Tensor> = (0..n).map(|_| v.clone()).collect();
                    Ok((Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?))
                })
                .collect()
        };
        Ok(KvCaches {
            text: rep(&self.text)?,
            context: rep(&self.context)?,
        })
    }
}

/// The Irodori DiT: encoders + timestep conditioning + N DiffusionBlocks → velocity prediction.
pub struct IrodoriDiT {
    encoders: Encoders,
    cond_module: CondModule,
    in_proj: QLinear,
    blocks: Vec<DiffusionBlock>,
    out_norm: RmsNorm,
    out_proj: QLinear,
    /// v3 duration predictor; `None` for v2 (no `duration_predictor.*` in the checkpoint).
    duration_predictor: Option<DurationPredictor>,
    cfg: DitConfig,
}

impl IrodoriDiT {
    /// Load from a DiT [`Vb`] root (f32 safetensors or q8 GGUF). `rope_max_seq` bounds the RoPE table.
    pub fn load(vb: Vb, cfg: DitConfig, rope_max_seq: usize) -> Result<Self> {
        let mlp_hidden = (cfg.model_dim as f64 * cfg.mlp_ratio) as usize;
        let blocks = (0..cfg.num_layers)
            .map(|i| DiffusionBlock::load(vb.pp("blocks").pp(i), &cfg, mlp_hidden))
            .collect::<Result<Vec<_>>>()?;
        // v3: the integrated duration predictor (absent in v2 → None).
        let duration_predictor = if cfg.use_duration_predictor {
            Some(DurationPredictor::load(vb.pp("duration_predictor"), &cfg)?)
        } else {
            None
        };
        Ok(Self {
            encoders: Encoders::load(vb.clone(), &cfg, rope_max_seq)?,
            cond_module: CondModule::load(vb.pp("cond_module"), cfg.timestep_embed_dim, cfg.model_dim)?,
            in_proj: vb.pp("in_proj").qlinear(cfg.patched_latent_dim(), cfg.model_dim, true)?,
            blocks,
            out_norm: RmsNorm::load(vb.pp("out_norm"), cfg.model_dim, cfg.norm_eps)?,
            out_proj: vb.pp("out_proj").qlinear(cfg.model_dim, cfg.patched_latent_dim(), true)?,
            duration_predictor,
            cfg,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }

    pub fn encoders(&self) -> &Encoders {
        &self.encoders
    }

    /// `text_state = text_norm(text_encoder(...))`, `speaker_state = speaker_norm(speaker_encoder(...))`.
    pub fn encode_conditions(
        &self,
        text_input_ids: &Tensor,
        text_mask: &Tensor,
        ref_latent: &Tensor,
        ref_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let text_state = self.encoders.encode_text(text_input_ids, text_mask)?;
        let speaker_state = self.encoders.encode_speaker(ref_latent, ref_mask)?;
        Ok((text_state, speaker_state))
    }

    /// Whether this DiT carries the v3 duration predictor.
    pub fn has_duration_predictor(&self) -> bool {
        self.duration_predictor.is_some()
    }

    /// Predict the output length in latent frames from text + speaker (v3 duration predictor).
    /// Returns `None` for a v2 model (no predictor). Re-encodes the text/speaker conditions —
    /// cheap next to sampling. `has_speaker` selects the speaker vector (`speaker_state[:,0]`)
    /// vs. the learned `null_speaker`.
    pub fn predict_duration_frames(
        &self,
        text_input_ids: &Tensor,
        text_mask: &Tensor,
        ref_latent: &Tensor,
        ref_mask: &Tensor,
        has_speaker: bool,
    ) -> Result<Option<f64>> {
        let Some(dp) = self.duration_predictor.as_ref() else {
            return Ok(None);
        };
        let (text_state, speaker_state) =
            self.encode_conditions(text_input_ids, text_mask, ref_latent, ref_mask)?;
        let frames = dp.forward(&text_state, text_mask, &speaker_state, has_speaker)?;
        Ok(Some(frames.to_vec1::<f32>()?[0] as f64))
    }

    /// Project the (constant) text/context KV for every block once, for reuse across steps.
    pub fn build_kv_cache(&self, text_state: &Tensor, context_state: &Tensor) -> Result<KvCaches> {
        let text = self
            .blocks
            .iter()
            .map(|b| b.attention.kv_text(text_state))
            .collect::<Result<Vec<_>>>()?;
        let context = self
            .blocks
            .iter()
            .map(|b| b.attention.kv_speaker(context_state))
            .collect::<Result<Vec<_>>>()?;
        Ok(KvCaches { text, context })
    }

    /// Velocity prediction. `x_t`: `(B,S,patched_latent_dim)`, `t`: `(B,)`. `kv` are the
    /// precomputed per-block caches (from [`build_kv_cache`](Self::build_kv_cache)).
    pub fn forward_with_conditions(
        &self,
        x_t: &Tensor,
        t: &Tensor,
        text_mask: &Tensor,
        context_mask: &Tensor,
        kv: &KvCaches,
        start_pos: usize,
    ) -> Result<Tensor> {
        let t_embed = timestep_embedding(t, self.cfg.timestep_embed_dim)?.to_dtype(x_t.dtype())?;
        let cond_embed = self.cond_module.forward(&t_embed)?; // (B,1,3*model_dim)
        let mut x = self.in_proj.forward(x_t)?; // (B,S,model_dim)
        let rope = self.encoders.rope();
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                &x,
                &cond_embed,
                text_mask,
                context_mask,
                rope,
                &kv.text[i],
                &kv.context[i],
                start_pos,
            )?;
        }
        let x = self.out_norm.forward(&x)?;
        self.out_proj.forward(&x)?.to_dtype(DType::F32)
    }
}
