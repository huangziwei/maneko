//! AR sampling: greedy or temperature + top-p (nucleus), mirroring the MioTTS server defaults
//! (`temperature 0.8`, `top_p 1.0`, `max_tokens 700`, `repetition_penalty 1.0` — i.e. rep-penalty
//! off, so pure temperature sampling). Seeded for reproducibility; `temperature <= 0` ⇒ greedy
//! argmax (identical to the M3 path, so the greedy goldens stay bit-exact).

use candle_core::{DType, Result, Tensor};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Decoding configuration for [`Mio::generate_with`](crate::Mio::generate_with).
#[derive(Clone, Debug)]
pub struct GenConfig {
    /// Max speech tokens to emit before stopping (upstream default 700; ~28 s at 25 Hz).
    pub max_new: usize,
    /// Softmax temperature. `<= 0.0` ⇒ deterministic greedy argmax (no RNG).
    pub temperature: f32,
    /// Nucleus cutoff in `(0, 1]`; `1.0` keeps the full distribution (just temperature sampling).
    pub top_p: f32,
    /// RNG seed for reproducible sampling; `None` ⇒ seed from entropy.
    pub seed: Option<u64>,
}

impl GenConfig {
    /// Deterministic greedy decode of up to `max_new` tokens (no sampling).
    pub fn greedy(max_new: usize) -> Self {
        Self { max_new, temperature: 0.0, top_p: 1.0, seed: None }
    }
}

impl Default for GenConfig {
    /// The MioTTS server defaults (temperature 0.8, top_p 1.0, 700 tokens).
    fn default() -> Self {
        Self { max_new: 700, temperature: 0.8, top_p: 1.0, seed: None }
    }
}

/// Per-generation sampler: holds the temperature/top-p settings and the seeded RNG.
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(cfg: &GenConfig) -> Self {
        let rng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { temperature: cfg.temperature, top_p: cfg.top_p, rng }
    }

    /// Draw a token id from last-position `logits` `(vocab,)`. Greedy when `temperature <= 0`.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        if self.temperature <= 0.0 {
            return logits.argmax(0)?.to_scalar::<u32>();
        }
        // Temperature-scaled, numerically-stable softmax over the full vocab.
        let mut logits = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let inv_t = 1.0 / self.temperature;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for l in logits.iter_mut() {
            *l = ((*l * inv_t) - max * inv_t).exp();
            sum += *l;
        }
        let probs = logits;
        let idx = if self.top_p < 1.0 {
            self.sample_top_p(&probs, sum)
        } else {
            let target = self.rng.gen_range(0.0f32..1.0) * sum;
            self.sample_cdf(&probs, target)
        };
        Ok(idx as u32)
    }

    /// Inverse-CDF draw over all `probs` (unnormalized). `target` is a pre-scaled uniform in
    /// `[0, sum)` where `sum` is the probs' total.
    fn sample_cdf(&self, probs: &[f32], target: f32) -> usize {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if target < cum {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Nucleus draw: keep the smallest set of highest-prob tokens whose cumulative mass reaches
    /// `top_p`, then inverse-CDF within it (`probs` unnormalized, totalling `sum`).
    fn sample_top_p(&mut self, probs: &[f32], sum: f32) -> usize {
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let threshold = self.top_p * sum;
        let mut kept_sum = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &i) in order.iter().enumerate() {
            kept_sum += probs[i];
            if kept_sum >= threshold {
                cutoff = rank + 1;
                break;
            }
        }
        let kept = &order[..cutoff];
        let target = self.rng.gen_range(0.0f32..1.0) * kept_sum;
        let mut cum = 0.0;
        for &i in kept {
            cum += probs[i];
            if target < cum {
                return i;
            }
        }
        *kept.last().unwrap()
    }
}
