//! Quantization-aware weight loading shared by maneko's TTS engines (pocket + irodori).
//!
//! Two weight sources sit behind one [`Vb`] enum so the model-construction code is written once:
//!
//! * [`Vb::Full`] wraps a [`candle_nn::VarBuilder`] (mmap'd f32 safetensors) — the original path.
//! * [`Vb::Quant`] reads a GGUF where Linear weights are stored as `Q8_0` and everything else
//!   (conv kernels, norms, embeddings, biases) as `F32`.
//!
//! Linear layers go through [`QLinear`], a thin wrapper over [`QMatMul`]. In the full-precision
//! path `QMatMul::Tensor(w)` computes `x @ wᵀ` exactly like [`candle_nn::Linear`], so the f32 path
//! is numerically unchanged; in the quantized path the `Q8_0` GEMV kernel runs instead. The
//! quantized-vs-not decision is made entirely at load time by which dtype the GGUF holds.

use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};

use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{DType, Device, Module, Result, Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

/// A Linear layer that may be full-precision or `Q8_0`-quantized.
///
/// The forward pass is `x @ wᵀ (+ bias)`. For the quantized variant the input is made contiguous
/// first (candle's quantized matmul requires it); `Tensor::contiguous` is a cheap no-op when the
/// input already is.
#[derive(Clone, Debug)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    pub fn from_parts(inner: QMatMul, bias: Option<Tensor>) -> Self {
        Self { inner, bias }
    }

    /// True if the weight is actually quantized (vs a dequantized/full-precision tensor).
    pub fn is_quantized(&self) -> bool {
        matches!(self.inner, QMatMul::QTensor(_))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = match &self.inner {
            QMatMul::QTensor(_) => self.inner.forward(&x.contiguous()?)?,
            _ => self.inner.forward(x)?,
        };
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// GGUF-backed weight store shared (via `Arc`) by every [`QVarBuilder`] derived from it.
struct GgufStore {
    content: gguf_file::Content,
    reader: Mutex<BufReader<File>>,
    device: Device,
}

/// A `VarBuilder`-shaped view into a GGUF file, tracking a dotted path prefix.
#[derive(Clone)]
pub struct QVarBuilder {
    store: Arc<GgufStore>,
    path: Vec<String>,
}

impl QVarBuilder {
    fn full_name(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.path.join("."), name)
        }
    }

    fn read_qtensor(&self, name: &str) -> Result<QTensor> {
        let full = self.full_name(name);
        let mut reader = self
            .reader_lock()
            .map_err(|e| candle_core::Error::Msg(format!("gguf reader poisoned: {e}")))?;
        self.store.content.tensor(&mut *reader, &full, &self.store.device)
    }

    fn reader_lock(&self) -> std::result::Result<std::sync::MutexGuard<'_, BufReader<File>>, String> {
        self.store.reader.lock().map_err(|e| e.to_string())
    }

    fn has(&self, name: &str) -> bool {
        self.store
            .content
            .tensor_infos
            .contains_key(&self.full_name(name))
    }

    /// Fetch a tensor as full precision (dequantizing if it happens to be stored quantized).
    fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> Result<Tensor> {
        let shape: Shape = shape.into();
        let t = self.read_qtensor(name)?.dequantize(&self.store.device)?;
        if t.shape() != &shape {
            candle_core::bail!(
                "shape mismatch for `{}`: gguf has {:?}, expected {:?}",
                self.full_name(name),
                t.shape(),
                shape
            );
        }
        Ok(t)
    }
}

/// Unified weight builder: full-precision safetensors or quantized GGUF.
///
/// Exposes the small VarBuilder subset the model uses (`pp` / `get` / `device` / `dtype`) plus
/// [`Vb::qlinear`], so swapping `VarBuilder` → `Vb` in a constructor signature is the only change
/// needed at most call sites.
#[derive(Clone)]
pub enum Vb<'a> {
    Full(VarBuilder<'a>),
    Quant(QVarBuilder),
}

impl Vb<'static> {
    /// Open a GGUF file as the root of a quantized weight tree.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("open gguf {}: {e}", path.display())))?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)?;
        let store = Arc::new(GgufStore {
            content,
            reader: Mutex::new(reader),
            device: device.clone(),
        });
        Ok(Vb::Quant(QVarBuilder {
            store,
            path: Vec::new(),
        }))
    }
}

impl<'a> Vb<'a> {
    /// Push a path component (mirrors [`VarBuilder::pp`]).
    pub fn pp<S: ToString>(&self, s: S) -> Vb<'a> {
        match self {
            Vb::Full(vb) => Vb::Full(vb.pp(s)),
            Vb::Quant(q) => {
                let mut path = q.path.clone();
                path.push(s.to_string());
                Vb::Quant(QVarBuilder {
                    store: q.store.clone(),
                    path,
                })
            }
        }
    }

    /// Fetch a non-linear tensor (conv kernel, norm, embedding, bias) as full precision.
    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str) -> Result<Tensor> {
        match self {
            Vb::Full(vb) => vb.get(shape, name),
            Vb::Quant(q) => q.get(shape, name),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Vb::Full(vb) => vb.device(),
            Vb::Quant(q) => &q.store.device,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Vb::Full(vb) => vb.dtype(),
            Vb::Quant(_) => DType::F32,
        }
    }

    /// Build a Linear layer rooted at the current path (reads `weight` and optionally `bias`).
    ///
    /// * Full path: loads `[out, in]` f32 weight → `QMatMul::Tensor` (identical math to
    ///   `candle_nn::linear`).
    /// * Quant path: loads the GGUF tensor → `QMatMul::from_qtensor` (stays `Q8_0`, or dequantizes
    ///   if the converter chose to keep this one full precision).
    pub fn qlinear(&self, in_dim: usize, out_dim: usize, bias: bool) -> Result<QLinear> {
        match self {
            Vb::Full(vb) => {
                let w = vb.get((out_dim, in_dim), "weight")?;
                let bias = if bias {
                    Some(vb.get(out_dim, "bias")?)
                } else {
                    None
                };
                Ok(QLinear::from_parts(QMatMul::Tensor(w), bias))
            }
            Vb::Quant(q) => {
                let qt = q.read_qtensor("weight")?;
                let dims = qt.shape().dims();
                if dims != [out_dim, in_dim] {
                    candle_core::bail!(
                        "linear weight `{}` has shape {:?}, expected [{}, {}]",
                        q.full_name("weight"),
                        dims,
                        out_dim,
                        in_dim
                    );
                }
                let inner = QMatMul::from_qtensor(qt)?;
                let bias = if bias {
                    Some(q.get(out_dim, "bias")?)
                } else {
                    None
                };
                Ok(QLinear::from_parts(inner, bias))
            }
        }
    }

    /// Build a Conv1d rooted at the current path. Conv kernels are always full precision (loaded
    /// via [`Vb::get`], which dequantizes if needed), matching `candle_nn::conv1d`.
    pub fn conv1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
        cfg: Conv1dConfig,
    ) -> Result<Conv1d> {
        let ws = self.get((out_channels, in_channels / cfg.groups, kernel_size), "weight")?;
        let bs = if bias {
            Some(self.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Conv1d::new(ws, bs, cfg))
    }

    /// Build a ConvTranspose1d rooted at the current path (matches `candle_nn::conv_transpose1d`;
    /// note the transposed weight layout `[in, out/groups, k]`).
    pub fn conv_transpose1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
        cfg: ConvTranspose1dConfig,
    ) -> Result<ConvTranspose1d> {
        let ws = self.get((in_channels, out_channels / cfg.groups, kernel_size), "weight")?;
        let bs = if bias {
            Some(self.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(ConvTranspose1d::new(ws, bs, cfg))
    }

    /// Whether a tensor exists at `<prefix>.<name>` (used by optional/versioned weights).
    pub fn contains_tensor(&self, name: &str) -> bool {
        match self {
            Vb::Full(vb) => vb.contains_tensor(name),
            Vb::Quant(q) => q.has(name),
        }
    }
}
