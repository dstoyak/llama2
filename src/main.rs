use ndarray::prelude::*;
use ndarray::RemoveAxis;
// use std::{env, f32};
// use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read};
// use std::time::SystemTime;
#[cfg(feature = "threads")]
use rayon::prelude::*;

// Poor man's num_traits
trait FromBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}
impl FromBytes for i32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        i32::from_le_bytes(bytes)
    }
}
impl FromBytes for u32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}
impl FromBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

fn read_vec<T: FromBytes>(rdr: &mut BufReader<File>, size: i32) -> Tensor<D> {
    (0..size).map(|_| read::<T>(rdr)).collect()
}

fn read<T: FromBytes>(rdr: &mut BufReader<File>) -> T {
    let mut buffer: [u8; 4] = [0u8; 4];
    rdr.read_exact(&mut buffer).expect("Error reading file");
    T::from_bytes(buffer)
}

#[derive(Debug, Clone)]
pub enum Tensor<D: Dimension> {
    // add quant support later
    // Qi8(QintArray<i8, D>),
    F32(Array<f32, D>),
}

impl<D> Tensor<D>
where
    D: Dimension,
{
    pub fn len(&self) -> usize {
        match self {
            // Tensor::Qi8(a) => a.len(),
            Tensor::F32(a) => a.len(),
        }
    }

    pub fn view(&self) -> TensorView<'_, D> {
        match self {
            // Tensor::Qi8(a) => TensorView::Qi8(a.view()),
            Tensor::F32(a) => TensorView::F32(a.view()),
        }
    }
    // implement quantization later
    // pub fn quantize(&self) -> Tensor<D> {
    //     match self {
    //         Tensor::Qi8(_) => panic!("Already quantized!"),
    //         Tensor::F32(a) => Tensor::Qi8(QintArray::quantize(DEFAULT_STRIDE, a.view())),
    //     }
    // }
}

pub enum TensorView<'a, D: Dimension> {
    // Qi8(QintArrayView<'a, i8, D>),
    F32(ArrayView<'a, f32, D>),
}

pub type TensorView1<'a> = TensorView<'a, Ix1>;
pub type TensorView2<'a> = TensorView<'a, Ix2>;

impl<'a, D> TensorView<'a, D>
where
    D: Dimension + RemoveAxis,
{
    pub fn len(&self) -> usize {
        match self {
            TensorView::Qi8(a) => a.len(),
            TensorView::F32(a) => a.len(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        match self {
            TensorView::Qi8(_) => true,
            TensorView::F32(_) => false,
        }
    }

    pub fn index_axis(&'a self, axis: Axis, index: usize) -> TensorView<'a, D::Smaller> {
        match self {
            TensorView::Qi8(a) => TensorView::Qi8(a.index_axis(axis, index)),
            TensorView::F32(a) => TensorView::F32(a.index_axis(axis, index)),
        }
    }

    pub fn unwrap_f32<'b>(&'b self) -> ArrayView<'b, f32, D> {
        match self {
            TensorView::Qi8(_) => panic!("expected f32 tensor"),
            TensorView::F32(a) => a.view(),
        }
    }
}

impl<'a> TensorView<'a, Ix1> {
    pub fn to_f32(&self) -> Array1<f32> {
        match self {
            // TensorView::Qi8(a) => a.to_f32(),
            TensorView::F32(a) => a.to_owned(),
        }
    }
}

impl<'a> TensorView<'a, Ix2> {
    pub fn shape(&self) -> &[usize] {
        match self {
            // TensorView::Qi8(a) => a.arr.shape(),
            TensorView::F32(a) => a.shape(),
        }
    }
    pub fn dot<'b>(&'b self, other: TensorView<'b, Ix1>) -> Array1<f32> {
        match (self, other) {
            // (TensorView::Qi8(a), TensorView::Qi8(b)) => a.dot(b),
            (TensorView::F32(a), TensorView::F32(b)) => a.dot(&b),
            _ => panic!("mismatched types"),
        }
    }
}

pub struct TensorReader<'a, R: BufRead> {
    conf: &'a Config,
    // quantize: bool,
    buf: &'a mut R,
}

impl<'a, R: BufRead> TensorReader<'a, R> {
    pub fn new(conf: &'a Config, quantize: bool, buf: &'a mut R) -> TensorReader<'a, R> {
        TensorReader { conf, buf }
    }

    fn read<D, S>(&mut self, shape: S) -> Tensor<D>
    where
        D: Dimension,
        S: Into<StrideShape<D>>,
    {
        if self.conf.q_type == QuantizationType::None {
            let f32a = read_array(self.buf, shape);
            if self.quantize {
                let qa = QintArray::quantize(DEFAULT_STRIDE, f32a.view());
                Tensor::Qi8(qa)
            } else {
                Tensor::F32(f32a)
            }
        } else {
            let shape = shape.into();
            Tensor::Qi8(QintArray {
                stride: self.conf.q_stride,
                scaling: read_array(self.buf, scaling_dim(shape.raw_dim(), self.conf.q_stride)),
                arr: read_array(self.buf, shape),
            })
        }
    }

    fn skip<S, D>(&mut self, shape: S)
    where
        D: Dimension,
        S: Into<StrideShape<D>>,
    {
        read_array::<_, f32, _, _>(self.buf, shape);
    }
}


pub type Tensor1 = Tensor<Ix1>;
pub type Tensor2 = Tensor<Ix2>;
pub type Tensor3 = Tensor<Ix3>;

struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl Config {
    fn from_buf_reader(f: &mut BufReader<File>) -> Self {
        let c: Config = Self {
            dim: read::<i32>(f) as usize,
            hidden_dim: read::<i32>(f) as usize,
            n_layers: read::<i32>(f) as usize,
            n_heads: read::<i32>(f) as usize,
            n_kv_heads: read::<i32>(f) as usize,
            vocab_size: read::<i32>(f) as usize,
            seq_len: read::<i32>(f).abs() as usize,
        };
        c
    }
}
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Tensor2, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Tensor2, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Tensor2, // (layer, dim)
    // weights for matmuls
    wq: Tensor3, // (layer, dim, dim)
    wk: Tensor3, // (layer, dim, dim)
    wv: Tensor3, // (layer, dim, dim)
    wo: Tensor3, // (layer, dim, dim)
    // weights for ffn
    w1: Tensor3, // (layer, hidden_dim, dim)
    w2: Tensor3, // (layer, dim, hidden_dim)
    w3: Tensor3, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Tensor2, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Tensor2, // (seq_len, dim/2)
    freq_cis_imag: Tensor2, // (seq_len, dim/2)
    // optional output embedding
    wcls: Option<Tensor2>, // (vocab_size, dim)
}

impl TransformerWeights {
    fn from_buf_reader(f: &mut BufReader<File>, c: &Config) {
        let mut tr = TensorView
        let token_embedding_table  =  read_vec::(f, c.n_layers * c.dim),
        let rms_att_weight = read_vec::(f, c.n_layers * c.dim),
        let rms_ffn_weight = read_vec::(f, c.n_layers * c.dim),
        let wq = read_vec::(f, c.n_layers * c.dim),
        let wk =  read_vec::(f, c.n_layers * c.dim),
        let wv = read_vec::(f, c.n_layers * c.dim),
        let wo = read_vec::(f, c.n_layers * c.dim),
        let w1 =  read_vec::(f, c.n_layers * c.dim),
        let w2 = read_vec::(f, c.n_layers * c.dim),
        let w3 =  read_vec::(f, c.n_layers * c.dim),
        let rms_final_weight =  read_vec::(f, c.n_layers * c.dim),
        let frq_cis_real = read_vec::(f, c.n_layers * c.dim),
        let freq_cis_imag =  read_vec::(f, c.n_layers * c.dim),
        let wcls =  read_vec::(f, c.n_layers * c.dim),
    }
}

struct RunState {
    x: f64,
    xb: f64,
    xb2: f64,
    hb: f64,
    hb2: f64,
    q: f64,
    k: f64,
    v: f64,
    att: f64,
    logits: f64,
    key_cache: f64,
    value_cache: f64,
}

struct Transformer {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
    fd: u64,
    data: f64,
    file_size: usize,
}
