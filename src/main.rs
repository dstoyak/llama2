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

fn read<T: FromBytes>(rdr: &mut BufReader<File>) -> T {
    let mut buffer = [0u8; 4];
    rdr.read_exact(&mut buffer).expect("Error reading file");
    T::from_bytes(buffer)
}

fn read_vec<T: FromBytes>(rdr: &mut BufReader<File>, size: i32) -> Vec<T> {
    (0..size).map(|_| read::<T>(rdr)).collect()
}

impl Config {
    fn from_buf_reader(f: &mut BufReader<File>) -> Self {
        let c = Self {
            dim: read::<i32>(f),
            hidden_dim: read::<i32>(f),
            n_layers: read::<i32>(f),
            n_heads: read::<i32>(f),
            n_kv_heads: read::<i32>(f),
            vocab_size: read::<i32>(f),
            seq_len: read::<i32>(f),
            shared_weight: false,
        };
        Self {
            shared_weight: c.vocab_size > 0,
            vocab_size: c.vocab_size.abs(),
            ..c
        }
    }
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
    // optional output embedding
    wcls: Option<Vec<f32>>, // (vocab_size, dim)
}

impl TransformerWeights {
    fn from_buf_reader(f: &mut BufReader<File>, c: &Config) -> Self {
        let token_embedding_table = read_vec::<f32>(f, c.vocab_size * c.dim);
        let rms_att_weight = read_vec::<f32>(f, c.n_layers * c.dim);
        let wq = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wk = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wv = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wo = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let rms_ffn_weight = read_vec::<f32>(f, c.n_layers * c.dim);
        let w1 = read_vec::<f32>(f, c.n_layers * c.dim * c.hidden_dim);
        let w2 = read_vec::<f32>(f, c.n_layers * c.hidden_dim * c.dim);
        let w3 = read_vec::<f32>(f, c.n_layers * c.dim * c.hidden_dim);
        let rms_final_weight = read_vec::<f32>(f, c.dim);
        let head_size = c.dim / c.n_heads;
        let freq_cis_real = read_vec::<f32>(f, c.seq_len * head_size / 2);
        let freq_cis_imag = read_vec::<f32>(f, c.seq_len * head_size / 2);
        let wcls = match c.shared_weight {
            true => None,
            false => Some(read_vec::<f32>(f, c.vocab_size * c.dim)),
        };

        Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        }
    }
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits (vocab_size,)
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(c: &Config) -> Self {
        Self {
            x: vec![0.0; c.dim as usize],
            xb: vec![0.0; c.dim as usize],
            xb2: vec![0.0; c.dim as usize],
            hb: vec![0.0; c.hidden_dim as usize],
            hb2: vec![0.0; c.hidden_dim as usize],
            q: vec![0.0; c.dim as usize],
            k: vec![0.0; c.dim as usize],
            v: vec![0.0; c.dim as usize],
            att: vec![0.0; (c.n_heads * c.seq_len) as usize],
            logits: vec![0.0; c.vocab_size as usize],
            key_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
            value_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
        }
    }
}
