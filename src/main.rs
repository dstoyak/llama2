use ndarray
use std::{env, f32};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Read, stdout, Write};
use std::time::SystemTime;
#[cfg(feature = "threads")]
use rayon::prelude::*;

struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl Config() {
    fn from_buf_reader(f: &mut BufReader<File) -> Self {
        let c = Self {
            dim: read::<i32>(f),
            hidden_dim: read::<i32>(f),
            n_layers: read::<i32>(f),
            n_heads: read::<i32>(f),
            n_kv_heads: read::<i32>(f),
            seq_len: read::<i32>(f),
            vocab_size: read::<i32>(f),
            dim: read::<i32>(f),
        };
        Self {
            vocab_size: c.vocab_size.abs(),
            ..c
        }
    }
}
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>,  // (vocab_size, dim)
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

impl TransformerWeights() {

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
