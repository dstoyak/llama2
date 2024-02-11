// use ndarray::parallel::prelude::IntoParallelRefMutIterator;
// use ndarray::prelude::*;
// use ndarray::RemoveAxis;
use std::error::Error;
use std::io::stdout;
use std::io::Write;
use std::time::SystemTime;
use std::{env, f32};
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

#[derive(Debug)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    #[allow(dead_code)]
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
    shared_weight: bool,
}

impl Config {
    fn from_buf_reader(f: &mut BufReader<File>) -> Self {
        let c = Self {
            dim: read::<i32>(f),
            hidden_dim: read::<i32>(f),
            n_layers: read::<i32>(f),
            n_heads: read::<i32>(f),
            //Number of key and value heads.
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
    //Linear transformation for queries.
    wq: Vec<f32>, // (layer, dim, dim)
    //Linear transformation for keys.
    wk: Vec<f32>, // (layer, dim, dim)
    //Linear transformation for values.
    wv: Vec<f32>, // (layer, dim, dim)
    //Linear transformation for output.
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    rope_real: Vec<f32>, // (seq_len, dim/2)
    rope_imag: Vec<f32>, // (seq_len, dim/2)
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
        let rope_real = read_vec::<f32>(f, c.seq_len * head_size / 2);
        let rope_imag = read_vec::<f32>(f, c.seq_len * head_size / 2);
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
            rope_real,
            rope_imag,
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
            // activation at current time stamp (dim,)
            x: vec![0.0; c.dim as usize],
            xb: vec![0.0; c.dim as usize],
            xb2: vec![0.0; c.dim as usize],
            hb: vec![0.0; c.hidden_dim as usize],
            hb2: vec![0.0; c.hidden_dim as usize],
            //query
            q: vec![0.0; c.dim as usize],
            //key
            k: vec![0.0; c.dim as usize],
            //value
            v: vec![0.0; c.dim as usize],
            att: vec![0.0; (c.n_heads * c.seq_len) as usize],
            logits: vec![0.0; c.vocab_size as usize],
            key_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
            value_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
        }
    }
}

#[cfg(not(feature = "threads"))]
fn rmsnorm(o: &mut Vec<f32>, x: &Vec<f32>, weight: &[f32]) {
    let mss: f32 = x.iter().map(|&y| y * y).sum::<f32>() / (x.len() as f32);
    let rsqrt: f32 = 1.0 / (mss.sqrt() + 1e-5f32);
    // for ((oi, xi), wi) in o.iter_mut().zip(&x[..]).zip(weight) {
    //     *oi = *xi * rsqrt * *wi;
    // }
    o.iter_mut()
        .zip(&x[..])
        .zip(weight)
        .for_each(|((oi, xi), wi)| *oi = *xi * rsqrt * *wi);
}

#[cfg(feature = "threads")]
fn rmsnorm(o: &mut Vec<f32>, x: &Vec<f32>, weight: &[f32]) {
    let mss = x.par_iter().map(|&y| y * y).sum::<f32>() / (x.len() as f32);
    let rsqrt = 1.0 / (mss.sqrt() + 1e-5f32);
    o.iter_mut()
        .zip(&x[..])
        .zip(weight)
        .for_each(|((oi, xi), wi)| *oi = *xi * rsqrt * *wi);
}

#[cfg(not(feature = "threads"))]
fn matmul(o: &mut Vec<f32>, x: &Vec<f32>, w: &[f32], n: usize, _d: usize) {
    // for i in 0..d {
    //     let mut val: f32 = 0.0;
    //     for j in 0..n {
    //         val += w[i * n + j] * x[j];
    //     }
    //     o[i] = val;
    // }
    o.iter_mut().enumerate().for_each(|(i, oi)| {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *oi = val;
    });
}

#[cfg(feature = "threads")]
fn matmul(o: &mut Vec<f32>, x: &Vec<f32>, w: &[f32], n: usize, d: usize) {
    o.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *oi = val;
    });
}

#[cfg(not(feature = "threads"))]
fn softmax(x: &mut [f32]) {
    let max: f32 = x.iter().fold(x[0], |acc, &x| acc.max(x));
    x.iter_mut().for_each(|a| *a = (*a - max).exp());
    let sum: f32 = x.iter().sum();
    x.iter_mut().for_each(|a| *a /= sum);
}

#[cfg(feature = "threads")]
fn softmax(x: &mut [f32]) {
    let max: f32 = x.iter().fold(x[0], |acc, &x| acc.max(x));
    x.par_iter_mut().for_each(|a| *a = (*a - max).exp());
    let sum: f32 = x.iter().sum();
    x.par_iter_mut().for_each(|a| *a /= sum);
}

fn transformer(p: &Config, w: &TransformerWeights, s: &mut RunState, token: i32, pos: usize) {
    let token: usize = token as usize;
    // activation at current time stamp (dim,)
    // let x = s.x;
    let dim: usize = p.dim as usize;
    let _kv_dim: i32 = (p.dim * p.n_kv_heads) / p.n_heads;
    let _kv_mul: i32 = p.n_heads / p.n_kv_heads;
    let hidden_dim: usize = p.hidden_dim as usize;
    let n_heads: usize = p.n_heads as usize;
    let head_size: usize = dim / n_heads;
    let seq_len: usize = p.seq_len as usize;

    // copy token embeddings into x//s.x is activations at current time (dim, )
    let content_row: &[f32] = &w.token_embedding_table[token * dim..(token + 1) * dim];
    s.x.copy_from_slice(content_row);

    //positional embeddings
    //first gonna do it based on the method used in rust implementation where frrequencies are given.
    //then need to change it to calculate frequencies from head_dim and head_size.
    let rope_real: &[f32] = &w.rope_real[pos * (head_size / 2)..(pos + 1) * (head_size / 2)];
    let rope_imag: &[f32] = &w.rope_imag[pos * (head_size / 2)..(pos + 1) * (head_size / 2)];

    //loop over each layer
    for layer in 0..(p.n_layers as usize) {
        // pre-attention normilization
        rmsnorm(
            &mut s.xb,
            &s.x,
            &w.rms_att_weight[layer * dim..(layer + 1) * dim],
        );

        // q, k, v projections
        // w.wq, w.wk, w.wv are weight vectors computed during training
        matmul(
            &mut s.q,
            &s.xb,
            &w.wq[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut s.k,
            &s.xb,
            &w.wk[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut s.v,
            &s.xb,
            &w.wv[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );

        //rotary positional embeddings
        for head in 0..n_heads {
            let q = &mut s.q[head * head_size..(head + 1) * head_size];
            let k = &mut s.k[head * head_size..(head + 1) * head_size];

            for i in 0..head_size / 2 {
                //fcr is frequency real
                let fcr = rope_real[i];
                //fci is frequncy imaginary
                let fci = rope_imag[i];

                (q[i * 2], q[i * 2 + 1]) = (
                    q[i * 2] * fcr - q[i * 2 + 1] * fci,
                    q[i * 2] * fci + q[i * 2 + 1] * fcr,
                );
                (k[i * 2], k[i * 2 + 1]) = (
                    k[i * 2] * fcr - k[i * 2 + 1] * fci,
                    k[i * 2] * fci + k[i * 2 + 1] * fcr,
                );
            }
        }

        // cache k and v values ater applying layer offset. Layer offset allows for better training and crosspollination of information between layer. Improves contetual understanding
        // honestly read some more about this
        let loff = layer * seq_len * dim;
        s.key_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.k);
        s.value_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.v);

        //multihead attention
        //add multiquery support in run.c. we can now train and infence multiquery models (where n_kv_heads < n_heads). this also means that we, in principle, support Llama 2 34B and 70B models, which are multiquery
        //add karpathy commit #284
        #[cfg(not(feature = "threads"))]
        for h in 0..n_heads {
            let q = &s.q[h * head_size..(h + 1) * head_size];
            let mut att = &mut s.att[h * seq_len..(h + 1) * seq_len];

            for p in 0..=pos {
                let koff: usize = loff + p * dim + h * head_size;
                let k: &[f32] = &s.key_cache[koff..(koff + head_size)];

                // calculating attention score
                att[p] = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                    / (head_size as f32).sqrt();
            }
            softmax(&mut att);

            //storing weighted sum of keys in buffer
            let xb = &mut s.xb[h * head_size..(h + 1) * head_size];
            xb.fill(0.0);
            for p in 0..=pos {
                let koff = loff + p * dim + h * head_size;
                let value_cache = &s.value_cache[koff..(koff + head_size)];
                let a = att[p];
                xb.iter_mut()
                    .zip(value_cache)
                    .for_each(|(xbi, &vi)| *xbi = a * vi);
            }
        }
        #[cfg(feature = "threads")]
        {
            let mut atts: Vec<&mut [f32]> = s.att.chunks_mut(seq_len).collect();
            let qs: Vec<&mut [f32]> = s.q.chunks_mut(head_size).collect();
            let xbs: Vec<&mut [f32]> = s.xb.chunks_mut(head_size).collect();

            atts.par_iter_mut()
                .zip(xbs)
                .enumerate()
                .for_each(|(h, (att, xb))| {
                    let q: &[f32] = qs[h];
                    for p in 0..=pos {
                        let koff: usize = loff + p * dim + h * head_size;
                        let k: &[f32] = &s.key_cache[koff..(koff + head_size)];
                        att[p] = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                            / (head_size as f32).sqrt();
                    }
                    softmax(&mut att[..(pos + 1)]);
                    xb.fill(0.0);
                    for p in 0..=pos {
                        let koff: usize = loff + p * dim + h * head_size;
                        let v = &s.value_cache[koff..(koff + head_size)];
                        let a = att[p];
                        xb.iter_mut().zip(v).for_each(|(xbi, &vi)| *xbi += a * vi);
                    }
                })
        }

        // output projection
        matmul(
            &mut s.xb2,
            &s.xb,
            &w.wo[layer * dim * dim..(layer + 1) * dim * dim],
            dim,
            dim,
        );

        // add xb2 back to x. ths is the "residual connection"
        s.x.iter_mut().zip(s.xb2.iter()).for_each(|(a, b)| *a += *b);

        // rms norm before feed forward network
        rmsnorm(
            &mut s.xb,
            &s.x,
            &w.rms_ffn_weight[layer * dim..(layer + 1) * dim],
        );
        // feed forward nettwork block
        matmul(
            &mut s.hb,
            &s.xb,
            &w.w1[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim],
            dim,
            hidden_dim,
        );
        matmul(
            &mut s.hb2,
            &s.xb,
            &w.w3[layer * hidden_dim * dim..(layer + 1) * hidden_dim * dim],
            dim,
            hidden_dim,
        );

        //activations
        //silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        #[cfg(not(feature = "threads"))]
        {
            s.hb.iter_mut()
                .for_each(|a| *a = *a * (1.0 / (1.0 + (-*a).exp())));
        }
        #[cfg(feature = "threads")]
        {
            s.hb.par_iter_mut()
                .for_each(|a| *a = *a * (1.0 / (1.0 + (-*a).exp())));
        }
        //multiply with hb2=w3(x) into hb
        // whats the purpose of this?
        s.hb.iter_mut()
            .zip(s.hb2.iter())
            .for_each(|(a, &b)| *a *= b);

        // this is just adding weighting? I think?
        // READ MORE REGARDING PURPOSE!
        matmul(
            &mut s.xb,
            &s.hb,
            &w.w2[layer * dim * hidden_dim..(layer + 1) * dim * hidden_dim],
            hidden_dim,
            dim,
        );

        // add xb back into x
        s.x.iter_mut().zip(s.xb.iter()).for_each(|(a, &b)| *a += b);
    }

    // final normilization
    s.xb.copy_from_slice(&s.x);
    rmsnorm(&mut s.x, &s.xb, &w.rms_final_weight);

    // logits
    let wcls = match &w.wcls {
        Some(wcls) => wcls,
        None => &w.token_embedding_table,
    };
    matmul(&mut s.logits, &s.x, wcls, dim, p.vocab_size as usize);
}

//tbh idk why I need to type the Score here like this
//nvm I get it its just a synonym. makes for eaier readability of code
type Score = f32;

//tokenizer read
fn read_tokenizer(vocab_size: usize) -> (Vec<(String, Score)>, u32) {
    let mut rdr = BufReader::new(File::open("tokenizer.bin").expect("Couldn't load tokenizer.bin"));
    let max_token_length: u32 = read::<u32>(&mut rdr);

    let mut vocab: Vec<(String, Score)> = Vec::with_capacity(vocab_size);
    //an undersore indicatates an unused variable in rust. helps with unused variable errors from rust compiler
    for _ in 0..vocab_size {
        let score: f32 = read::<f32>(&mut rdr);
        let len: usize = read::<i32>(&mut rdr) as usize;
        let mut word: Vec<u8> = vec![0_u8; len];
        rdr.read_exact(&mut word).unwrap();
        vocab.push((String::from_utf8(word).unwrap(), score));
    }
    (vocab, max_token_length)
}

//utilities

// random number generator. Permuted congruential generator

pub struct PCG {
    state: u64,
    inc: u64,
}

impl PCG {
    fn randint(&mut self) -> u32 {
        let old_state: u64 = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(self.inc);
        let xor_shifted = (((old_state >> 18u32) ^ old_state) >> 27u32) as u32;
        let rot = (old_state >> 59u32) as u32;
        (xor_shifted >> rot) | (xor_shifted << (((!rot).wrapping_add(1)) & 31))
    }

    // decent,but not great, way of generating random float
    pub fn rand(&mut self) -> f32 {
        let r_int = self.randint();
        (r_int as f32) * (2.0_f32.powi(-32))
    }

    pub fn new(init_state: u64, init_seq: u64) -> Self {
        let mut rng = PCG {
            state: 0,
            inc: (init_seq << 1u32) | 1u64,
        };
        _ = rng.randint();
        rng.state += init_state;
        _ = rng.randint();
        rng
    }
}

// given a vec of probabilities and a random float, this cycles through the probabilities until the cummilative preceeding probabilities are greater than the rand number. at which point it returns the index of the last probability that made the cdf greater than the rand float. if the whole list of probabilities does not result in a number greater than the rand float then the last index is returned.
fn sample(probabilities: &Vec<f32>, rng: &mut PCG) -> usize {
    let r: f32 = rng.rand();
    let mut cdf: f32 = 0.0;

    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1
}

fn bpe_encode(text: &[u8], vocab: &Vec<(String, Score)>, max_token_length: usize) -> Vec<usize> {
    let mut tokens: Vec<usize> = Vec::new();

    for i in 0..text.len() {
        let char: &str = std::str::from_utf8(&text[i..i + 1]).unwrap();
        let (id, _) = vocab
            .iter()
            .enumerate()
            .find(|x| (*(*x).1).0 == char)
            .expect("illegal character");
        tokens.push(id);
    }
    let mut buffer = String::with_capacity(max_token_length);

    loop {
        // structured as (score, (vocab index, tokens index))
        let mut best = (-1e10_f32, (usize::MAX, usize::MAX));

        for i in 0..tokens.len() - 1 {
            buffer.clear();
            buffer.push_str(&vocab[tokens[i]].0);
            buffer.push_str(&vocab[tokens[i + 1]].0);
            //vid means vocab index
            if let Some((vid, (_, score))) =
                vocab.iter().enumerate().find(|x| (*(*x).1).0 == buffer)
            {
                if *score > best.0 {
                    //i is token index
                    best = (*score, (vid, i));
                }
            }
        }
        if best.1 .0 == usize::MAX {
            break; // this is for safety i think
        }
        //merging
        tokens[best.1 .1] = best.1 .0;
        tokens.remove(best.1 .1 + 1);
    }
    tokens
}

fn main() -> Result<(), Box<dyn Error>> {
    //pasrse arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!(
            "Usage:{} <checkpoint_file> [temperature] [steps] [prompt]",
            &args[0]
        );
        return Ok(());
    }
    //finds checkpoint file
    println!("{:#?}", &args);
    let ckpt_file = &args[1];
    println!("ckpt_file: {:?}", ckpt_file);

    let temperature: f32 = args.get(2).map_or(0.9, |x| x.parse().unwrap());
    let steps: usize = args.get(3).map_or(256, |x| x.parse().unwrap());

    let rng_seed = (42, 54);
    let mut rng = PCG::new(rng_seed.0, rng_seed.1);
    println!("Model File: {ckpt_file}, Temperature: {temperature}, Steps: {steps}");

    let mut rdr = BufReader::new(File::open(&ckpt_file)?);

    let config = Config::from_buf_reader(&mut rdr);

    println!("Model: {:?}", config);

    let weights = TransformerWeights::from_buf_reader(&mut rdr, &config);
    let (vocab, max_token_length) = read_tokenizer(config.vocab_size as usize);

    // proccess user input
    let prompt = match args.get(4) {
        Some(p) => String::from(p.trim()),
        None => String::new(),
    };

    let prompt_tokens = match prompt.len() {
        0 => Vec::new(),
        _ => bpe_encode(prompt.as_bytes(), &vocab, max_token_length as usize),
    };

    // final text generation loop

    let mut state = RunState::new(&config);
    let start = SystemTime::now();
    let mut next;
    let mut token = 1; // token 1 is <s> (bos) in the vocab
    let mut pos: usize = 0;
    println!("<s>");

    while pos < steps {
        transformer(&config, &weights, &mut state, token, pos);

        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else {
            if temperature == 0.0 {
                // greedy decoding? choose argmax?
                next = state
                    .logits
                    .iter()
                    .enumerate()
                    .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                    .map(|(i, _)| i)
                    .unwrap();
            } else {
                // temperature scaling
                if temperature < 1.0 {
                    state.logits.iter_mut().for_each(|z| *z /= temperature);
                }
                // compute probabilities
                softmax(&mut state.logits);
                next = sample(&state.logits, &mut rng);
            }
        }

        print!("{}", vocab[next].0);
        stdout().flush()?;

        token = next as i32;
        pos += 1;
    }
    let elapsed = start.elapsed().unwrap();
    println!();
    println!("--------------------------------");
    println!(
        "elapsed: {}.{:03} s, avg tok/s: {}",
        elapsed.as_secs(),
        elapsed.subsec_millis(),
        (steps - 1) as f32 / elapsed.as_secs_f32()
    );

    Ok(())
}
