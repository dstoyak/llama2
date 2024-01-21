struct Config {
    dim: u64,
    hidden_dim: u64,
    n_layers: u64,
    n_heads: u64,
    n_kv_heads: u64,
    vocab_size: u64,
    seq_len: u64,
}
struct TransformerWeights {
    token_embedding_table: f64,
    rms_att_weight: f64,
    rms_ffn_weight: f64,
    wq: f64,
    wk: f64,
    wv: f64,
    wo: f64,
    w1: f64,
    w2: f64,
    w3: f64,
    rms_fnal_weight: f64,
    wcls: f64,
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
