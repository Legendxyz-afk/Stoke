//! Reads a GGUF file's header -- metadata and tensor table -- without touching the
//! weights, and decides whether the frozen engine (src/ at b1ee6b3) can run it (spec §3.6).
//! The engine would stop with an error on an unsupported file anyway; this says why first.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// The engine clamps every model to this context (STOKE_M2_MAX_CTX, model.c:43).
pub const ENGINE_MAX_CTX: u32 = 4096;
/// Parallel sequences the app runs the engine with (`--seqs`, Global Constraints).
pub const ENGINE_SEQS: u32 = 2;
/// The engine's own working memory on top of the file and the KV cache: 256 MiB.
pub const RUN_OVERHEAD_BYTES: u64 = 256 << 20;

const SUPPORTED_ARCH: [&str; 2] = ["llama", "qwen3"];
/// ggml tensor type ids the engine loads: F32, Q8_0, Q4_K, Q6_K.
const SUPPORTED_TYPES: [u32; 4] = [0, 8, 12, 14];
/// String arrays worth keeping; every other array is skipped (merges alone are ~280k strings).
const KEEP_ARRAYS: [&str; 2] = ["tokenizer.ggml.tokens", "general.tags"];

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    StrArray(Vec<String>),
    /// An array we did not keep: its length only.
    Array(u64),
}

#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensor_types: Vec<u32>,
}

impl Header {
    pub fn str(&self, k: &str) -> Option<&str> {
        match self.meta.get(k) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn uint(&self, k: &str) -> Option<u64> {
        match self.meta.get(k) {
            Some(Value::U64(v)) => Some(*v),
            Some(Value::I64(v)) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn num(&self, k: &str) -> Option<f64> {
        match self.meta.get(k) {
            Some(Value::F64(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn tokens(&self) -> Option<&[String]> {
        match self.meta.get("tokenizer.ggml.tokens") {
            Some(Value::StrArray(v)) => Some(v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    ChatMl,
    Llama3,
    None,
}

impl Template {
    pub fn as_str(self) -> &'static str {
        match self {
            Template::ChatMl => "chatml",
            Template::Llama3 => "llama3",
            Template::None => "none",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Facts {
    pub arch: String,
    pub name: Option<String>,
    pub size_label: Option<String>,
    pub file_type: Option<u32>,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub ctx_file: u32,
    pub template: Template,
}

impl Facts {
    pub fn engine_ctx(&self) -> u32 {
        self.ctx_file.min(ENGINE_MAX_CTX)
    }
    /// KV bytes per token at `--kv q8`: layers x (K+V) x KV heads x head dim x 34/32 (Q8_0 blocks).
    pub fn kv_bytes_per_token_q8(&self) -> u64 {
        let elems = (self.n_layer as u64)
            .saturating_mul(2)
            .saturating_mul(self.n_head_kv as u64)
            .saturating_mul(self.head_dim as u64);
        // u64::MAX = "needs more memory than exists" (fail closed).
        elems.checked_mul(34).map_or(u64::MAX, |v| v / 32)
    }
}

/// The memory a loaded model needs: the file (the engine maps it whole), the KV cache for `seqs`
/// sequences of `ctx` tokens, and the runtime's own overhead. The supervisor refuses a load when free
/// memory is below this plus its spare; the Models tab shows the same number.
pub fn run_bytes(file_bytes: u64, kv_bytes_per_token: u64, ctx: u32, seqs: u32) -> u64 {
    file_bytes
        .saturating_add(kv_bytes_per_token.saturating_mul(ctx as u64).saturating_mul(seqs as u64))
        .saturating_add(RUN_OVERHEAD_BYTES)
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

struct Rd<T: Read> {
    r: T,
}

impl<T: Read> Rd<T> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()?;
        if n > 1 << 24 {
            return Err(bad("string longer than 16 MiB"));
        }
        let mut b = vec![0u8; n as usize];
        self.r.read_exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn skip(&mut self, n: u64) -> io::Result<()> {
        let copied = io::copy(&mut (&mut self.r).take(n), &mut io::sink())?;
        if copied != n {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated"));
        }
        Ok(())
    }
    fn scalar(&mut self, t: u32) -> io::Result<Value> {
        Ok(match t {
            0 => Value::U64(self.bytes::<1>()?[0] as u64),
            1 => Value::I64(self.bytes::<1>()?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.bytes::<2>()?) as u64),
            3 => Value::I64(i16::from_le_bytes(self.bytes::<2>()?) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(bad(&format!("unknown metadata type {t}"))),
        })
    }
}

fn scalar_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4 | 5 | 6 => Some(4),
        10 | 11 | 12 => Some(8),
        _ => None,
    }
}

pub fn read_header(path: &Path) -> io::Result<Header> {
    let mut r = Rd { r: BufReader::with_capacity(1 << 20, File::open(path)?) };
    if &r.bytes::<4>()? != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = r.u32()?;
    if version != 3 {
        return Err(bad(&format!("GGUF v{version} is not supported (the engine reads v3 only)")));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_kv > 1 << 20 || n_tensors > 1 << 24 {
        return Err(bad("implausible header counts"));
    }
    let mut meta = HashMap::new();
    for _ in 0..n_kv {
        let key = r.string()?;
        let t = r.u32()?;
        let value = if t == 9 {
            let et = r.u32()?;
            let n = r.u64()?;
            if et == 8 {
                if KEEP_ARRAYS.contains(&key.as_str()) {
                    let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                    for _ in 0..n {
                        v.push(r.string()?);
                    }
                    Value::StrArray(v)
                } else {
                    for _ in 0..n {
                        let len = r.u64()?;
                        r.skip(len)?;
                    }
                    Value::Array(n)
                }
            } else {
                let size = scalar_size(et).ok_or_else(|| bad("nested or unknown array type"))?;
                r.skip(size.checked_mul(n).ok_or_else(|| bad("array byte count overflows"))?)?;
                Value::Array(n)
            }
        } else {
            r.scalar(t)?
        };
        meta.insert(key, value);
    }
    let mut tensor_types = Vec::with_capacity((n_tensors as usize).min(4096));
    for _ in 0..n_tensors {
        let _name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > 8 {
            return Err(bad("tensor with more than 8 dimensions"));
        }
        for _ in 0..n_dims {
            r.u64()?;
        }
        tensor_types.push(r.u32()?);
        r.u64()?; // data offset
    }
    Ok(Header { version, meta, tensor_types })
}

pub fn facts(h: &Header) -> Result<Facts, String> {
    let arch = h.str("general.architecture").ok_or("no general.architecture")?.to_string();
    let key = |k: &str| format!("{arch}.{k}");
    // Mirrors model.c req_u32: the key must be present; a value past u32 is out of range.
    let need = |k: &str| -> Result<u32, String> {
        let v = h.uint(&key(k)).ok_or_else(|| format!("missing {}", key(k)))?;
        u32::try_from(v).map_err(|_| format!("{} out of range", key(k)))
    };
    let need_f = |k: &str| -> Result<f64, String> {
        h.num(&key(k)).ok_or_else(|| format!("missing {}", key(k)))
    };
    let n_layer = need("block_count")?;
    let ctx_file = need("context_length")?;
    let emb = need("embedding_length")?;
    let n_ff = need("feed_forward_length")?;
    let n_head = need("attention.head_count")?;
    let n_head_kv = need("attention.head_count_kv")?;
    let head_dim = need("attention.key_length")?;
    let value_len = need("attention.value_length")?;
    let freq_base = need_f("rope.freq_base")?;
    let eps = need_f("attention.layer_norm_rms_epsilon")?;
    if n_layer == 0 || emb == 0 || n_ff == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0 {
        return Err(format!(
            "{arch}: block_count, embedding_length, feed_forward_length, head_count, head_count_kv or key_length is 0"
        ));
    }
    if ctx_file == 0 {
        return Err(format!("{} must be positive", key("context_length")));
    }
    if value_len != head_dim {
        return Err(format!(
            "{} ({value_len}) differs from {} ({head_dim})",
            key("attention.value_length"),
            key("attention.key_length")
        ));
    }
    if n_head % n_head_kv != 0 {
        return Err(format!("head_count {n_head} is not divisible by head_count_kv {n_head_kv}"));
    }
    if head_dim % 2 != 0 {
        return Err(format!("{} ({head_dim}) must be even", key("attention.key_length")));
    }
    if !(freq_base.is_finite() && freq_base > 0.0) {
        return Err(format!("{} must be finite and positive", key("rope.freq_base")));
    }
    if !(eps.is_finite() && eps >= 0.0) {
        return Err(format!(
            "{} must be finite and non-negative",
            key("attention.layer_norm_rms_epsilon")
        ));
    }
    let tokens = h.tokens();
    let vocab = tokens
        .map(|t| t.len() as u32)
        .or_else(|| h.uint(&key("vocab_size")).and_then(|v| u32::try_from(v).ok()))
        .unwrap_or(0);
    let has = |s: &str| tokens.map_or(false, |t| t.iter().any(|x| x == s));
    let template = if has("<|im_start|>") && has("<|im_end|>") {
        Template::ChatMl
    } else if has("<|begin_of_text|>")
        && has("<|start_header_id|>")
        && has("<|end_header_id|>")
        && has("<|eot_id|>")
    {
        Template::Llama3
    } else {
        Template::None
    };
    Ok(Facts {
        name: h.str("general.name").map(String::from),
        size_label: h.str("general.size_label").map(String::from),
        file_type: h.uint("general.file_type").and_then(|v| u32::try_from(v).ok()),
        n_layer,
        n_head,
        n_head_kv,
        head_dim,
        vocab,
        ctx_file,
        template,
        arch,
    })
}

pub fn check_runnable(h: &Header, f: &Facts) -> Result<(), String> {
    if !SUPPORTED_ARCH.contains(&f.arch.as_str()) {
        return Err(format!(
            "architecture '{}' is not supported (the engine runs llama and qwen3)",
            f.arch
        ));
    }
    if let Some(t) = h.tensor_types.iter().find(|t| !SUPPORTED_TYPES.contains(t)) {
        return Err(format!(
            "tensor type {} is not supported (F32, Q8_0, Q4_K and Q6_K only)",
            ggml_type_name(*t)
        ));
    }
    if f.head_dim % 32 != 0 {
        return Err(format!(
            "head dimension {} is not a multiple of 32 (the q8 KV cache needs whole 32-element blocks)",
            f.head_dim
        ));
    }
    if f.ctx_file < ENGINE_MAX_CTX {
        return Err(format!(
            "context of {} tokens is below the 4,096 the engine runs at",
            f.ctx_file
        ));
    }
    if f.template == Template::None {
        return Err("no chat template tokens (needs ChatML or Llama-3 header tokens)".into());
    }
    Ok(())
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
        14 => "Q6_K", 15 => "Q8_K", 30 => "BF16", _ => "an unknown type",
    }
}

pub fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 7 => "Q8_0", 10 => "Q2_K", 12 => "Q3_K_M",
        14 => "Q4_K_S", 15 => "Q4_K_M", 16 => "Q5_K_S", 17 => "Q5_K_M", 18 => "Q6_K",
        _ => "other",
    }
}

/// Writes small but valid GGUF v3 headers for tests and the fake engine. No weights.
pub mod fixture {
    use super::*;

    pub const LLAMA3_TOKENS: [&str; 5] =
        ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "hello"];
    pub const CHATML_TOKENS: [&str; 4] = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "hello"];

    pub struct Spec<'a> {
        pub arch: &'a str,
        pub name: &'a str,
        pub n_layer: u32,
        pub n_head: u32,
        pub n_head_kv: u32,
        pub emb: u32,
        pub ctx: u32,
        pub file_type: u32,
        pub tokens: &'a [&'a str],
        pub tensor_types: &'a [u32],
    }

    impl<'a> Spec<'a> {
        pub fn llama(name: &'a str) -> Self {
            Spec { arch: "llama", name, n_layer: 28, n_head: 24, n_head_kv: 8, emb: 3072,
                   ctx: 131_072, file_type: 15, tokens: &LLAMA3_TOKENS, tensor_types: &[0, 12, 14] }
        }
        pub fn qwen3(name: &'a str) -> Self {
            Spec { arch: "qwen3", name, n_layer: 36, n_head: 32, n_head_kv: 8, emb: 2560,
                   ctx: 262_144, file_type: 15, tokens: &CHATML_TOKENS, tensor_types: &[0, 12, 14] }
        }
    }

    enum Kv {
        U32(u32),
        F32(f32),
        Str(String),
        Strs(Vec<String>),
    }

    fn wstr(w: &mut impl Write, s: &str) -> io::Result<()> {
        w.write_all(&(s.len() as u64).to_le_bytes())?;
        w.write_all(s.as_bytes())
    }

    pub fn write(path: &Path, s: &Spec) -> io::Result<()> {
        write_with(path, s, 3, &[], &[])
    }

    /// `write` with a GGUF version, keys to omit, and u32 keys to override (full key names).
    pub fn write_with(
        path: &Path,
        s: &Spec,
        version: u32,
        omit: &[&str],
        set_u32: &[(&str, u32)],
    ) -> io::Result<()> {
        let a = s.arch;
        let kv = vec![
            ("general.architecture".to_string(), Kv::Str(a.to_string())),
            ("general.name".to_string(), Kv::Str(s.name.to_string())),
            ("general.file_type".to_string(), Kv::U32(s.file_type)),
            (format!("{a}.block_count"), Kv::U32(s.n_layer)),
            (format!("{a}.attention.head_count"), Kv::U32(s.n_head)),
            (format!("{a}.attention.head_count_kv"), Kv::U32(s.n_head_kv)),
            (format!("{a}.embedding_length"), Kv::U32(s.emb)),
            (format!("{a}.feed_forward_length"), Kv::U32(if a == "qwen3" { 9728 } else { 8192 })),
            (format!("{a}.attention.key_length"), Kv::U32(128)),
            (format!("{a}.attention.value_length"), Kv::U32(128)),
            (format!("{a}.rope.freq_base"), Kv::F32(500_000.0)),
            (format!("{a}.attention.layer_norm_rms_epsilon"), Kv::F32(1e-5)),
            (format!("{a}.context_length"), Kv::U32(s.ctx)),
            ("tokenizer.ggml.tokens".to_string(), Kv::Strs(s.tokens.iter().map(|t| t.to_string()).collect())),
            ("tokenizer.ggml.merges".to_string(), Kv::Strs(vec!["h e".into(), "l l".into()])),
        ];
        let mut kv: Vec<(String, Kv)> = kv.into_iter().filter(|(k, _)| !omit.contains(&k.as_str())).collect();
        for (k, v) in set_u32 {
            if let Some(e) = kv.iter_mut().find(|(kk, _)| kk == k) {
                e.1 = Kv::U32(*v);
            }
        }
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"GGUF")?;
        w.write_all(&version.to_le_bytes())?;
        w.write_all(&(s.tensor_types.len() as u64).to_le_bytes())?;
        w.write_all(&(kv.len() as u64).to_le_bytes())?;
        for (k, v) in &kv {
            wstr(&mut w, k)?;
            match v {
                Kv::F32(x) => {
                    w.write_all(&6u32.to_le_bytes())?;
                    w.write_all(&x.to_le_bytes())?;
                }
                Kv::U32(x) => {
                    w.write_all(&4u32.to_le_bytes())?;
                    w.write_all(&x.to_le_bytes())?;
                }
                Kv::Str(x) => {
                    w.write_all(&8u32.to_le_bytes())?;
                    wstr(&mut w, x)?;
                }
                Kv::Strs(xs) => {
                    w.write_all(&9u32.to_le_bytes())?;
                    w.write_all(&8u32.to_le_bytes())?;
                    w.write_all(&(xs.len() as u64).to_le_bytes())?;
                    for x in xs {
                        wstr(&mut w, x)?;
                    }
                }
            }
        }
        for (i, t) in s.tensor_types.iter().enumerate() {
            wstr(&mut w, &format!("blk.{i}.weight"))?;
            w.write_all(&1u32.to_le_bytes())?; // one dimension
            w.write_all(&256u64.to_le_bytes())?;
            w.write_all(&t.to_le_bytes())?;
            w.write_all(&0u64.to_le_bytes())?; // data offset (no data)
        }
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{self, Spec};
    use super::*;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("lorren-gguf-{}-{}", std::process::id(), name))
    }

    #[test]
    fn reads_a_llama_fixture() {
        let p = tmp("llama.gguf");
        fixture::write(&p, &Spec::llama("Nexus_Model")).unwrap();
        let h = read_header(&p).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!(f.arch, "llama");
        assert_eq!(f.name.as_deref(), Some("Nexus_Model"));
        assert_eq!((f.n_layer, f.n_head, f.n_head_kv, f.head_dim), (28, 24, 8, 128));
        assert_eq!(f.template, Template::Llama3);
        assert_eq!(f.ctx_file, 131_072);
        assert_eq!(f.engine_ctx(), 4096);
        assert_eq!(f.kv_bytes_per_token_q8(), 60_928); // 59.5 KiB, spec §3.1
        assert!(check_runnable(&h, &f).is_ok());
    }

    #[test]
    fn reads_a_qwen3_fixture() {
        let p = tmp("qwen3.gguf");
        fixture::write(&p, &Spec::qwen3("Qwen3-4B")).unwrap();
        let h = read_header(&p).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!(f.template, Template::ChatMl);
        assert_eq!(f.kv_bytes_per_token_q8(), 78_336); // 76.5 KiB, spec §3.1
        assert!(check_runnable(&h, &f).is_ok());
    }

    #[test]
    fn run_bytes_is_the_file_plus_the_kv_cache_plus_the_runtime() {
        // Qwen3-4B Q4_K_M: 2,497,280,736 B file + 78,336 B/token x 4,096 x 2 (641,728,512 B) + 256 MiB (268,435,456 B)
        assert_eq!(run_bytes(2_497_280_736, 78_336, 4096, 2), 3_407_444_704);
    }

    #[test]
    fn a_vision_architecture_is_refused_with_its_name() {
        let p = tmp("vl.gguf");
        let spec = Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("qwen2vl"), "{err}");
    }

    #[test]
    fn an_unsupported_tensor_type_is_refused() {
        let p = tmp("q5k.gguf");
        let spec = Spec { tensor_types: &[0, 13], ..Spec::llama("x") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("Q5_K"), "{err}");
    }

    #[test]
    fn a_context_below_what_the_engine_runs_at_is_refused() {
        let p = tmp("short.gguf");
        let spec = Spec { ctx: 2048, ..Spec::llama("x") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert_eq!(err, "context of 2048 tokens is below the 4,096 the engine runs at");
    }

    #[test]
    fn missing_template_tokens_are_refused() {
        let p = tmp("notpl.gguf");
        let spec = Spec { tokens: &["hello", "world"], ..Spec::llama("x") };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let err = check_runnable(&h, &facts(&h).unwrap()).unwrap_err();
        assert!(err.contains("chat template"), "{err}");
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let p = tmp("cut.gguf");
        fixture::write(&p, &Spec::llama("x")).unwrap();
        let len = std::fs::metadata(&p).unwrap().len();
        std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(len / 2).unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn a_file_that_is_not_gguf_is_an_error() {
        let p = tmp("junk.gguf");
        std::fs::write(&p, b"GGML this is not a gguf file").unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn an_int_array_whose_byte_count_overflows_is_an_error_not_a_panic() {
        let p = tmp("overflow.gguf");
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // kvs
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b'x');
        b.extend_from_slice(&9u32.to_le_bytes()); // array
        b.extend_from_slice(&4u32.to_le_bytes()); // of u32
        b.extend_from_slice(&(1u64 << 62).to_le_bytes()); // 4 * 2^62 overflows u64
        std::fs::write(&p, b).unwrap();
        assert!(read_header(&p).is_err());
    }

    #[test]
    fn gguf_v2_is_refused_because_the_engine_takes_only_v3() {
        let p = tmp("v2.gguf");
        fixture::write_with(&p, &Spec::llama("x"), 2, &[], &[]).unwrap();
        let err = read_header(&p).unwrap_err().to_string();
        assert!(err.contains("v3"), "{err}");
    }

    #[test]
    fn a_llama3_template_without_begin_of_text_is_not_runnable() {
        let p = tmp("nobot.gguf");
        let spec = Spec {
            tokens: &["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
            ..Spec::llama("x")
        };
        fixture::write(&p, &spec).unwrap();
        let h = read_header(&p).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!(f.template, Template::None);
        assert!(check_runnable(&h, &f).is_err());
    }

    #[test]
    fn each_key_the_engine_requires_is_named_when_missing() {
        for k in [
            "feed_forward_length",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "rope.freq_base",
            "attention.layer_norm_rms_epsilon",
        ] {
            let p = tmp("missing.gguf");
            let full = format!("llama.{k}");
            fixture::write_with(&p, &Spec::llama("x"), 3, &[full.as_str()], &[]).unwrap();
            let h = read_header(&p).unwrap();
            let err = facts(&h).unwrap_err();
            assert_eq!(err, format!("missing {full}"));
        }
    }

    #[test]
    fn a_head_dim_that_is_not_a_multiple_of_32_is_refused() {
        let p = tmp("hd48.gguf");
        let set = [("llama.attention.key_length", 48), ("llama.attention.value_length", 48)];
        fixture::write_with(&p, &Spec::llama("x"), 3, &[], &set).unwrap();
        let h = read_header(&p).unwrap();
        let err = facts(&h).and_then(|f| check_runnable(&h, &f)).unwrap_err();
        assert!(err.contains("32"), "{err}");
    }

    #[test]
    fn value_length_must_equal_key_length_and_heads_must_divide() {
        let p = tmp("vlkl.gguf");
        fixture::write_with(&p, &Spec::llama("x"), 3, &[], &[("llama.attention.value_length", 64)]).unwrap();
        assert!(facts(&read_header(&p).unwrap()).is_err());
        fixture::write_with(&p, &Spec::llama("x"), 3, &[], &[("llama.attention.head_count_kv", 7)]).unwrap();
        assert!(facts(&read_header(&p).unwrap()).is_err());
        fixture::write_with(&p, &Spec::llama("x"), 3, &[], &[("llama.block_count", 0)]).unwrap();
        assert!(facts(&read_header(&p).unwrap()).is_err());
    }

    #[test]
    fn the_memory_arithmetic_saturates_instead_of_panicking() {
        assert_eq!(run_bytes(u64::MAX, u64::MAX, u32::MAX, u32::MAX), u64::MAX);
        // Two headers that `facts` accepts (it checks each field on its own, not their product) and whose product does not
        // fit in 64 bits. Unconditional: were `facts` ever to reject one, the test fails there instead of asserting nothing.
        // The second is exactly 2^64 elements per token: a wrapping multiply reads that as 0, and 0 x 34 does not overflow,
        // so the `checked_mul` that follows could not rescue it (on the first header it would). A plain `*` panics on both.
        let huge = [("llama.block_count", u32::MAX), ("llama.attention.key_length", 1 << 31), ("llama.attention.value_length", 1 << 31)];
        let two_to_the_64 = [
            ("llama.block_count", 1 << 31),
            ("llama.attention.head_count", 1 << 15),
            ("llama.attention.head_count_kv", 1 << 15),
            ("llama.attention.key_length", 1 << 17),
            ("llama.attention.value_length", 1 << 17),
        ];
        for (name, set) in [("huge", &huge[..]), ("two_to_the_64", &two_to_the_64[..])] {
            let p = tmp(&format!("sat-{name}.gguf"));
            fixture::write_with(&p, &Spec::llama("x"), 3, &[], set).unwrap();
            let f = facts(&read_header(&p).unwrap()).unwrap_or_else(|e| panic!("{name}: a header with huge sizes is accepted: {e}"));
            assert_eq!(f.kv_bytes_per_token_q8(), u64::MAX, "{name}");
        }
    }

    /// Kabir's real files. Run with: cargo test real_models -- --ignored
    #[test]
    #[ignore]
    fn real_models() {
        let nexus = std::path::Path::new("M:/New Drive/ai_models/nexus.gguf");
        let h = read_header(nexus).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!((f.arch.as_str(), f.n_layer, f.n_head_kv), ("llama", 28, 8));
        assert_eq!(f.template, Template::Llama3);
        assert_eq!(f.kv_bytes_per_token_q8(), 60_928);
        assert!(check_runnable(&h, &f).is_ok());
        let qwen = std::path::Path::new(
            "M:/New Drive/Desktop/Programming/stoke/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        );
        let h = read_header(qwen).unwrap();
        let f = facts(&h).unwrap();
        assert_eq!((f.arch.as_str(), f.n_layer), ("qwen3", 36));
        assert_eq!(f.template, Template::ChatMl);
        assert!(check_runnable(&h, &f).is_ok());
    }
}
