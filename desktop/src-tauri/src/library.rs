//! The model library (spec §3.6): GGUF files in watched folders, Ollama's manifests, and a
//! Modelfile sidecar next to a GGUF. Every entry says whether the engine can run it and why not.

use crate::gguf;
use serde::Serialize;
use std::path::{Path, PathBuf};
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    /// "folder" or "ollama"
    pub source: String,
    #[ts(type = "number")]
    pub size_bytes: u64,
    pub arch: String,
    pub quant: String,
    pub size_label: String,
    pub n_layer: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub ctx_file: u32,
    pub ctx_engine: u32,
    #[ts(type = "number")]
    pub kv_bytes_per_token: u64,
    /// The memory the engine needs to run this model: file + KV cache + runtime (`gguf::run_bytes`).
    #[ts(type = "number")]
    pub run_bytes: u64,
    /// "chatml", "llama3" or "none"
    pub template: String,
    pub runnable: bool,
    pub reason: Option<String>,
    pub system_prompt: Option<String>,
    /// "Modelfile" or "Ollama"
    pub system_source: Option<String>,
    pub is_default: bool,
}

pub struct LibraryConfig {
    pub folders: Vec<PathBuf>,
    pub ollama_models_dir: Option<PathBuf>,
    /// Lower-case file-name prefix of the default model (spec §2: Qwen3-4B-Instruct-2507).
    pub default_prefix: String,
}

impl LibraryConfig {
    /// Spec §3.6 defaults. `LORREN_MODEL_DIRS` (semicolon-separated) and `LORREN_OLLAMA_DIR`
    /// (`none` disables Ollama) override them for tests and the end-to-end run.
    pub fn from_env(downloads: PathBuf) -> Self {
        let folders = match std::env::var("LORREN_MODEL_DIRS") {
            Ok(v) => v.split(';').filter(|s| !s.trim().is_empty()).map(PathBuf::from).collect(),
            Err(_) => vec![
                PathBuf::from(r"M:\New Drive\ai_models"),
                PathBuf::from(r"M:\New Drive\Desktop\Programming\stoke\models"),
                downloads,
            ],
        };
        let ollama_models_dir = match std::env::var("LORREN_OLLAMA_DIR") {
            Ok(v) if v == "none" => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => std::env::var("OLLAMA_MODELS").ok().map(PathBuf::from).or_else(|| {
                std::env::var("USERPROFILE").ok().map(|h| PathBuf::from(h).join(".ollama").join("models"))
            }),
        };
        LibraryConfig { folders, ollama_models_dir, default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into() }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Modelfile {
    pub from: Option<String>,
    pub system: Option<String>,
    pub params: Vec<(String, String)>,
}

pub fn parse_modelfile(text: &str) -> Modelfile {
    let mut m = Modelfile::default();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some((kw, rest)) = line.trim().split_once(char::is_whitespace) else { continue };
        let rest = rest.trim();
        match kw.to_ascii_uppercase().as_str() {
            "FROM" => m.from = Some(rest.to_string()),
            "SYSTEM" => {
                let value = if let Some(body) = rest.strip_prefix("\"\"\"") {
                    if let Some(end) = body.find("\"\"\"") {
                        body[..end].to_string()
                    } else {
                        let mut acc = body.to_string();
                        for l in lines.by_ref() {
                            acc.push('\n');
                            if let Some(end) = l.find("\"\"\"") {
                                acc.push_str(&l[..end]);
                                break;
                            }
                            acc.push_str(l);
                        }
                        acc
                    }
                } else if rest.len() >= 2 && rest.starts_with('"') && rest.ends_with('"') {
                    rest[1..rest.len() - 1].to_string()
                } else {
                    rest.to_string()
                };
                m.system = Some(value.trim().to_string());
            }
            "PARAMETER" => {
                if let Some((k, v)) = rest.split_once(char::is_whitespace) {
                    m.params.push((k.to_string(), v.trim().to_string()));
                }
            }
            _ => {}
        }
    }
    m
}

fn file_name_lower(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_lowercase()
}

pub fn scan(cfg: &LibraryConfig) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    for dir in &cfg.folders {
        scan_folder(dir, &mut out);
    }
    if let Some(root) = &cfg.ollama_models_dir {
        scan_ollama(root, &mut out);
    }
    out.sort_by(|a, b| {
        b.runnable.cmp(&a.runnable).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    out.dedup_by(|a, b| a.id == b.id);
    let default = out
        .iter()
        .position(|m| m.runnable && file_name_lower(&m.path).starts_with(&cfg.default_prefix))
        .or_else(|| out.iter().position(|m| m.runnable));
    if let Some(i) = default {
        out[i].is_default = true;
    }
    out
}

fn scan_folder(dir: &Path, out: &mut Vec<ModelInfo>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut ggufs = Vec::new();
    let mut modelfiles = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if name.to_lowercase().ends_with(".gguf") && p.is_file() {
            ggufs.push(p);
        } else if name == "Modelfile" || name.to_lowercase().ends_with(".modelfile") {
            if let Ok(text) = std::fs::read_to_string(&p) {
                modelfiles.push(parse_modelfile(&text));
            }
        }
    }
    for p in ggufs {
        let own = file_name_lower(&p.to_string_lossy());
        let system = modelfiles
            .iter()
            .find(|m| m.from.as_deref().map_or(false, |f| file_name_lower(f) == own))
            .and_then(|m| m.system.clone());
        let source = system.as_ref().map(|_| "Modelfile".to_string());
        let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let id = format!("file:{}", p.to_string_lossy().to_lowercase());
        out.push(describe(&p, "folder", id, name, system, source));
    }
}

fn scan_ollama(root: &Path, out: &mut Vec<ModelInfo>) {
    let library = root.join("manifests").join("registry.ollama.ai").join("library");
    let Ok(names) = std::fs::read_dir(&library) else { return };
    for n in names.flatten() {
        let Ok(tags) = std::fs::read_dir(n.path()) else { continue };
        for t in tags.flatten() {
            let Ok(text) = std::fs::read_to_string(t.path()) else { continue };
            let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
            let layers = manifest["layers"].as_array().cloned().unwrap_or_default();
            let blob = |media: &str| {
                layers
                    .iter()
                    .find(|l| l["mediaType"] == media)
                    .and_then(|l| l["digest"].as_str())
                    .map(|d| root.join("blobs").join(d.replace(':', "-")))
            };
            let Some(model) = blob("application/vnd.ollama.image.model") else { continue };
            if !model.is_file() {
                continue;
            }
            let system = blob("application/vnd.ollama.image.system")
                .and_then(|p| std::fs::read_to_string(p).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let name = format!("{}:{}", n.file_name().to_string_lossy(), t.file_name().to_string_lossy());
            let source = system.as_ref().map(|_| "Ollama".to_string());
            out.push(describe(&model, "ollama", format!("ollama:{name}"), name, system, source));
        }
    }
}

fn describe(
    path: &Path,
    source: &str,
    id: String,
    name: String,
    system_prompt: Option<String>,
    system_source: Option<String>,
) -> ModelInfo {
    let mut m = ModelInfo {
        id,
        name,
        path: path.to_string_lossy().to_string(),
        source: source.into(),
        size_bytes: std::fs::metadata(path).map(|x| x.len()).unwrap_or(0),
        arch: String::new(),
        quant: String::new(),
        size_label: String::new(),
        n_layer: 0,
        n_head: 0,
        n_head_kv: 0,
        head_dim: 0,
        vocab: 0,
        ctx_file: 0,
        ctx_engine: 0,
        kv_bytes_per_token: 0,
        run_bytes: 0,
        template: "none".into(),
        runnable: false,
        reason: None,
        system_prompt,
        system_source,
        is_default: false,
    };
    match gguf::read_header(path) {
        Err(e) => m.reason = Some(format!("cannot read the file header: {e}")),
        Ok(h) => match gguf::facts(&h) {
            Err(e) => m.reason = Some(e),
            Ok(f) => {
                m.arch = f.arch.clone();
                m.quant = f.file_type.map(gguf::file_type_name).unwrap_or("other").into();
                m.size_label = f.size_label.clone().unwrap_or_default();
                m.n_layer = f.n_layer;
                m.n_head = f.n_head;
                m.n_head_kv = f.n_head_kv;
                m.head_dim = f.head_dim;
                m.vocab = f.vocab;
                m.ctx_file = f.ctx_file;
                m.ctx_engine = f.engine_ctx();
                m.kv_bytes_per_token = f.kv_bytes_per_token_q8();
                m.run_bytes = gguf::run_bytes(m.size_bytes, m.kv_bytes_per_token, gguf::ENGINE_MAX_CTX, gguf::ENGINE_SEQS);
                m.template = f.template.as_str().into();
                match gguf::check_runnable(&h, &f) {
                    Ok(()) => m.runnable = true,
                    Err(e) => m.reason = Some(e),
                }
            }
        },
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::fixture::{self, Spec};
    use std::fs;

    /// A library laid out like Kabir's: a folder with nexus.gguf + its Modelfile, a folder
    /// with Qwen3, a vision model, and an Ollama store. The root has a space in it on purpose.
    fn layout(tag: &str) -> (PathBuf, LibraryConfig) {
        let root = std::env::temp_dir().join(format!("lorren lib {}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&root);
        let ai = root.join("ai models");
        let stoke = root.join("stoke models");
        let ollama = root.join("ollama");
        fs::create_dir_all(&ai).unwrap();
        fs::create_dir_all(&stoke).unwrap();
        fixture::write(&ai.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
        fs::write(
            ai.join("Modelfile"),
            "FROM C:/Users/Kabir/ai_models/nexus.gguf\nSYSTEM \"You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts.\"\n",
        )
        .unwrap();
        fixture::write(&stoke.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
        fixture::write(&stoke.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), &Spec::llama("Llama-3.2-3B")).unwrap();
        let man = ollama.join("manifests/registry.ollama.ai/library/qwen2.5vl");
        fs::create_dir_all(&man).unwrap();
        fs::create_dir_all(ollama.join("blobs")).unwrap();
        fixture::write(&ollama.join("blobs/sha256-aaa"), &Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") }).unwrap();
        fs::write(ollama.join("blobs/sha256-bbb"), "You are a helpful vision assistant.").unwrap();
        fs::write(
            man.join("3b"),
            r#"{"schemaVersion":2,"layers":[
                {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaa","size":1},
                {"mediaType":"application/vnd.ollama.image.system","digest":"sha256:bbb","size":1}]}"#,
        )
        .unwrap();
        let cfg = LibraryConfig {
            folders: vec![ai, stoke, root.join("missing folder")],
            ollama_models_dir: Some(ollama),
            default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
        };
        (root, cfg)
    }

    fn by_name<'a>(v: &'a [ModelInfo], name: &str) -> &'a ModelInfo {
        v.iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no model {name} in {v:?}"))
    }

    #[test]
    fn nexus_gets_its_system_prompt_from_the_modelfile_matched_by_file_name() {
        let (_root, cfg) = layout("nexus");
        let models = scan(&cfg);
        let nexus = by_name(&models, "nexus");
        assert!(nexus.runnable, "{:?}", nexus.reason);
        assert_eq!(nexus.system_source.as_deref(), Some("Modelfile"));
        assert!(nexus.system_prompt.as_deref().unwrap().starts_with("You are Nexus AI"));
        assert!(nexus.path.contains("lorren lib"), "the path keeps its spaces: {}", nexus.path);
        assert_eq!(nexus.template, "llama3");
        assert_eq!(nexus.ctx_engine, 4096);
        // run_bytes = the file + 60,928 B/token x 4,096 x 2 (499,122,176 B of KV cache) + 256 MiB (268,435,456 B)
        assert_eq!(nexus.run_bytes, nexus.size_bytes + 499_122_176 + 268_435_456);
    }

    #[test]
    fn qwen3_4b_is_the_default_and_only_one_model_is() {
        let (_root, cfg) = layout("default");
        let models = scan(&cfg);
        let defaults: Vec<_> = models.iter().filter(|m| m.is_default).collect();
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].name, "Qwen3-4B-Instruct-2507-Q4_K_M");
    }

    #[test]
    fn the_ollama_vision_model_is_listed_but_cannot_run() {
        let (_root, cfg) = layout("ollama");
        let models = scan(&cfg);
        let vl = by_name(&models, "qwen2.5vl:3b");
        assert_eq!(vl.id, "ollama:qwen2.5vl:3b");
        assert!(!vl.runnable);
        assert!(vl.reason.as_deref().unwrap().contains("qwen2vl"));
        assert_eq!(vl.system_source.as_deref(), Some("Ollama"));
        assert!(models.first().unwrap().runnable, "runnable models come first");
    }

    #[test]
    fn a_missing_folder_is_skipped_quietly() {
        let cfg = LibraryConfig {
            folders: vec![PathBuf::from("Z:/does/not/exist")],
            ollama_models_dir: None,
            default_prefix: "x".into(),
        };
        assert!(scan(&cfg).is_empty());
    }

    #[test]
    fn modelfile_parsing_handles_quotes_triple_quotes_and_parameters() {
        let m = parse_modelfile("from ./a.gguf\nSYSTEM \"\"\"line one\nline two\"\"\"\nPARAMETER temperature 0.4\nPARAMETER stop <|eot_id|>\n");
        assert_eq!(m.from.as_deref(), Some("./a.gguf"));
        assert_eq!(m.system.as_deref(), Some("line one\nline two"));
        assert_eq!(m.params, vec![("temperature".into(), "0.4".into()), ("stop".into(), "<|eot_id|>".into())]);
        let plain = parse_modelfile("FROM x.gguf\nSYSTEM be brief\n");
        assert_eq!(plain.system.as_deref(), Some("be brief"));
    }
}
