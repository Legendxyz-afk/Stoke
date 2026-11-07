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
    /// The memory the engine needs to run this model at the fewest replies at once, 2: file + KV cache + runtime (`gguf::run_bytes`).
    #[ts(type = "number")]
    pub run_bytes: u64,
    /// The same at the most replies at once, 8.
    #[ts(type = "number")]
    pub run_bytes_max: u64,
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
        let model_dirs = std::env::var("LORREN_MODEL_DIRS").ok();
        let userprofile = std::env::var("USERPROFILE").ok();
        let folders = Self::folders_from(downloads, model_dirs.as_deref(), userprofile.as_deref());
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

impl LibraryConfig {
    /// The watched folders from the environment's values. The downloads folder is always in the list. Only the
    /// default list (no `LORREN_MODEL_DIRS`) also holds the user's Windows Downloads folder, where the website
    /// tells a new user to save a model; a test's list never reaches the real user folders.
    pub fn folders_from(downloads: PathBuf, model_dirs: Option<&str>, userprofile: Option<&str>) -> Vec<PathBuf> {
        let mut folders: Vec<PathBuf> = match model_dirs {
            Some(v) => v.split(';').filter(|s| !s.trim().is_empty()).map(PathBuf::from).collect(),
            None => {
                let mut v = vec![
                    PathBuf::from(r"M:\New Drive\ai_models"),
                    PathBuf::from(r"M:\New Drive\Desktop\Programming\stoke\models"),
                    downloads.clone(),
                ];
                if let Some(home) = userprofile.filter(|h| !h.is_empty()) {
                    let win = PathBuf::from(home).join("Downloads");
                    if !v.contains(&win) {
                        v.push(win);
                    }
                }
                v
            }
        };
        if !folders.contains(&downloads) {
            folders.push(downloads);
        }
        folders
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
    out.sort_by(library_order);
    out.dedup_by(|a, b| a.id == b.id);
    if let Some(i) = default_index(&out, &cfg.default_prefix) {
        out[i].is_default = true;
    }
    out
}

/// The default model: the first runnable one whose file name starts with the prefix, else the first runnable.
fn default_index(lib: &[ModelInfo], default_prefix: &str) -> Option<usize> {
    lib.iter()
        .position(|m| m.runnable && file_name_lower(&m.path).starts_with(default_prefix))
        .or_else(|| lib.iter().position(|m| m.runnable))
}

/// The library's order: runnable first, then by name.
fn library_order(a: &ModelInfo, b: &ModelInfo) -> std::cmp::Ordering {
    b.runnable.cmp(&a.runnable).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
}

/// The entry of one GGUF file, as `scan_folder` makes it (the id, the name, a sidecar Modelfile's system prompt).
fn describe_file(p: &Path, modelfiles: &[Modelfile]) -> ModelInfo {
    let own = file_name_lower(&p.to_string_lossy());
    let system = modelfiles
        .iter()
        .find(|m| m.from.as_deref().map_or(false, |f| file_name_lower(f) == own))
        .and_then(|m| m.system.clone());
    let source = system.as_ref().map(|_| "Modelfile".to_string());
    let name = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let id = format!("file:{}", p.to_string_lossy().to_lowercase());
    describe(p, "folder", id, name, system, source)
}

/// Adds one file to a scanned library without scanning again: described exactly as `scan_folder` would, put in the
/// library's order, and the default when the library has none. An id already present is returned unchanged.
pub fn add_file(lib: &mut Vec<ModelInfo>, path: &Path, default_prefix: &str) -> ModelInfo {
    let modelfiles = path.parent().map(read_modelfiles).unwrap_or_default();
    let m = describe_file(path, &modelfiles);
    if let Some(have) = lib.iter().find(|x| x.id == m.id) {
        return have.clone();
    }
    let at = lib.iter().position(|x| library_order(&m, x).is_lt()).unwrap_or(lib.len());
    lib.insert(at, m);
    if !lib.iter().any(|x| x.is_default) {
        if let Some(i) = default_index(lib, default_prefix) {
            lib[i].is_default = true;
        }
    }
    lib[at].clone()
}

/// The Modelfiles in a folder (top level only), as `scan_folder` reads them.
fn read_modelfiles(dir: &Path) -> Vec<Modelfile> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name == "Modelfile" || name.to_lowercase().ends_with(".modelfile") {
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                out.push(parse_modelfile(&text));
            }
        }
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
        out.push(describe_file(&p, &modelfiles));
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
        run_bytes_max: 0,
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
                m.run_bytes = gguf::run_bytes(m.size_bytes, m.kv_bytes_per_token, gguf::ENGINE_MAX_CTX, gguf::ENGINE_SEQS_MIN);
                m.run_bytes_max = gguf::run_bytes(m.size_bytes, m.kv_bytes_per_token, gguf::ENGINE_MAX_CTX, gguf::ENGINE_SEQS_MAX);
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
        // run_bytes = the file + 60,928 B/token x 4,096 x 2 (499,122,176 B of KV cache) + 256 MiB (268,435,456 B), at the fewest replies, 2
        assert_eq!(nexus.run_bytes, nexus.size_bytes + 499_122_176 + 268_435_456);
        // 60,928 B/token x 4,096 x 8
        assert_eq!(nexus.run_bytes_max, nexus.size_bytes + 1_996_488_704 + 268_435_456);
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

    fn folders(downloads: &str, model_dirs: Option<&str>, userprofile: Option<&str>) -> Vec<PathBuf> {
        LibraryConfig::folders_from(PathBuf::from(downloads), model_dirs, userprofile)
    }

    #[test]
    fn the_downloads_folder_is_watched_even_with_model_dirs_set() {
        let with_dirs = folders(r"D:\dl", Some(r"D:\a;D:\b"), Some(r"C:\Users\x"));
        assert_eq!(with_dirs, vec![PathBuf::from(r"D:\a"), PathBuf::from(r"D:\b"), PathBuf::from(r"D:\dl")]);
        // Already in the list: not added twice.
        let held = folders(r"D:\dl", Some(r"D:\dl;D:\b"), None);
        assert_eq!(held, vec![PathBuf::from(r"D:\dl"), PathBuf::from(r"D:\b")]);
    }

    #[test]
    fn the_windows_downloads_folder_is_in_the_default_list_only() {
        let win = PathBuf::from(r"C:\Users\x").join("Downloads");
        let ours = PathBuf::from(r"C:\Users\x\Lorren\models");
        let default = folders(ours.to_str().unwrap(), None, Some(r"C:\Users\x"));
        let at = |p: &PathBuf| default.iter().position(|d| d == p);
        assert_eq!(at(&win), Some(at(&ours).unwrap() + 1));
        assert!(!folders(r"D:\dl", Some(r"D:\a"), Some(r"C:\Users\x")).contains(&win));
        // Not twice when the downloads folder is that folder.
        let same = folders(win.to_str().unwrap(), None, Some(r"C:\Users\x"));
        assert_eq!(same.iter().filter(|d| **d == win).count(), 1);
    }

    /// A one-folder library root; the folder name has a space in it.
    fn one_folder(tag: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("lorren add {}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&root);
        let dir = root.join("my models");
        fs::create_dir_all(&dir).unwrap();
        (root, dir)
    }

    fn cfg_for(dir: &Path, prefix: &str) -> LibraryConfig {
        LibraryConfig { folders: vec![dir.to_path_buf()], ollama_models_dir: None, default_prefix: prefix.into() }
    }

    #[test]
    fn add_file_gives_the_id_a_scan_gives() {
        let (_root, dir) = one_folder("id");
        let file = dir.join("Nexus-Q4.gguf");
        fixture::write(&file, &Spec::llama("Nexus")).unwrap();
        let mut lib = Vec::new();
        let added = add_file(&mut lib, &file, "nexus");
        let scanned = scan(&cfg_for(&dir, "nexus"));
        assert_eq!(scanned.len(), 1);
        let s = &scanned[0];
        assert_eq!(added.id, s.id);
        assert_eq!(added.name, s.name);
        assert_eq!(added.path, s.path);
        assert_eq!(added.size_bytes, s.size_bytes);
        assert_eq!(added.runnable, s.runnable);
        assert_eq!(added.arch, s.arch);
        assert_eq!(added.run_bytes, s.run_bytes);
        assert_eq!(added.is_default, s.is_default);
        assert_eq!(added.source, "folder");
        assert_eq!(lib.len(), 1);
    }

    #[test]
    fn add_file_into_an_empty_library_makes_the_default() {
        let (_root, dir) = one_folder("empty");
        let file = dir.join("Other.gguf");
        fixture::write(&file, &Spec::llama("Other")).unwrap();
        let mut lib = Vec::new();
        let added = add_file(&mut lib, &file, "qwen3-4b-instruct-2507-q4_k_m");
        assert!(added.is_default);
        assert!(lib[0].is_default);
    }

    #[test]
    fn add_file_keeps_an_existing_default() {
        let (_root, dir) = one_folder("keep");
        let a = dir.join("Alpha.gguf");
        let b = dir.join("Zeta.gguf");
        fixture::write(&a, &Spec::llama("Alpha")).unwrap();
        fixture::write(&b, &Spec::llama("Zeta")).unwrap();
        let mut lib = Vec::new();
        add_file(&mut lib, &b, "zzz");
        assert!(lib[0].is_default, "the first file is the default");
        let added = add_file(&mut lib, &a, "alpha");
        assert!(!added.is_default, "the default stays where it was");
        assert_eq!(lib.iter().filter(|m| m.is_default).count(), 1);
        assert!(lib.iter().find(|m| m.name == "Zeta").unwrap().is_default);
        // Runnable first, then by name.
        assert_eq!(lib.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), vec!["Alpha", "Zeta"]);
    }

    #[test]
    fn add_file_twice_adds_once() {
        let (_root, dir) = one_folder("twice");
        let file = dir.join("Nexus.gguf");
        fixture::write(&file, &Spec::llama("Nexus")).unwrap();
        let mut lib = Vec::new();
        let first = add_file(&mut lib, &file, "nexus");
        let second = add_file(&mut lib, &file, "nexus");
        assert_eq!(lib.len(), 1);
        assert_eq!(first.id, second.id);
    }

    #[test]
    fn add_file_reads_a_sidecar_modelfile() {
        let (_root, dir) = one_folder("sidecar");
        let file = dir.join("nexus.gguf");
        fixture::write(&file, &Spec::llama("Nexus")).unwrap();
        fs::write(dir.join("Modelfile"), "FROM ./nexus.gguf\nSYSTEM \"You are Nexus AI.\"\n").unwrap();
        let mut lib = Vec::new();
        let added = add_file(&mut lib, &file, "nexus");
        assert_eq!(added.system_prompt.as_deref(), Some("You are Nexus AI."));
        assert_eq!(added.system_source.as_deref(), Some("Modelfile"));
    }
}
