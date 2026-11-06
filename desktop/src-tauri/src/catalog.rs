//! The model catalog (spec §3.7, step 2): a signed JSON document the app reads ONLY when Discover opens, verified with a
//! public key of its own (compiled into the build; the app's update key cannot sign a catalog), and kept in the store so
//! Discover works offline. What is shown is always verified: the cache is verified again every time it is read back.
//!
//! The entry rules are the SAME list as `scripts/release/check.mjs` `validateEntry` (one test table each).

use crate::monitor::Monitor;
use crate::store::Store;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use ts_rs::TS;

/// The `plugins` section of the build's config that names the catalog (`tauri.catalog-test.conf.json`, or the overlay the
/// release script writes): `{ url, pubkey, insecure? }`.
pub const CATALOG_PLUGIN: &str = "lorren-catalog";

const MAX_SIZE: u64 = 68_719_476_736; // 64 GiB
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_BODY: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct CatalogSource {
    pub url: String,
    pub public_key: String,
    /// A test build: `http://` addresses are taken (the catalog's and its entries'). Never set in a release.
    pub insecure: bool,
}

/// None without both a non-empty `url` and `pubkey`: the build has no catalog and Discover says so.
pub fn source_from_config(section: Option<&serde_json::Value>) -> Option<CatalogSource> {
    let s = section?;
    let text = |k: &str| s.get(k).and_then(|v| v.as_str()).map(str::trim).filter(|v| !v.is_empty()).map(str::to_string);
    Some(CatalogSource { url: text("url")?, public_key: text("pubkey")?, insecure: s.get("insecure").and_then(|v| v.as_bool()).unwrap_or(false) })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct Licence {
    pub name: String,
    pub url: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct EntrySource {
    pub repo: String,
    pub commit: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    #[ts(type = "number")]
    pub version: u64,
    pub description: String,
    pub architecture: String,
    pub quantization: String,
    /// The name the file is saved under in the downloads folder.
    pub file: String,
    #[ts(type = "number")]
    pub size: u64,
    pub sha256: String,
    pub url: String,
    pub licence: Licence,
    pub source: EntrySource,
    pub system_prompt: Option<String>,
    pub min_app_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct CatalogView {
    /// False when the build names no catalog (no `url` or no `pubkey`): there is nothing to read.
    pub configured: bool,
    pub entries: Vec<CatalogEntry>,
    /// Ids of entries whose `min_app_version` is newer than this app.
    pub needs_app: Vec<String>,
    #[ts(type = "number | null")]
    pub fetched_ms: Option<i64>,
    /// The entries are the last verified catalog, and the newest read of it failed (`error` says why).
    pub stale: bool,
    pub error: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CatalogError {
    #[error("the catalog's public key cannot be read: {0}")]
    Key(String),
    #[error("the catalog's signature cannot be read: {0}")]
    Unreadable(String),
    #[error("the catalog is signed by a key this Lorren does not trust")]
    UntrustedKey,
    #[error("the catalog does not match its signature, so it changed after it was signed")]
    Altered,
    #[error("the catalog is not in a format this Lorren reads: {0}")]
    Format(String),
}

/// A catalog whose signature checked out: the entries that passed the rules, and one sentence for each that did not.
#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub generated: String,
    pub entries: Vec<CatalogEntry>,
    pub dropped: Vec<String>,
}

// ---- The entry rules (the same list as check.mjs `validateEntry`) ----

fn units(s: &str) -> usize {
    s.encode_utf16().count() // JavaScript's `.length`, so the two lists agree on every string
}

fn text_between(v: &str, min: usize, max: usize) -> bool {
    (min..=max).contains(&units(v))
}

fn id_ok(id: &str) -> bool {
    let b = id.as_bytes();
    !b.is_empty()
        && b.len() <= 64
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b.iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-'))
}

fn file_ok(file: &str) -> bool {
    let b = file.as_bytes();
    if !file.is_ascii() || b.len() < 6 || b.len() > 133 || !file[file.len() - 5..].eq_ignore_ascii_case(".gguf") || file.contains("..") {
        return false;
    }
    let stem = &b[..b.len() - 5];
    stem[0].is_ascii_alphanumeric() && stem.iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

fn sha_ok(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn version_ok(v: &str) -> bool {
    semver::Version::parse(v).is_ok_and(|v| v.build.is_empty())
}

/// One entry against the rules. `Err` names every rule it breaks, each starting with its field, joined by "; ".
pub fn validate_entry(e: &CatalogEntry, insecure: bool) -> Result<(), String> {
    let mut out: Vec<String> = Vec::new();
    let mut bad = |field: &str, rule: &str| out.push(format!("{field} {rule}"));
    let url = |v: &str| (v.starts_with("https://") || (insecure && v.starts_with("http://"))) && units(v) > 8;
    if !id_ok(&e.id) {
        bad("id", "is not 1 to 64 lowercase letters, digits, dots, dashes or underscores, starting with a letter or digit");
    }
    if !text_between(&e.name, 1, 80) {
        bad("name", "is not 1 to 80 characters");
    }
    if !text_between(&e.description, 0, 400) {
        bad("description", "is not text of at most 400 characters");
    }
    if e.version < 1 || e.version > MAX_SAFE_INTEGER {
        bad("version", "is not a positive whole number");
    }
    if e.architecture != "qwen3" && e.architecture != "llama" {
        bad("architecture", "is not qwen3 or llama");
    }
    if !text_between(&e.quantization, 1, 16) {
        bad("quantization", "is not 1 to 16 characters");
    }
    if !file_ok(&e.file) {
        bad("file", "is not a plain .gguf file name");
    }
    if e.size < 1 || e.size > MAX_SIZE {
        bad("size", &format!("is not a whole number from 1 to {MAX_SIZE}"));
    }
    if !sha_ok(&e.sha256) {
        bad("sha256", "is not 64 lowercase hex digits");
    }
    if !url(&e.url) {
        bad("url", if insecure { "is not an http:// or https:// address" } else { "is not an https:// address" });
    }
    if !text_between(&e.licence.name, 1, 80) {
        bad("licence.name", "is not 1 to 80 characters");
    }
    if !e.licence.url.starts_with("https://") || units(&e.licence.url) <= 8 {
        bad("licence.url", "is not an https:// address");
    }
    if !version_ok(&e.min_app_version) {
        bad("min_app_version", "is not a version");
    }
    if e.system_prompt.as_deref().is_some_and(|p| !text_between(p, 0, 4000)) {
        bad("system_prompt", "is neither null nor text of at most 4000 characters");
    }
    if out.is_empty() { Ok(()) } else { Err(out.join("; ")) }
}

/// The entries of a catalog's `models` list that pass the rules (in order), and a sentence for each that does not: a
/// broken rule, a field of the wrong kind, a second entry with an id already taken, or with a file name already taken
/// (compared in lower case: Windows treats the two spellings as one file).
pub fn filter_models(models: &[serde_json::Value], insecure: bool) -> (Vec<CatalogEntry>, Vec<String>) {
    let (mut entries, mut dropped): (Vec<CatalogEntry>, Vec<String>) = (Vec::new(), Vec::new());
    for (i, value) in models.iter().enumerate() {
        let label = match value.get("id").and_then(|v| v.as_str()) {
            Some(id) => format!("models[{i}] ({id})"),
            None => format!("models[{i}]"),
        };
        let entry: CatalogEntry = match serde_json::from_value(value.clone()) {
            Ok(e) => e,
            Err(e) => {
                dropped.push(format!("{label}: {e}"));
                continue;
            }
        };
        if let Err(why) = validate_entry(&entry, insecure) {
            dropped.push(format!("{label}: {why}"));
        } else if entries.iter().any(|k| k.id == entry.id) {
            dropped.push(format!("{label}: id \"{}\" appears twice", entry.id));
        } else if entries.iter().any(|k| k.file.eq_ignore_ascii_case(&entry.file)) {
            dropped.push(format!("{label}: file \"{}\" appears twice", entry.file));
        } else {
            entries.push(entry);
        }
    }
    (entries, dropped)
}

// ---- Verification ----

/// The key as the build holds it: the `.pub` file's text, that text base64-encoded (the form the updater's config takes),
/// or the bare key line.
fn decode_key(text: &str) -> Result<minisign_verify::PublicKey, CatalogError> {
    use minisign_verify::PublicKey;
    let text = text.trim();
    let key = |e: minisign_verify::Error| CatalogError::Key(e.to_string());
    if text.starts_with("untrusted comment:") {
        return PublicKey::decode(text).map_err(key);
    }
    if let Some(inner) = base64::engine::general_purpose::STANDARD.decode(text).ok().and_then(|b| String::from_utf8(b).ok()) {
        if inner.starts_with("untrusted comment:") {
            return PublicKey::decode(&inner).map_err(key);
        }
    }
    PublicKey::from_base64(text).map_err(key)
}

/// The catalog `json` and its signature `minisig` against `public_key`; nothing is read from the document before the
/// signature holds. Entries that break the rules are dropped and named in `dropped`; the rest are kept.
pub fn verify(json: &[u8], minisig: &str, public_key: &str, insecure: bool) -> Result<Catalog, CatalogError> {
    use minisign_verify::Error as M;
    let key = decode_key(public_key)?;
    let sig = minisign_verify::Signature::decode(minisig).map_err(|e| CatalogError::Unreadable(e.to_string()))?;
    key.verify(json, &sig, false).map_err(|e| match e {
        M::UnexpectedKeyId => CatalogError::UntrustedKey,
        M::InvalidSignature => CatalogError::Altered,
        other => CatalogError::Unreadable(other.to_string()),
    })?;
    let doc: serde_json::Value = serde_json::from_slice(json).map_err(|e| CatalogError::Format(format!("it does not parse ({e})")))?;
    match doc.get("format").and_then(|f| f.as_i64()) {
        Some(1) => {}
        Some(n) => return Err(CatalogError::Format(format!("format {n}"))),
        None => return Err(CatalogError::Format("it names no format".into())),
    }
    let models = doc.get("models").and_then(|m| m.as_array()).ok_or_else(|| CatalogError::Format("it has no list of models".into()))?;
    let (entries, dropped) = filter_models(models, insecure);
    let generated = doc.get("generated").and_then(|g| g.as_str()).unwrap_or_default().to_string();
    Ok(Catalog { generated, entries, dropped })
}

// ---- The service ----

struct Loaded {
    catalog: Catalog,
    fetched_ms: i64,
}

#[derive(Default)]
struct State {
    loaded: Option<Loaded>,
    stale: bool,
    error: Option<String>,
}

pub struct CatalogService {
    source: Option<CatalogSource>,
    store: Arc<Mutex<Store>>,
    monitor: Arc<Monitor>,
    app_version: semver::Version,
    state: Mutex<State>,
}

impl CatalogService {
    /// Makes no request and reads nothing: the catalog is read only when `refresh` is called (Discover opening).
    pub fn new(source: Option<CatalogSource>, store: Arc<Mutex<Store>>, monitor: Arc<Monitor>, app_version: semver::Version) -> Arc<Self> {
        Arc::new(CatalogService { source, store, monitor, app_version, state: Mutex::new(State::default()) })
    }

    /// The last verified catalog if memory has none yet: the store's copy, verified again (a copy that fails is never shown).
    fn load_cached(&self, st: &mut State) {
        let Some(source) = &self.source else { return };
        if st.loaded.is_some() {
            return;
        }
        let row = match self.store.lock().unwrap().load_catalog() {
            Ok(row) => row,
            Err(e) => {
                self.monitor.log("app", "warn", format!("catalog: the saved copy could not be read: {e}"));
                return;
            }
        };
        let Some((json, minisig, fetched_ms)) = row else { return };
        match verify(&json, &minisig, &source.public_key, source.insecure) {
            Ok(catalog) => {
                self.log_dropped(&catalog);
                st.loaded = Some(Loaded { catalog, fetched_ms });
            }
            Err(e) => {
                self.monitor.log("app", "warn", format!("catalog: the saved copy is refused: {e}"));
                st.error = Some(format!("the saved catalog is refused: {e}"));
            }
        }
    }

    fn log_dropped(&self, catalog: &Catalog) {
        for d in &catalog.dropped {
            self.monitor.log("app", "warn", format!("catalog: entry dropped: {d}"));
        }
    }

    fn view_of(&self, st: &State) -> CatalogView {
        let configured = self.source.is_some();
        let Some(l) = &st.loaded else {
            return CatalogView { configured, entries: vec![], needs_app: vec![], fetched_ms: None, stale: false, error: st.error.clone() };
        };
        let needs_app = l
            .catalog
            .entries
            .iter()
            .filter(|e| semver::Version::parse(&e.min_app_version).is_ok_and(|min| min > self.app_version))
            .map(|e| e.id.clone())
            .collect();
        CatalogView { configured, entries: l.catalog.entries.clone(), needs_app, fetched_ms: Some(l.fetched_ms), stale: st.stale, error: st.error.clone() }
    }

    /// What Discover shows now, from memory, else from the store's copy verified again. No network.
    pub fn view(&self) -> CatalogView {
        let mut st = self.state.lock().unwrap();
        self.load_cached(&mut st);
        self.view_of(&st)
    }

    /// One verified entry by id.
    pub fn entry(&self, id: &str) -> Option<CatalogEntry> {
        let mut st = self.state.lock().unwrap();
        self.load_cached(&mut st);
        st.loaded.as_ref()?.catalog.entries.iter().find(|e| e.id == id).cloned()
    }

    async fn fetch(&self, source: &CatalogSource) -> Result<(Vec<u8>, String), String> {
        if !(source.url.starts_with("https://") || (source.insecure && source.url.starts_with("http://"))) {
            return Err("the catalog's address is not an https:// address".into());
        }
        let client = reqwest::Client::builder().timeout(Duration::from_secs(20)).build().map_err(|e| e.to_string())?;
        let get = |url: String| {
            let client = client.clone();
            async move {
                let resp = client.get(&url).send().await.map_err(|e| format!("could not reach the catalog ({e})"))?;
                if !resp.status().is_success() {
                    return Err(format!("the catalog's server answered {}", resp.status()));
                }
                let body = resp.bytes().await.map_err(|e| format!("could not read the catalog ({e})"))?;
                if body.len() > MAX_BODY {
                    return Err("the catalog is larger than this Lorren reads".to_string());
                }
                Ok(body.to_vec())
            }
        };
        let json = get(source.url.clone()).await?;
        let sig = get(format!("{}.minisig", source.url)).await?;
        let sig = String::from_utf8(sig).map_err(|_| "the catalog's signature is not text".to_string())?;
        Ok((json, sig))
    }

    /// Reads the catalog from its address and verifies it. Verified: cached and shown. Any failure: the last verified
    /// catalog, marked stale, with the reason.
    pub async fn refresh(&self) -> CatalogView {
        let Some(source) = self.source.clone() else { return self.view() };
        let result = match self.fetch(&source).await {
            Ok((json, sig)) => verify(&json, &sig, &source.public_key, source.insecure).map(|c| (c, json, sig)).map_err(|e| e.to_string()),
            Err(e) => Err(e),
        };
        let mut st = self.state.lock().unwrap();
        match result {
            Ok((catalog, json, sig)) => {
                let now = crate::store::now_ms();
                if let Err(e) = self.store.lock().unwrap().save_catalog(&json, &sig, now) {
                    self.monitor.log("app", "warn", format!("catalog: could not keep a copy for offline: {e}"));
                }
                self.log_dropped(&catalog);
                self.monitor.log("app", "info", format!("catalog: {} models ({} dropped)", catalog.entries.len(), catalog.dropped.len()));
                *st = State { loaded: Some(Loaded { catalog, fetched_ms: now }), stale: false, error: None };
            }
            Err(why) => {
                self.monitor.log("app", "warn", format!("catalog: not refreshed: {why}"));
                st.error = None;
                self.load_cached(&mut st);
                let cache_error = st.error.take();
                st.stale = st.loaded.is_some();
                st.error = Some(match cache_error {
                    Some(c) => format!("{why}; {c}"),
                    None => why,
                });
            }
        }
        self.view_of(&st)
    }
}

#[cfg(test)]
mod tests {
    // The fixtures in tests/fixtures/catalog/ were made once with Task 2's `signCatalog` (desktop/scripts/release/catalog.mjs),
    // password `lorren-test` (the M1 kit's), by a one-off node script that is not committed:
    //
    //   import { signCatalog } from 'desktop/scripts/release/catalog.mjs'
    //   // catalog.json = { format: 1, generated: '2026-11-06T09:30:00.000Z', models: [the plan's Qwen entry, its Llama entry,
    //   //                  Qwen's with id 'evil' and file '../evil.gguf'] }, written with JSON.stringify(doc, null, 2) + '\n'
    //   signCatalog({ path: 'catalog.json', keyPath: '.keys/catalog.key', password: 'lorren-test' }) -> catalog.json.minisig
    //   signCatalog({ path: 'catalog.json', keyPath: '.keys/test.key',    password: 'lorren-test' }) -> catalog.json.appkey.minisig
    //   signCatalog({ path: 'catalog.json', keyPath: '.keys/wrong.key',   password: 'lorren-test' }) -> catalog.json.wrongkey.minisig
    //   // catalog-format2.json = the same with format: 2 and only the Qwen entry, signed with catalog.key
    //   //   -> catalog-format2.json.minisig;  catalog-test.pub = .keys/catalog.key.pub
    //
    // (signCatalog writes the minisign text; each call writes <path>.sig, which it deletes.) `.gitattributes` there keeps git
    // from rewriting line endings: one changed byte breaks a signature.
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/catalog");
    const QWEN: &str = "qwen3-4b-instruct-2507-q4_k_m";
    const LLAMA: &str = "llama-3.2-3b-instruct-q4_k_m";

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!("{DIR}/{name}")).unwrap()
    }
    fn fixture_text(name: &str) -> String {
        String::from_utf8(fixture(name)).unwrap()
    }
    fn pubkey() -> String {
        fixture_text("catalog-test.pub")
    }

    fn good() -> CatalogEntry {
        let c = verify(&fixture("catalog.json"), &fixture_text("catalog.json.minisig"), &pubkey(), false).unwrap();
        c.entries.into_iter().find(|e| e.id == QWEN).unwrap()
    }

    #[test]
    fn a_signed_catalog_verifies_and_the_bad_entry_is_dropped_by_name() {
        let c = verify(&fixture("catalog.json"), &fixture_text("catalog.json.minisig"), &pubkey(), false).unwrap();
        assert_eq!(c.generated, "2026-11-06T09:30:00.000Z");
        assert_eq!(c.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), [QWEN, LLAMA]);
        assert_eq!(c.entries[0].size, 2497280736);
        assert_eq!(c.entries[0].licence.name, "Apache-2.0");
        assert_eq!(c.entries[0].source.commit, "ae44f08e1392f39c0e474af10c3ff8355c8b6688");
        assert_eq!(c.entries[0].system_prompt, None);
        assert_eq!(c.dropped.len(), 1, "{:?}", c.dropped);
        assert!(c.dropped[0].starts_with("models[2] (evil): file is not a plain .gguf file name"), "{}", c.dropped[0]);
        // the key in the form the build's config holds it (the .pub file's text, base64-encoded) and as the file's own text
        let text = String::from_utf8(base64::Engine::decode(&base64::engine::general_purpose::STANDARD, pubkey().trim()).unwrap()).unwrap();
        assert!(verify(&fixture("catalog.json"), &fixture_text("catalog.json.minisig"), &text, false).is_ok());
        let bare = text.lines().nth(1).unwrap();
        assert!(verify(&fixture("catalog.json"), &fixture_text("catalog.json.minisig"), bare, false).is_ok());
    }

    #[test]
    fn one_changed_byte_is_refused() {
        let mut json = fixture("catalog.json");
        let at = json.windows(4).position(|w| w == b"Qwen").unwrap();
        json[at] = b'q';
        assert_eq!(verify(&json, &fixture_text("catalog.json.minisig"), &pubkey(), false), Err(CatalogError::Altered));
    }

    #[test]
    fn the_apps_update_key_cannot_sign_a_catalog() {
        let r = verify(&fixture("catalog.json"), &fixture_text("catalog.json.appkey.minisig"), &pubkey(), false);
        assert_eq!(r, Err(CatalogError::UntrustedKey));
    }

    #[test]
    fn another_key_is_refused() {
        let r = verify(&fixture("catalog.json"), &fixture_text("catalog.json.wrongkey.minisig"), &pubkey(), false);
        assert_eq!(r, Err(CatalogError::UntrustedKey));
        assert!(matches!(verify(&fixture("catalog.json"), "not a signature", &pubkey(), false), Err(CatalogError::Unreadable(_))));
        assert!(matches!(verify(&fixture("catalog.json"), &fixture_text("catalog.json.minisig"), "not a key", false), Err(CatalogError::Key(_))));
    }

    #[test]
    fn format_2_is_refused() {
        // correctly signed, and still refused: this Lorren reads format 1 only
        let r = verify(&fixture("catalog-format2.json"), &fixture_text("catalog-format2.json.minisig"), &pubkey(), false);
        assert!(matches!(&r, Err(CatalogError::Format(m)) if m == "format 2"), "{r:?}");
    }

    /// The same table as `validateEntry: each rule` in scripts/release/test/catalog.test.mjs. (A fractional or negative
    /// number cannot be a field here: it fails to read, and the entry is dropped by `filter_models`, below.)
    #[test]
    fn validate_entry_each_rule() {
        assert_eq!(validate_entry(&good(), false), Ok(()));
        let long = |c: char, n: usize| c.to_string().repeat(n);
        let cases: Vec<(&str, Box<dyn Fn(&mut CatalogEntry)>)> = vec![
            ("id", Box::new(|e| e.id = "Qwen".into())),
            ("id", Box::new(|e| e.id = "".into())),
            ("id", Box::new(|e| e.id = long('a', 65))),
            ("name", Box::new(|e| e.name = "".into())),
            ("name", Box::new(|e| e.name = long('n', 81))),
            ("description", Box::new(|e| e.description = long('d', 401))),
            ("version", Box::new(|e| e.version = 0)),
            ("version", Box::new(|e| e.version = 9_007_199_254_740_992)),
            ("architecture", Box::new(|e| e.architecture = "qwen2vl".into())),
            ("quantization", Box::new(|e| e.quantization = "".into())),
            ("quantization", Box::new(|e| e.quantization = long('Q', 17))),
            ("file", Box::new(|e| e.file = "../x.gguf".into())),
            ("file", Box::new(|e| e.file = "a..b.gguf".into())),
            ("file", Box::new(|e| e.file = "dir/x.gguf".into())),
            ("file", Box::new(|e| e.file = "x.bin".into())),
            ("size", Box::new(|e| e.size = 0)),
            ("size", Box::new(|e| e.size = 68_719_476_737)),
            ("sha256", Box::new(|e| e.sha256 = long('a', 63))),
            ("sha256", Box::new(|e| e.sha256 = long('A', 64))),
            ("url", Box::new(|e| e.url = "http://huggingface.co/x/y/resolve/c/f.gguf".into())),
            ("licence.url", Box::new(|e| e.licence = Licence { name: "L".into(), url: "http://example.com/LICENSE".into() })),
            ("licence.name", Box::new(|e| e.licence = Licence { name: "".into(), url: "https://example.com/LICENSE".into() })),
            ("min_app_version", Box::new(|e| e.min_app_version = "1.0".into())),
            ("system_prompt", Box::new(|e| e.system_prompt = Some(long('s', 4001)))),
        ];
        for (field, change) in &cases {
            let mut e = good();
            change(&mut e);
            let why = validate_entry(&e, false).expect_err(field);
            assert!(!why.contains("; "), "{field}: more than one rule: {why}");
            assert!(why.starts_with(&format!("{field} ")), "{field}: {why}");
        }
        // the edges that are allowed
        let mut e = good();
        (e.size, e.system_prompt, e.file, e.url) = (1, Some(long('s', 4000)), "A.GGUF".into(), "https://a.example/f".into());
        assert_eq!(validate_entry(&e, false), Ok(()));
        e.size = 68_719_476_736;
        assert_eq!(validate_entry(&e, false), Ok(()));
        for v in ["1.0.0", "0.1.0-beta.1", "10.20.30"] {
            e.min_app_version = v.into();
            assert_eq!(validate_entry(&e, false), Ok(()), "{v}");
        }
        for v in ["v1.0.0", "1.0.0+build", "01.0.0", "1.0.0-", ""] {
            e.min_app_version = v.into();
            assert!(validate_entry(&e, false).is_err(), "{v}");
        }
    }

    #[test]
    fn filter_models_drops_what_the_rules_refuse_and_names_duplicates_once_each() {
        let q = serde_json::to_value(good()).unwrap();
        let with = |k: &str, v: serde_json::Value| {
            let mut e = q.clone();
            e[k] = v;
            e
        };
        let models = vec![
            q.clone(),
            with("id", "b".into()),                                         // same file as the first (not the same id)
            with("version", serde_json::json!(1.5)),                        // not a whole number: does not read
            with("size", serde_json::json!(-1)),                            // negative: does not read
            serde_json::json!("nope"),
            { let mut e = with("id", "c".into()); e["file"] = "QWEN3-4B-INSTRUCT-2507-Q4_K_M.GGUF".into(); e }, // the other spelling
            { let mut e = with("id", "d".into()); e["file"] = "D.gguf".into(); e },
            with("file", "E.gguf".into()),                                  // same id as the first
        ];
        let (entries, dropped) = filter_models(&models, false);
        assert_eq!(entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), [QWEN, "d"]);
        assert_eq!(dropped.len(), 6, "{dropped:?}");
        assert!(dropped[0].starts_with("models[1] (b): file \"") && dropped[0].ends_with("appears twice"), "{}", dropped[0]);
        assert!(dropped[1].starts_with("models[2] (qwen3-4b-instruct-2507-q4_k_m): "), "{}", dropped[1]);
        assert!(dropped[3].starts_with("models[4]: "), "{}", dropped[3]);
        assert!(dropped[4].starts_with("models[5] (c): file"), "{}", dropped[4]);
        assert_eq!(dropped[5], format!("models[7] ({QWEN}): id \"{QWEN}\" appears twice"));
    }

    // ---- local listeners only ----

    /// A server on 127.0.0.1 that answers each path from `routes` (404 otherwise) and counts the requests it sees.
    fn serve(routes: Vec<(&'static str, Vec<u8>)>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(AtomicUsize::new(0));
        let counter = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                counter.fetch_add(1, Ordering::SeqCst);
                let line = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or_default().to_string();
                let path = line.split(' ').nth(1).unwrap_or("").to_string();
                let (status, body) = match routes.iter().find(|(p, _)| *p == path) {
                    Some((_, b)) => ("200 OK", b.clone()),
                    None => ("404 Not Found", b"no".to_vec()),
                };
                let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(&body);
            }
        });
        (base, seen)
    }

    /// A 127.0.0.1 port with nothing listening on it.
    fn closed_port() -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lorren-catalog-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
    fn store_in(dir: &std::path::Path) -> Arc<Mutex<Store>> {
        Arc::new(Mutex::new(Store::open(&dir.join("lorren.db")).unwrap()))
    }
    fn source(base: &str, insecure: bool) -> Option<CatalogSource> {
        Some(CatalogSource { url: format!("{base}/catalog/catalog.json"), public_key: pubkey(), insecure })
    }
    fn service(src: Option<CatalogSource>, store: Arc<Mutex<Store>>, app: &str) -> Arc<CatalogService> {
        CatalogService::new(src, store, Monitor::new(None), semver::Version::parse(app).unwrap())
    }
    fn routes() -> Vec<(&'static str, Vec<u8>)> {
        vec![("/catalog/catalog.json", fixture("catalog.json")), ("/catalog/catalog.json.minisig", fixture("catalog.json.minisig"))]
    }

    #[test]
    fn http_is_taken_only_from_an_insecure_source() {
        let mut e = good();
        e.url = "http://127.0.0.1:8766/f.gguf".into();
        assert!(validate_entry(&e, false).unwrap_err().starts_with("url is not an https:// address"));
        assert_eq!(validate_entry(&e, true), Ok(()));
        // the licence link opens in a browser: https only, in every build
        e.licence.url = "http://example.com/LICENSE".into();
        assert!(validate_entry(&e, true).unwrap_err().starts_with("licence.url "));

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let (base, seen) = serve(routes());
        let dir = scratch("http");
        // not insecure: refused before any request
        let view = rt.block_on(service(source(&base, false), store_in(&dir), "0.0.1").refresh());
        assert!(view.entries.is_empty());
        assert!(view.error.as_deref().unwrap().contains("https://"), "{:?}", view.error);
        assert_eq!(seen.load(Ordering::SeqCst), 0, "an http source made a request");
        // insecure: taken
        let view = rt.block_on(service(source(&base, true), store_in(&dir), "0.0.1").refresh());
        assert_eq!(view.entries.len(), 2, "{:?}", view.error);
        assert_eq!(view.error, None);
        assert_eq!(seen.load(Ordering::SeqCst), 2, "the catalog and its signature");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn an_entry_for_a_newer_app_is_listed_in_needs_app() {
        let (base, _) = serve(routes());
        let dir = scratch("needs");
        let old = service(source(&base, true), store_in(&dir), "0.0.1").refresh().await;
        assert_eq!(old.entries.len(), 2);
        assert_eq!(old.needs_app, [QWEN, LLAMA]); // both ask for 0.1.0
        let new = service(source(&base, true), store_in(&dir), "0.1.0").refresh().await;
        assert!(new.needs_app.is_empty());
        let beta = service(source(&base, true), store_in(&dir), "0.1.0-beta.1").refresh().await;
        assert_eq!(beta.needs_app.len(), 2, "a pre-release comes before its release");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_from_config_reads_the_overlay_and_none_without_it() {
        use serde_json::json;
        let full = json!({"url": "http://127.0.0.1:8766/catalog/catalog.json", "pubkey": "KEY", "insecure": true});
        assert_eq!(
            source_from_config(Some(&full)),
            Some(CatalogSource { url: "http://127.0.0.1:8766/catalog/catalog.json".into(), public_key: "KEY".into(), insecure: true })
        );
        let release = json!({"url": "https://x.example/catalog.json", "pubkey": "KEY"});
        assert_eq!(source_from_config(Some(&release)).map(|s| s.insecure), Some(false));
        assert_eq!(source_from_config(Some(&json!({"url": "https://x.example/c.json"}))), None, "no pubkey");
        assert_eq!(source_from_config(Some(&json!({"pubkey": "KEY"}))), None, "no url");
        assert_eq!(source_from_config(Some(&json!({"url": "", "pubkey": "KEY"}))), None, "an empty url");
        assert_eq!(source_from_config(Some(&json!({"url": 5, "pubkey": "KEY"}))), None, "not text");
        assert_eq!(source_from_config(None), None);
    }

    #[tokio::test]
    async fn offline_discover_shows_the_cached_catalog_verified_again() {
        let dir = scratch("offline");
        let store = store_in(&dir);
        // an earlier session read the catalog
        let (base, _) = serve(routes());
        let first = service(source(&base, true), store.clone(), "0.1.0").refresh().await;
        assert_eq!((first.entries.len(), first.stale, first.error.clone()), (2, false, None));
        assert!(first.fetched_ms.is_some());
        // a new session, offline: view() already shows the saved copy; a refresh against a closed port keeps it, stale
        let offline = service(source(&closed_port(), true), store.clone(), "0.1.0");
        let cached = offline.view();
        assert_eq!((cached.entries.len(), cached.stale, cached.error.clone()), (2, false, None));
        let view = offline.refresh().await;
        assert_eq!(view.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), [QWEN, LLAMA]);
        assert!(view.stale);
        assert!(view.error.as_deref().is_some_and(|e| e.contains("could not reach")), "{:?}", view.error);
        assert_eq!(view.fetched_ms, first.fetched_ms, "the time of the read that is shown, not of the failed one");
        assert!(offline.entry(QWEN).is_some() && offline.entry("evil").is_none());
        // the saved copy altered in the store: never shown
        let (json, sig, at) = store.lock().unwrap().load_catalog().unwrap().unwrap();
        let mut altered = json.clone();
        let i = altered.windows(4).position(|w| w == b"Qwen").unwrap();
        altered[i] = b'q';
        store.lock().unwrap().save_catalog(&altered, &sig, at).unwrap();
        let tampered = service(source(&closed_port(), true), store.clone(), "0.1.0");
        let v = tampered.view();
        assert!(v.entries.is_empty() && !v.stale && v.error.is_some(), "{v:?}");
        let v = tampered.refresh().await;
        assert!(v.entries.is_empty() && !v.stale && v.fetched_ms.is_none(), "{v:?}");
        assert!(v.error.as_deref().is_some_and(|e| e.contains("could not reach") && e.contains("refused")), "{:?}", v.error);
        assert!(tampered.entry(QWEN).is_none());
        // a build with no catalog says so
        let none = service(None, store, "0.1.0").refresh().await;
        assert!(!none.configured && none.entries.is_empty() && none.error.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_refresh_that_fails_verification_keeps_the_last_verified_catalog_and_caches_nothing() {
        let dir = scratch("badsig");
        let store = store_in(&dir);
        let (good_base, _) = serve(routes());
        let svc = service(source(&good_base, true), store.clone(), "0.1.0");
        let first = svc.refresh().await;
        // the same address now answers with the app key's signature
        let (bad_base, _) = serve(vec![
            ("/catalog/catalog.json", fixture("catalog.json")),
            ("/catalog/catalog.json.minisig", fixture("catalog.json.appkey.minisig")),
        ]);
        let svc2 = service(source(&bad_base, true), store.clone(), "0.1.0");
        let v = svc2.refresh().await;
        assert!(v.stale && v.entries.len() == 2, "{v:?}");
        assert!(v.error.as_deref().unwrap().contains("does not trust"), "{:?}", v.error);
        assert_eq!(store.lock().unwrap().load_catalog().unwrap().unwrap().2, first.fetched_ms.unwrap(), "the cache was not replaced");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_catalog_is_read_only_by_refresh() {
        // a listener that accepts nothing the test does not count: new() and view() must not even connect
        let (base, seen) = serve(routes());
        let dir = scratch("readonly");
        let svc = service(source(&base, true), store_in(&dir), "0.1.0");
        let view = svc.view();
        assert_eq!((view.configured, view.entries.len(), view.fetched_ms, view.stale, view.error), (true, 0, None, false, None));
        assert!(svc.entry(QWEN).is_none());
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert_eq!(seen.load(Ordering::SeqCst), 0, "new/view/entry made a request");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
