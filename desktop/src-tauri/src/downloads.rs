//! The download manager: models the user starts from Discover. A download resumes (HTTP Range), retries with backoff,
//! is checked against its sha256 as the bytes arrive, and joins the library only once it matches.
//!
//! One worker per entry: a per-entry async lock is held by a worker for its whole run, so a paused-then-resumed (or
//! cancelled-then-restarted) entry never has two writers on its `.part`. Every request starts at the catalog's address:
//! the CDN address a redirect leads to is signed and short-lived, so nothing keeps it.

use crate::catalog::{validate_entry, CatalogEntry};
use crate::library::{self, ModelInfo};
use crate::store::{now_ms, DownloadRow, Store};
use futures_util::StreamExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use ts_rs::TS;

/// A download never leaves less than this free (spec §5 keeps 1 GB of memory for the machine; the disk gets the same).
pub const MARGIN_BYTES: u64 = 1 << 30;
const EMIT_EVERY: Duration = Duration::from_millis(250);
const RETRY_AFTER_CAP: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub first: Duration,
    pub cap: Duration,
    /// No byte for this long is a dropped connection.
    pub stall: Duration,
}

impl RetryPolicy {
    pub const DEFAULT: RetryPolicy = RetryPolicy { first: Duration::from_secs(1), cap: Duration::from_secs(30), stall: Duration::from_secs(30) };
}

/// `min(cap, first x 2^attempt)`: 1, 2, 4, 8, 16, 30, 30 ... s.
pub fn backoff(p: &RetryPolicy, attempt: u32) -> Duration {
    let mult = 1u32 << attempt.min(31);
    p.first.checked_mul(mult).unwrap_or(p.cap).min(p.cap)
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1_073_741_824.0)
}

pub fn room_check(free: u64, remaining: u64) -> Result<(), String> {
    if free.saturating_sub(MARGIN_BYTES) >= remaining {
        Ok(())
    } else {
        Err(format!("needs {} with 1 GB to spare; {} free", gb(remaining), gb(free)))
    }
}

/// Free bytes on the volume of `dir` (or of its nearest existing parent, for a folder not made yet).
#[cfg(windows)]
pub fn free_bytes(dir: &Path) -> io::Result<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let mut at = dir;
    while !at.exists() {
        match at.parent() {
            Some(p) if !p.as_os_str().is_empty() => at = p,
            _ => break,
        }
    }
    let wide: Vec<u16> = at.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut avail = 0u64;
    unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut avail), None, None) }.map_err(|e| io::Error::other(e.to_string()))?;
    Ok(avail)
}

#[cfg(not(windows))]
pub fn free_bytes(_dir: &Path) -> io::Result<u64> {
    Err(io::Error::other("free space is only read on Windows"))
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DownloadView {
    pub entry_id: String,
    pub name: String,
    pub file: String,
    #[ts(type = "number")]
    pub version: u32,
    /// downloading | paused | verifying | done | failed
    pub state: String,
    #[ts(type = "number")]
    pub done: u64,
    #[ts(type = "number")]
    pub total: u64,
    #[ts(type = "number")]
    pub bytes_per_sec: u64,
    #[ts(type = "number | null")]
    pub retry_in_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
pub struct DiskRoom {
    pub dir: String,
    #[ts(type = "number")]
    pub free_bytes: u64,
}

pub fn disk_room(dir: &Path) -> DiskRoom {
    DiskRoom { dir: dir.to_string_lossy().to_string(), free_bytes: free_bytes(dir).unwrap_or(0) }
}

pub fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Opens the `.part` for writing at `offset` (the file is cut to it first: 0 starts it over).
pub fn open_part(p: &Path, offset: u64) -> io::Result<Box<dyn Write + Send>> {
    let mut f = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(p)?;
    f.set_len(offset)?;
    f.seek(SeekFrom::Start(offset))?;
    Ok(Box::new(f))
}

type FreeFn = Arc<dyn Fn(&Path) -> io::Result<u64> + Send + Sync>;
type OpenFn = Arc<dyn Fn(&Path, u64) -> io::Result<Box<dyn Write + Send>> + Send + Sync>;

pub struct DownloadConfig {
    pub dir: PathBuf,
    /// A test build's catalog may name `http://` addresses.
    pub insecure: bool,
    pub policy: RetryPolicy,
    pub default_prefix: String,
    pub free_bytes: FreeFn,
    pub open_part: OpenFn,
    /// A finished download joins the shared library only after the startup scan has published it.
    pub library_ready: Option<tokio::sync::watch::Receiver<bool>>,
}

impl DownloadConfig {
    pub fn new(dir: PathBuf, insecure: bool, default_prefix: String) -> Self {
        DownloadConfig {
            dir,
            insecure,
            policy: RetryPolicy::DEFAULT,
            default_prefix,
            free_bytes: Arc::new(free_bytes),
            open_part: Arc::new(open_part),
            library_ready: None,
        }
    }
}

type EmitFn = Arc<dyn Fn(&DownloadView) + Send + Sync>;
type ModelFn = Arc<dyn Fn(ModelInfo, bool) + Send + Sync>;

/// What stops a running worker, shared with the manager.
struct Ctl {
    token: CancellationToken,
    /// Set by `cancel`: the worker deletes the `.part` once its writer is closed.
    discard: AtomicBool,
}

struct Job {
    row: DownloadRow,
    view: DownloadView,
    gen: u64,
    /// Some while a worker may still be running for this job.
    ctl: Option<Arc<Ctl>>,
    last_emit: Instant,
    rate_at: Instant,
    rate_from: u64,
}

enum Outcome {
    Stopped,
    Done,
    Failed(String),
}

pub struct DownloadManager {
    cfg: DownloadConfig,
    store: Arc<Mutex<Store>>,
    library: Arc<RwLock<Vec<ModelInfo>>>,
    emit: EmitFn,
    on_model: ModelFn,
    /// Built by the first worker, not by `new`: `new` runs in `setup`, on the opening's path.
    client: OnceLock<reqwest::Client>,
    jobs: Mutex<HashMap<String, Job>>,
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    next_gen: AtomicU64,
}

fn running(state: &str) -> bool {
    state == "downloading" || state == "verifying"
}

fn view_of(row: &DownloadRow, state: &str, done: u64) -> DownloadView {
    DownloadView {
        entry_id: row.entry_id.clone(),
        name: row.name.clone(),
        file: row.file.clone(),
        version: u32::try_from(row.version).unwrap_or(u32::MAX),
        state: state.into(),
        done,
        total: row.size,
        bytes_per_sec: 0,
        retry_in_ms: None,
        error: if state == "failed" { row.error.clone() } else { None },
    }
}

fn already_in(file: &str, dir: &Path) -> String {
    format!("{file} is already in {}", dir.display())
}

fn stem(file: &str) -> &str {
    if file.to_ascii_lowercase().ends_with(".gguf") {
        &file[..file.len() - 5]
    } else {
        file
    }
}

fn is_disk_full(e: &io::Error) -> bool {
    matches!(e.raw_os_error(), Some(112) | Some(39))
}

fn parse_range_start(h: Option<&reqwest::header::HeaderValue>) -> Option<u64> {
    let s = h?.to_str().ok()?.strip_prefix("bytes ")?;
    s.split('-').next()?.trim().parse().ok()
}

impl DownloadManager {
    pub fn new(cfg: DownloadConfig, store: Arc<Mutex<Store>>, library: Arc<RwLock<Vec<ModelInfo>>>, emit: EmitFn, on_model: ModelFn) -> Arc<Self> {
        let m = Arc::new(DownloadManager {
            cfg,
            store,
            library,
            emit,
            on_model,
            client: OnceLock::new(),
            jobs: Mutex::new(HashMap::new()),
            locks: Mutex::new(HashMap::new()),
            next_gen: AtomicU64::new(1),
        });
        m.load();
        m
    }

    pub fn dir(&self) -> &Path {
        &self.cfg.dir
    }

    fn part_path(&self, file: &str) -> PathBuf {
        self.cfg.dir.join(format!("{file}.part"))
    }

    /// The store's rows as jobs with no worker: a row marked downloading is shown paused (nothing runs yet).
    fn load(&self) {
        let rows = self.store.lock().unwrap().downloads().unwrap_or_default();
        let mut jobs = self.jobs.lock().unwrap();
        jobs.clear();
        for row in rows {
            let (state, done) = match row.state.as_str() {
                "done" => ("done", row.size),
                "failed" => ("failed", file_len(&self.part_path(&row.file))),
                _ => ("paused", file_len(&self.part_path(&row.file))),
            };
            let view = view_of(&row, state, done);
            let gen = self.next_gen.fetch_add(1, Ordering::SeqCst);
            jobs.insert(row.entry_id.clone(), Job { row, view, gen, ctl: None, last_emit: Instant::now(), rate_at: Instant::now(), rate_from: done });
        }
    }

    pub fn list(&self) -> Vec<DownloadView> {
        let jobs = self.jobs.lock().unwrap();
        let mut v: Vec<(i64, DownloadView)> = jobs.values().map(|j| (j.row.created_ms, j.view.clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.entry_id.cmp(&b.1.entry_id)));
        v.into_iter().map(|x| x.1).collect()
    }

    /// At start nothing runs: running rows become paused, a `.part` no row owns is deleted.
    pub fn reconcile_at_start(&self) {
        let rows = {
            let st = self.store.lock().unwrap();
            let _ = st.pause_running_downloads();
            st.downloads().unwrap_or_default()
        };
        if let Ok(rd) = std::fs::read_dir(&self.cfg.dir) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.ends_with(".part") && !rows.iter().any(|r| format!("{}.part", r.file) == name) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
        self.load();
    }

    pub fn start(self: &Arc<Self>, entry: &CatalogEntry) -> Result<DownloadView, String> {
        validate_entry(entry, self.cfg.insecure)?;
        let version = i64::try_from(entry.version).map_err(|_| "the version is too large".to_string())?;
        {
            let jobs = self.jobs.lock().unwrap();
            if let Some(j) = jobs.get(&entry.id) {
                if running(&j.view.state) {
                    if j.row.version == version {
                        return Ok(j.view.clone());
                    }
                    return Err(format!("another version of {} is downloading", entry.name));
                }
            }
        }
        let rows = self.store.lock().unwrap().downloads().map_err(|e| e.to_string())?;
        let existing = rows.iter().find(|r| r.entry_id == entry.id);
        let dir = &self.cfg.dir;
        let (file, resume, created) = match existing {
            Some(r) if r.state != "done" && r.version == version => (r.file.clone(), true, r.created_ms),
            Some(r) if r.state == "done" && r.version < version => (format!("{}.v{}.gguf", stem(&entry.file), entry.version), false, now_ms()),
            Some(r) if r.state == "done" => return Err(already_in(&r.file, dir)),
            _ => (entry.file.clone(), false, now_ms()),
        };
        if dir.join(&file).exists() || rows.iter().any(|r| r.entry_id != entry.id && r.file.eq_ignore_ascii_case(&file)) {
            return Err(already_in(&file, dir));
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
        if !resume {
            if let Some(r) = existing {
                if r.state != "done" {
                    let _ = std::fs::remove_file(self.part_path(&r.file));
                }
            }
            let _ = std::fs::remove_file(self.part_path(&file));
        }
        let have = if resume { file_len(&self.part_path(&file)) } else { 0 };
        let free = (self.cfg.free_bytes)(dir).map_err(|e| format!("could not read the free space: {e}"))?;
        room_check(free, entry.size.saturating_sub(have))?;
        let row = DownloadRow {
            entry_id: entry.id.clone(),
            name: entry.name.clone(),
            version,
            url: entry.url.clone(),
            file,
            size: entry.size,
            sha256: entry.sha256.to_ascii_lowercase(),
            system_prompt: entry.system_prompt.clone(),
            state: "downloading".into(),
            error: None,
            created_ms: created,
            updated_ms: now_ms(),
        };
        self.store.lock().unwrap().upsert_download(&row).map_err(|e| e.to_string())?;
        Ok(self.launch(row, have))
    }

    /// Installs a job for the row and starts its worker. The caller has seen that none is running.
    fn launch(self: &Arc<Self>, row: DownloadRow, have: u64) -> DownloadView {
        let gen = self.next_gen.fetch_add(1, Ordering::SeqCst);
        let ctl = Arc::new(Ctl { token: CancellationToken::new(), discard: AtomicBool::new(false) });
        let view = view_of(&row, "downloading", have);
        {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(row.entry_id.clone(), Job { row: row.clone(), view: view.clone(), gen, ctl: Some(ctl.clone()), last_emit: Instant::now(), rate_at: Instant::now(), rate_from: have });
        }
        (self.emit)(&view);
        tauri::async_runtime::spawn(self.clone().work(row, gen, ctl));
        view
    }

    pub fn pause(&self, id: &str) -> Result<(), String> {
        let view = {
            let mut jobs = self.jobs.lock().unwrap();
            let j = jobs.get_mut(id).ok_or_else(|| format!("no such download: {id}"))?;
            if j.view.state == "paused" {
                return Ok(());
            }
            if !running(&j.view.state) {
                return Err(format!("{} is not downloading", j.row.name));
            }
            if let Some(c) = &j.ctl {
                c.token.cancel();
            }
            j.view.state = "paused".into();
            j.view.retry_in_ms = None;
            j.view.bytes_per_sec = 0;
            let _ = self.store.lock().unwrap().set_download_state(id, "paused", None);
            j.view.clone()
        };
        (self.emit)(&view);
        Ok(())
    }

    pub fn resume(self: &Arc<Self>, id: &str) -> Result<(), String> {
        let row = {
            let jobs = self.jobs.lock().unwrap();
            let j = jobs.get(id).ok_or_else(|| format!("no such download: {id}"))?;
            match j.view.state.as_str() {
                s if running(s) => return Ok(()),
                "done" => return Err(format!("{} is already downloaded", j.row.name)),
                _ => j.row.clone(),
            }
        };
        let dir = &self.cfg.dir;
        if dir.join(&row.file).exists() {
            return Err(already_in(&row.file, dir));
        }
        let have = file_len(&self.part_path(&row.file)).min(row.size);
        let free = (self.cfg.free_bytes)(dir).map_err(|e| format!("could not read the free space: {e}"))?;
        room_check(free, row.size.saturating_sub(have))?;
        let _ = self.store.lock().unwrap().set_download_state(id, "downloading", None);
        let mut row = row;
        row.state = "downloading".into();
        row.error = None;
        self.launch(row, have);
        Ok(())
    }

    pub fn cancel(&self, id: &str) -> Result<(), String> {
        let job = self.jobs.lock().unwrap().remove(id).ok_or_else(|| format!("no such download: {id}"))?;
        if let Some(c) = &job.ctl {
            c.discard.store(true, Ordering::SeqCst);
            c.token.cancel();
        }
        let _ = self.store.lock().unwrap().delete_download(id);
        if job.view.state != "done" {
            let _ = std::fs::remove_file(self.part_path(&job.row.file));
        }
        Ok(())
    }

    fn client(&self) -> &reqwest::Client {
        self.client.get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(20))
                .redirect(reqwest::redirect::Policy::limited(10))
                .build()
                .expect("an http client")
        })
    }

    fn lock_for(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.locks.lock().unwrap().entry(id.to_string()).or_default().clone()
    }

    /// Changes the job's view if it is still this worker's job; `f` says whether to emit at once (else at most every 250 ms).
    ///
    /// Not once the job's token is cancelled (`pause` cancels it under this same lock and sets "paused"): a worker that
    /// had already taken a chunk must not move the view back to downloading. Terminal changes use `apply_terminal`.
    fn apply(&self, id: &str, gen: u64, f: impl FnOnce(&mut Job) -> bool) {
        self.apply_at(id, gen, true, f)
    }

    /// For the end of a job (failed, done): applies even after a pause, which a finished download must win over.
    fn apply_terminal(&self, id: &str, gen: u64, f: impl FnOnce(&mut Job) -> bool) {
        self.apply_at(id, gen, false, f)
    }

    fn apply_at(&self, id: &str, gen: u64, live_only: bool, f: impl FnOnce(&mut Job) -> bool) {
        let view = {
            let mut jobs = self.jobs.lock().unwrap();
            let Some(j) = jobs.get_mut(id) else { return };
            if j.gen != gen || (live_only && j.ctl.as_ref().is_some_and(|c| c.token.is_cancelled())) {
                return;
            }
            let force = f(j);
            if force || j.last_emit.elapsed() >= EMIT_EVERY {
                j.last_emit = Instant::now();
                Some(j.view.clone())
            } else {
                None
            }
        };
        if let Some(v) = view {
            (self.emit)(&v);
        }
    }

    fn fail(&self, id: &str, gen: u64, msg: &str) {
        {
            let jobs = self.jobs.lock().unwrap();
            if jobs.get(id).map(|j| j.gen) != Some(gen) {
                return;
            }
            let _ = self.store.lock().unwrap().set_download_state(id, "failed", Some(msg));
        }
        self.apply_terminal(id, gen, |j| {
            j.view.state = "failed".into();
            j.view.error = Some(msg.to_string());
            j.view.retry_in_ms = None;
            j.view.bytes_per_sec = 0;
            true
        });
    }

    async fn work(self: Arc<Self>, row: DownloadRow, gen: u64, ctl: Arc<Ctl>) {
        let id = row.entry_id.clone();
        let lock = self.lock_for(&id);
        let guard = tokio::select! {
            _ = ctl.token.cancelled() => None,
            g = lock.lock_owned() => Some(g),
        };
        if let Some(_g) = guard {
            let outcome = if ctl.token.is_cancelled() { Outcome::Stopped } else { self.run(&row, gen, &ctl).await };
            if let Outcome::Failed(msg) = outcome {
                self.fail(&id, gen, &msg);
            }
            if ctl.discard.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(self.part_path(&row.file));
            }
        }
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(j) = jobs.get_mut(&id) {
            if j.gen == gen {
                j.ctl = None;
            }
        }
    }

    /// Sleeps `delay` (shown as `retry_in_ms`); false when stopped meanwhile.
    async fn wait_retry(&self, id: &str, gen: u64, ctl: &Ctl, delay: Duration) -> bool {
        self.apply(id, gen, |j| {
            j.view.state = "downloading".into();
            j.view.retry_in_ms = Some(delay.as_millis() as u64);
            j.view.bytes_per_sec = 0;
            true
        });
        tokio::select! {
            _ = ctl.token.cancelled() => false,
            _ = tokio::time::sleep(delay) => true,
        }
    }

    fn reset_to_zero(&self, id: &str, gen: u64) {
        self.apply(id, gen, |j| {
            j.view.done = 0;
            false
        });
    }

    async fn run(&self, row: &DownloadRow, gen: u64, ctl: &Ctl) -> Outcome {
        let id = row.entry_id.as_str();
        let part = self.part_path(&row.file);
        let size = row.size;
        let policy = self.cfg.policy;
        let mut have = file_len(&part);
        if have > size {
            let _ = std::fs::remove_file(&part);
            have = 0;
        }
        let mut hasher = Sha256::new();
        if have > 0 {
            let (p, t, n) = (part.clone(), ctl.token.clone(), have);
            match tokio::task::spawn_blocking(move || rehash(&p, n, &t)).await {
                Ok(Ok(Some(h))) => hasher = h,
                Ok(Ok(None)) => return Outcome::Stopped,
                _ => {
                    let _ = std::fs::remove_file(&part);
                    have = 0;
                }
            }
        }
        self.apply(id, gen, |j| {
            j.view.done = have;
            false
        });
        let mut attempt = 0u32;
        let mut resets = 0u32;
        loop {
            if ctl.token.is_cancelled() {
                return Outcome::Stopped;
            }
            if have == size {
                return self.finish(row, gen, ctl, hasher).await;
            }
            let mut req = self.client().get(&row.url);
            if have > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
            }
            let sent = tokio::select! {
                _ = ctl.token.cancelled() => return Outcome::Stopped,
                r = tokio::time::timeout(policy.stall, req.send()) => r,
            };
            let resp = match sent {
                Ok(Ok(r)) => r,
                _ => {
                    if !self.wait_retry(id, gen, ctl, backoff(&policy, attempt)).await {
                        return Outcome::Stopped;
                    }
                    attempt += 1;
                    continue;
                }
            };
            let status = resp.status().as_u16();
            match status {
                206 => {
                    if parse_range_start(resp.headers().get(reqwest::header::CONTENT_RANGE)) != Some(have) {
                        resets += 1;
                        if resets > 3 {
                            return Outcome::Failed("the server answered a different range than asked".into());
                        }
                        let _ = std::fs::remove_file(&part);
                        have = 0;
                        hasher = Sha256::new();
                        self.reset_to_zero(id, gen);
                        continue;
                    }
                }
                200 => {
                    if have > 0 {
                        have = 0;
                        hasher = Sha256::new();
                        self.reset_to_zero(id, gen);
                    }
                }
                416 => {
                    if have == size {
                        return self.finish(row, gen, ctl, hasher).await;
                    }
                    if have == 0 {
                        return Outcome::Failed("416 from the server".into());
                    }
                    let _ = std::fs::remove_file(&part);
                    have = 0;
                    hasher = Sha256::new();
                    self.reset_to_zero(id, gen);
                    continue;
                }
                408 | 429 | 500..=599 => {
                    let mut delay = backoff(&policy, attempt);
                    if status == 429 || status == 503 {
                        if let Some(s) = resp.headers().get(reqwest::header::RETRY_AFTER).and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse::<u64>().ok()) {
                            delay = Duration::from_secs(s).min(RETRY_AFTER_CAP);
                        }
                    }
                    if !self.wait_retry(id, gen, ctl, delay).await {
                        return Outcome::Stopped;
                    }
                    attempt += 1;
                    continue;
                }
                s => return Outcome::Failed(format!("{s} from the server")),
            }
            let mut w = match (self.cfg.open_part)(&part, have) {
                Ok(w) => w,
                Err(e) if is_disk_full(&e) => return Outcome::Failed("the disk is full".into()),
                Err(e) => return Outcome::Failed(format!("could not write {}: {e}", row.file)),
            };
            let mut stream = resp.bytes_stream();
            loop {
                let next = tokio::select! {
                    _ = ctl.token.cancelled() => {
                        let _ = w.flush();
                        return Outcome::Stopped;
                    }
                    n = tokio::time::timeout(policy.stall, stream.next()) => n,
                };
                let chunk = match next {
                    Ok(Some(Ok(c))) => c,
                    _ => break, // the end, an error, or a stall: below decides
                };
                if have + chunk.len() as u64 > size {
                    drop(w);
                    let _ = std::fs::remove_file(&part);
                    return Outcome::Failed("the server sent more than the catalog's size".into());
                }
                if let Err(e) = w.write_all(&chunk) {
                    let _ = w.flush();
                    return Outcome::Failed(if is_disk_full(&e) { "the disk is full".into() } else { format!("could not write {}: {e}", row.file) });
                }
                hasher.update(&chunk);
                have += chunk.len() as u64;
                attempt = 0;
                self.apply(id, gen, |j| {
                    let force = j.view.retry_in_ms.is_some() || j.view.state != "downloading";
                    j.view.state = "downloading".into();
                    j.view.retry_in_ms = None;
                    j.view.done = have;
                    let dt = j.rate_at.elapsed();
                    if dt >= EMIT_EVERY {
                        j.view.bytes_per_sec = (have.saturating_sub(j.rate_from) as f64 / dt.as_secs_f64()) as u64;
                        j.rate_at = Instant::now();
                        j.rate_from = have;
                    }
                    force
                });
            }
            let _ = w.flush();
            drop(w);
            if have == size {
                continue; // the top verifies
            }
            // A drop, an error or a stall before the end: resume from `have` after the backoff.
            if !self.wait_retry(id, gen, ctl, backoff(&policy, attempt)).await {
                return Outcome::Stopped;
            }
            attempt += 1;
        }
    }

    /// All the bytes are here: check them, then the file joins the library.
    async fn finish(&self, row: &DownloadRow, gen: u64, ctl: &Ctl, hasher: Sha256) -> Outcome {
        let id = row.entry_id.as_str();
        self.apply(id, gen, |j| {
            j.view.state = "verifying".into();
            j.view.retry_in_ms = None;
            j.view.bytes_per_sec = 0;
            true
        });
        let part = self.part_path(&row.file);
        if hex(&hasher.finalize()) != row.sha256 {
            let _ = std::fs::remove_file(&part);
            return Outcome::Failed("the file did not match its checksum and was deleted".into());
        }
        let dir = &self.cfg.dir;
        let target = dir.join(&row.file);
        if target.exists() {
            return Outcome::Failed(already_in(&row.file, dir));
        }
        if ctl.token.is_cancelled() {
            return Outcome::Stopped;
        }
        if let Some(prompt) = &row.system_prompt {
            let mf = dir.join(format!("{}.modelfile", stem(&row.file)));
            if !mf.exists() {
                // The Modelfile reader ends a value at the first closing triple quote, so one inside is softened.
                let text = format!("FROM {}\nSYSTEM \"\"\"{}\"\"\"\n", row.file, prompt.replace("\"\"\"", "\"\"'"));
                if let Err(e) = std::fs::write(&mf, text) {
                    return Outcome::Failed(format!("could not write {}: {e}", mf.display()));
                }
            }
        }
        if let Err(e) = std::fs::rename(&part, &target) {
            return Outcome::Failed(format!("could not move the download into place: {e}"));
        }
        if let Some(mut rx) = self.cfg.library_ready.clone() {
            let _ = rx.wait_for(|ready| *ready).await.is_ok();
        }
        let (model, none_before) = {
            let mut lib = self.library.write().unwrap();
            let none = !lib.iter().any(|m| m.runnable);
            (library::add_file(&mut lib, &target, &self.cfg.default_prefix), none)
        };
        let _ = self.store.lock().unwrap().set_download_state(id, "done", None);
        let size = row.size;
        self.apply_terminal(id, gen, |j| {
            j.view.state = "done".into();
            j.view.done = size;
            j.view.error = None;
            j.view.retry_in_ms = None;
            j.view.bytes_per_sec = 0;
            true
        });
        (self.on_model)(model, none_before);
        Outcome::Done
    }
}

/// The sha256 state of the first `len` bytes of a `.part` (None when stopped meanwhile).
fn rehash(p: &Path, len: u64, stop: &CancellationToken) -> io::Result<Option<Sha256>> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        if stop.is_cancelled() {
            return Ok(None);
        }
        let want = buf.len().min(left as usize);
        let n = f.read(&mut buf[..want])?;
        if n == 0 {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        h.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(Some(h))
}


/// An in-process HTTP/1.1 server on 127.0.0.1 that serves one body with Range, and misbehaves on request.
#[cfg(test)]
mod server {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[derive(Clone, Debug)]
    pub struct Req {
        pub path: String,
        pub range: Option<String>,
        pub at: Instant,
    }

    #[derive(Default)]
    pub struct Ctl {
        pub body: Vec<u8>,
        /// Once: send this many bytes of the next body, then close the connection.
        pub drop_after: Option<usize>,
        /// Once: send this many bytes of the next body, then say nothing more and keep the connection open.
        pub hold_after: Option<usize>,
        pub fail_500_once: bool,
        pub ignore_range: bool,
        /// Answer every request with this status.
        pub status: Option<u16>,
        /// `/file` answers 302 to `/cdn?once=<n>`, which serves once (a signed address that expires).
        pub redirect: bool,
        /// Bytes beyond the body, sent after it (the Content-Length counts them).
        pub extra: usize,
        pub log: Vec<Req>,
        redirects: u32,
        cdn_used: HashSet<u32>,
    }

    pub struct Server {
        pub ctl: Arc<Mutex<Ctl>>,
        pub addr: String,
    }

    enum Reply {
        Plain(u16),
        Redirect(String),
        Body { status: u16, headers: String, data: Vec<u8>, cut: Option<usize>, hold: Option<usize> },
    }

    fn serve(c: &mut Ctl, range: Option<&str>) -> Reply {
        let total = c.body.len();
        let from = if c.ignore_range { None } else { range.and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.strip_suffix('-')).and_then(|n| n.parse::<usize>().ok()) };
        let (status, headers, mut data) = match from {
            Some(n) if n == total => (416, format!("Content-Range: bytes */{total}\r\n"), Vec::new()),
            Some(n) if n > 0 && n < total => (206, format!("Content-Range: bytes {n}-{}/{total}\r\n", total - 1), c.body[n..].to_vec()),
            _ => (200, String::new(), c.body.clone()),
        };
        data.extend(std::iter::repeat(0u8).take(c.extra));
        let (cut, hold) = if data.is_empty() { (None, None) } else { (c.drop_after.take(), c.hold_after.take()) };
        Reply::Body { status, headers, data, cut, hold }
    }

    async fn handle(mut s: TcpStream, ctl: Arc<Mutex<Ctl>>) {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 2048];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            match s.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
        let head = String::from_utf8_lossy(&buf).to_string();
        let path = head.lines().next().and_then(|l| l.split(' ').nth(1)).unwrap_or("/").to_string();
        let range = head.lines().find(|l| l.to_ascii_lowercase().starts_with("range:")).map(|l| l[6..].trim().to_string());
        let reply = {
            let mut c = ctl.lock().unwrap();
            c.log.push(Req { path: path.clone(), range: range.clone(), at: Instant::now() });
            if let Some(code) = c.status {
                Reply::Plain(code)
            } else if let Some(id) = path.strip_prefix("/cdn?once=").and_then(|n| n.parse::<u32>().ok()) {
                if c.cdn_used.insert(id) {
                    serve(&mut c, range.as_deref())
                } else {
                    Reply::Plain(403)
                }
            } else if path == "/file" {
                if c.redirect {
                    c.redirects += 1;
                    Reply::Redirect(format!("/cdn?once={}", c.redirects))
                } else if c.fail_500_once {
                    c.fail_500_once = false;
                    Reply::Plain(500)
                } else {
                    serve(&mut c, range.as_deref())
                }
            } else {
                Reply::Plain(404)
            }
        };
        match reply {
            Reply::Plain(code) => {
                let _ = s.write_all(format!("HTTP/1.1 {code} X\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await;
            }
            Reply::Redirect(to) => {
                let _ = s.write_all(format!("HTTP/1.1 302 Found\r\nLocation: {to}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await;
            }
            Reply::Body { status, headers, data, cut, hold } => {
                let h = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n{headers}\r\n", data.len());
                let _ = s.write_all(h.as_bytes()).await;
                if let Some(n) = cut.or(hold) {
                    let _ = s.write_all(&data[..n.min(data.len())]).await;
                    let _ = s.flush().await;
                    if hold.is_some() {
                        std::future::pending::<()>().await;
                    }
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    return;
                }
                let _ = s.write_all(&data).await;
                let _ = s.shutdown().await;
            }
        }
    }

    impl Server {
        pub async fn start(body: Vec<u8>) -> Server {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let ctl = Arc::new(Mutex::new(Ctl { body, ..Ctl::default() }));
            let c = ctl.clone();
            tokio::spawn(async move {
                loop {
                    let Ok((s, _)) = listener.accept().await else { return };
                    tokio::spawn(handle(s, c.clone()));
                }
            });
            Server { ctl, addr }
        }

        pub fn url(&self) -> String {
            format!("http://{}/file", self.addr)
        }

        pub fn log(&self) -> Vec<Req> {
            self.ctl.lock().unwrap().log.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::server::Server;
    use super::*;
    use crate::catalog::{EntrySource, Licence};
    use crate::gguf::fixture;

    const FILE: &str = "fixture-model.gguf";

    fn body_of(spec: &fixture::Spec, tag: &str) -> Vec<u8> {
        // Unique per call: the tests run in parallel and each makes its own body.
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!("lorren-dl-fixture-{}-{tag}-{}.gguf", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        fixture::write(&p, spec).unwrap();
        let mut b = std::fs::read(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        b.resize(b.len().max(64 * 1024), 0);
        b
    }

    fn body() -> Vec<u8> {
        body_of(&fixture::Spec::qwen3("fixture"), "a")
    }

    fn sha(b: &[u8]) -> String {
        hex(&Sha256::digest(b))
    }

    fn fast() -> RetryPolicy {
        RetryPolicy { first: Duration::from_millis(20), cap: Duration::from_millis(100), stall: Duration::from_secs(3) }
    }

    async fn until(what: &str, mut f: impl FnMut() -> bool) {
        for _ in 0..2000 {
            if f() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("timed out waiting for {what}");
    }

    type Models = Arc<Mutex<Vec<(ModelInfo, bool)>>>;
    type Events = Arc<Mutex<Vec<DownloadView>>>;

    fn make(
        dir: &Path,
        store: &Arc<Mutex<Store>>,
        lib: &Arc<RwLock<Vec<ModelInfo>>>,
        models: &Models,
        events: &Events,
        tweak: impl FnOnce(&mut DownloadConfig),
    ) -> Arc<DownloadManager> {
        let mut cfg = DownloadConfig::new(dir.to_path_buf(), true, "fixture".into());
        cfg.policy = fast();
        cfg.free_bytes = Arc::new(|_| Ok(u64::MAX));
        tweak(&mut cfg);
        let (models, events) = (models.clone(), events.clone());
        DownloadManager::new(
            cfg,
            store.clone(),
            lib.clone(),
            Arc::new(move |v| events.lock().unwrap().push(v.clone())),
            Arc::new(move |m, none| models.lock().unwrap().push((m, none))),
        )
    }

    struct Rig {
        dir: PathBuf,
        store: Arc<Mutex<Store>>,
        lib: Arc<RwLock<Vec<ModelInfo>>>,
        srv: Server,
        mgr: Arc<DownloadManager>,
        models: Models,
        events: Events,
        body: Vec<u8>,
    }

    impl Rig {
        async fn new(tag: &str) -> Rig {
            Rig::with(tag, |_| {}).await
        }

        async fn with(tag: &str, tweak: impl FnOnce(&mut DownloadConfig)) -> Rig {
            let dir = std::env::temp_dir().join(format!("lorren-dl-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let body = body();
            let srv = Server::start(body.clone()).await;
            let store = Arc::new(Mutex::new(Store::open_in_memory().unwrap()));
            let lib = Arc::new(RwLock::new(Vec::new()));
            let models = Arc::new(Mutex::new(Vec::new()));
            let events = Arc::new(Mutex::new(Vec::new()));
            let mgr = make(&dir, &store, &lib, &models, &events, tweak);
            Rig { dir, store, lib, srv, mgr, models, events, body }
        }

        /// A manager over this rig's store, folder and library (a second one is a restart).
        fn manager(&self, tweak: impl FnOnce(&mut DownloadConfig)) -> Arc<DownloadManager> {
            make(&self.dir, &self.store, &self.lib, &self.models, &self.events, tweak)
        }

        fn entry(&self) -> CatalogEntry {
            CatalogEntry {
                id: "fixture".into(),
                name: "Fixture".into(),
                version: 1,
                description: "A fixture.".into(),
                architecture: "qwen3".into(),
                quantization: "Q4_K_M".into(),
                file: FILE.into(),
                size: self.body.len() as u64,
                sha256: sha(&self.body),
                url: self.srv.url(),
                licence: Licence { name: "MIT".into(), url: "https://example.com/licence".into() },
                source: EntrySource { repo: "x/y".into(), commit: "abc".into(), path: FILE.into() },
                system_prompt: None,
                min_app_version: "0.1.0".into(),
            }
        }

        fn view(&self) -> DownloadView {
            self.mgr.list().into_iter().find(|v| v.entry_id == "fixture").expect("a view for the entry")
        }

        async fn until_state(&self, state: &str) {
            until(state, || self.mgr.list().iter().any(|v| v.state == state)).await;
        }

        fn requests(&self) -> usize {
            self.srv.log().len()
        }

        fn end(&self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn backoff_doubles_to_the_cap() {
        let got: Vec<u64> = (0..8).map(|a| backoff(&RetryPolicy::DEFAULT, a).as_secs()).collect();
        assert_eq!(got, [1, 2, 4, 8, 16, 30, 30, 30]);
        assert_eq!(backoff(&RetryPolicy::DEFAULT, u32::MAX), Duration::from_secs(30));
    }

    #[test]
    fn room_is_checked_with_a_gigabyte_to_spare() {
        assert!(room_check(5 * MARGIN_BYTES, 4 * MARGIN_BYTES).is_ok());
        assert_eq!(room_check(MARGIN_BYTES, MARGIN_BYTES).unwrap_err(), "needs 1.0 GB with 1 GB to spare; 1.0 GB free");
    }

    #[test]
    fn free_bytes_reads_the_volume_of_a_folder() {
        assert!(free_bytes(&std::env::temp_dir()).unwrap() > 0);
        assert!(disk_room(&std::env::temp_dir().join("lorren-no-such-folder-yet")).free_bytes > 0);
    }

    #[tokio::test]
    async fn a_download_completes_verifies_and_joins_the_library() {
        let r = Rig::new("complete").await;
        let v = r.mgr.start(&r.entry()).unwrap();
        assert_eq!((v.state.as_str(), v.total), ("downloading", r.body.len() as u64));
        r.until_state("done").await;
        let target = r.dir.join(FILE);
        assert_eq!(std::fs::read(&target).unwrap(), r.body);
        assert!(!r.dir.join(format!("{FILE}.part")).exists());
        let row = r.store.lock().unwrap().downloads().unwrap().remove(0);
        assert_eq!((row.state.as_str(), row.version), ("done", 1));
        let lib = r.lib.lock_free();
        assert_eq!(lib.len(), 1);
        assert_eq!(lib[0].id, format!("file:{}", target.to_string_lossy().to_lowercase()));
        let models = r.models.lock().unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].1, "the library had no runnable model before");
        assert_eq!(models[0].0.id, lib[0].id);
        let last = r.events.lock().unwrap().last().cloned().unwrap();
        assert_eq!((last.state.as_str(), last.done), ("done", r.body.len() as u64));
        r.end();
    }

    #[tokio::test]
    async fn a_dropped_connection_resumes_from_where_it_stopped() {
        let r = Rig::new("drop").await;
        r.srv.ctl.lock().unwrap().drop_after = Some(20_000);
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].range, None);
        assert_eq!(log[1].range.as_deref(), Some("bytes=20000-"));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn a_500_is_retried_after_the_backoff() {
        let r = Rig::with("500", |c| c.policy = RetryPolicy { first: Duration::from_millis(250), cap: Duration::from_secs(1), stall: Duration::from_secs(3) }).await;
        r.srv.ctl.lock().unwrap().fail_500_once = true;
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 2);
        assert!(log[1].at.duration_since(log[0].at) >= Duration::from_millis(250));
        assert!(r.events.lock().unwrap().iter().any(|v| v.retry_in_ms == Some(250)), "the wait is shown");
        r.end();
    }

    #[tokio::test]
    async fn a_404_fails_without_a_retry() {
        let r = Rig::new("404").await;
        r.srv.ctl.lock().unwrap().status = Some(404);
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("failed").await;
        tokio::time::sleep(Duration::from_millis(200)).await; // ten backoffs: a retry would have come
        assert_eq!(r.requests(), 1);
        assert_eq!(r.view().error.as_deref(), Some("404 from the server"));
        assert_eq!(r.store.lock().unwrap().downloads().unwrap()[0].state, "failed");
        r.end();
    }

    #[tokio::test]
    async fn a_server_that_ignores_range_restarts_from_zero_and_still_verifies() {
        let r = Rig::new("norange").await;
        {
            let mut c = r.srv.ctl.lock().unwrap();
            c.drop_after = Some(20_000);
            c.ignore_range = true;
        }
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[1].range.as_deref(), Some("bytes=20000-"), "it asked, and was answered with the whole file");
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn a_checksum_mismatch_deletes_the_file_and_says_so() {
        let r = Rig::new("mismatch").await;
        let mut e = r.entry();
        e.sha256 = "0".repeat(64);
        r.mgr.start(&e).unwrap();
        r.until_state("failed").await;
        assert_eq!(r.view().error.as_deref(), Some("the file did not match its checksum and was deleted"));
        assert!(!r.dir.join(format!("{FILE}.part")).exists());
        assert!(!r.dir.join(FILE).exists());
        assert!(r.lib.lock_free().is_empty());
        assert!(r.models.lock().unwrap().is_empty());
        r.end();
    }

    #[tokio::test]
    async fn pause_stops_the_requests_and_resume_finishes() {
        let stall = Duration::from_millis(400);
        let r = Rig::with("pause", |c| c.policy = RetryPolicy { first: Duration::from_millis(20), cap: Duration::from_millis(100), stall }).await;
        r.srv.ctl.lock().unwrap().hold_after = Some(16_384);
        r.mgr.start(&r.entry()).unwrap();
        until("16 KiB", || r.view().done >= 16_384).await;
        r.mgr.pause("fixture").unwrap();
        assert_eq!(r.view().state, "paused");
        assert_eq!(r.store.lock().unwrap().downloads().unwrap()[0].state, "paused");
        tokio::time::sleep(stall * 3).await;
        assert_eq!(r.requests(), 1, "no request while paused");
        assert_eq!(r.view().state, "paused");
        r.mgr.resume("fixture").unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[1].range.as_deref(), Some("bytes=16384-"));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn a_pause_during_a_write_stays_paused_and_resume_finishes() {
        type Slot = Arc<Mutex<Option<Arc<DownloadManager>>>>;
        struct PauseOnWrite {
            inner: Box<dyn Write + Send>,
            slot: Slot,
            done: bool,
        }
        impl Write for PauseOnWrite {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                let n = self.inner.write(b)?;
                if !self.done {
                    self.done = true;
                    // The pause lands while this chunk is being written: the worker has already taken the chunk.
                    let m = self.slot.lock().unwrap().clone();
                    if let Some(m) = m {
                        m.pause("fixture").unwrap();
                    }
                }
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.inner.flush()
            }
        }
        let slot: Slot = Arc::new(Mutex::new(None));
        let s2 = slot.clone();
        let r = Rig::with("pausewrite", move |c| {
            c.open_part = Arc::new(move |p, off| Ok(Box::new(PauseOnWrite { inner: open_part(p, off)?, slot: s2.clone(), done: false })));
        })
        .await;
        *slot.lock().unwrap() = Some(r.mgr.clone());
        r.mgr.start(&r.entry()).unwrap();
        until("the pause", || r.store.lock().unwrap().downloads().unwrap().first().is_some_and(|d| d.state == "paused")).await;
        tokio::time::sleep(Duration::from_millis(300)).await; // the worker has stopped by now
        assert_eq!(r.view().state, "paused");
        assert_eq!(r.store.lock().unwrap().downloads().unwrap()[0].state, "paused");
        {
            let ev = r.events.lock().unwrap();
            let at = ev.iter().position(|v| v.state == "paused").expect("a paused event");
            assert!(ev[at..].iter().all(|v| v.state == "paused"), "after paused: {:?}", ev[at..].iter().map(|v| v.state.clone()).collect::<Vec<_>>());
        }
        // The slot's writer pauses once per open; a resumed worker opens a fresh one, so release the manager first.
        *slot.lock().unwrap() = None;
        r.mgr.resume("fixture").unwrap();
        r.until_state("done").await;
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn cancel_removes_the_part_and_the_row() {
        let r = Rig::new("cancel").await;
        r.srv.ctl.lock().unwrap().hold_after = Some(16_384);
        r.mgr.start(&r.entry()).unwrap();
        until("16 KiB", || r.view().done >= 16_384).await;
        assert!(r.dir.join(format!("{FILE}.part")).exists());
        r.mgr.cancel("fixture").unwrap();
        assert!(r.store.lock().unwrap().downloads().unwrap().is_empty());
        assert!(r.mgr.list().is_empty());
        until("the part to go", || !r.dir.join(format!("{FILE}.part")).exists()).await;
        assert!(r.mgr.cancel("fixture").is_err());
        r.end();
    }

    #[tokio::test]
    async fn a_stall_is_a_dropped_connection() {
        let r = Rig::with("stall", |c| c.policy = RetryPolicy { first: Duration::from_millis(20), cap: Duration::from_millis(100), stall: Duration::from_millis(300) }).await;
        r.srv.ctl.lock().unwrap().hold_after = Some(16_384);
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[1].range.as_deref(), Some("bytes=16384-"));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn more_bytes_than_the_size_fail() {
        let r = Rig::new("more").await;
        r.srv.ctl.lock().unwrap().extra = 100;
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("failed").await;
        assert_eq!(r.view().error.as_deref(), Some("the server sent more than the catalog's size"));
        assert!(!r.dir.join(FILE).exists());
        assert_eq!(r.requests(), 1);
        r.end();
    }

    #[tokio::test]
    async fn not_enough_room_is_refused_before_any_request() {
        let size = body().len() as u64;
        let r = Rig::with("room", move |c| c.free_bytes = Arc::new(move |_| Ok(size - 1 + MARGIN_BYTES))).await;
        let err = r.mgr.start(&r.entry()).unwrap_err();
        assert_eq!(err, room_check(size - 1 + MARGIN_BYTES, size).unwrap_err());
        assert!(err.starts_with("needs ") && err.contains("with 1 GB to spare"), "{err}");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(r.requests(), 0);
        assert!(r.mgr.list().is_empty(), "no row for a download that never started");
        r.end();
    }

    #[tokio::test]
    async fn a_restart_pauses_a_running_download_and_resume_rehashes_the_part() {
        let r = Rig::new("restart").await;
        // What a killed app leaves: 20,000 bytes in the `.part` and a row still marked downloading.
        std::fs::create_dir_all(&r.dir).unwrap();
        std::fs::write(r.dir.join(format!("{FILE}.part")), &r.body[..20_000]).unwrap();
        let e = r.entry();
        r.store
            .lock()
            .unwrap()
            .upsert_download(&DownloadRow {
                entry_id: e.id.clone(),
                name: e.name.clone(),
                version: 1,
                url: e.url.clone(),
                file: e.file.clone(),
                size: e.size,
                sha256: e.sha256.clone(),
                system_prompt: None,
                state: "downloading".into(),
                error: None,
                created_ms: 1,
                updated_ms: 1,
            })
            .unwrap();
        let second = r.manager(|_| {});
        second.reconcile_at_start();
        let v = second.list().remove(0);
        assert_eq!((v.state.as_str(), v.done), ("paused", 20_000));
        second.resume("fixture").unwrap();
        until("done", || second.list()[0].state == "done").await;
        let log = r.srv.log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].range.as_deref(), Some("bytes=20000-"));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body, "the digest still matched");
        r.end();
    }

    #[tokio::test]
    async fn a_part_with_no_row_is_deleted_at_start() {
        let r = Rig::new("orphan").await;
        std::fs::create_dir_all(&r.dir).unwrap();
        std::fs::write(r.dir.join("orphan.gguf.part"), b"abc").unwrap();
        std::fs::write(r.dir.join("keep.gguf"), b"abc").unwrap();
        r.mgr.reconcile_at_start();
        assert!(!r.dir.join("orphan.gguf.part").exists());
        assert!(r.dir.join("keep.gguf").exists());
        r.end();
    }

    #[tokio::test]
    async fn an_entry_with_a_system_prompt_gets_a_modelfile_the_library_reads() {
        let r = Rig::new("modelfile").await;
        let mut e = r.entry();
        e.system_prompt = Some("Be brief.".into());
        r.mgr.start(&e).unwrap();
        r.until_state("done").await;
        let mf = std::fs::read_to_string(r.dir.join("fixture-model.modelfile")).unwrap();
        assert_eq!(mf, format!("FROM {FILE}\nSYSTEM \"\"\"Be brief.\"\"\"\n"));
        let lib = r.lib.lock_free();
        assert_eq!(lib[0].system_prompt.as_deref(), Some("Be brief."));
        r.end();
    }

    #[tokio::test]
    async fn an_update_downloads_beside_the_old_version() {
        let r = Rig::new("update").await;
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let v2 = body_of(&fixture::Spec::llama("fixture two"), "b");
        r.srv.ctl.lock().unwrap().body = v2.clone();
        let mut e = r.entry();
        e.version = 2;
        e.size = v2.len() as u64;
        e.sha256 = sha(&v2);
        r.mgr.start(&e).unwrap();
        until("version 2", || r.view().version == 2 && r.view().state == "done").await;
        assert_eq!(std::fs::read(r.dir.join("fixture-model.v2.gguf")).unwrap(), v2);
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body, "the old file is untouched");
        assert_eq!(r.lib.lock_free().len(), 2);
        let row = r.store.lock().unwrap().downloads().unwrap().remove(0);
        assert_eq!((row.version, row.file.as_str()), (2, "fixture-model.v2.gguf"));
        assert!(!r.models.lock().unwrap()[1].1, "the library had a runnable model by then");
        // The same version again is refused, whichever file it would be saved as.
        assert!(r.mgr.start(&e).unwrap_err().contains("is already in"));
        r.end();
    }

    #[tokio::test]
    async fn a_resume_asks_the_catalog_url_again_not_the_redirect_target() {
        let r = Rig::new("redirect").await;
        {
            let mut c = r.srv.ctl.lock().unwrap();
            c.redirect = true;
            c.drop_after = Some(20_000);
        }
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("done").await;
        let log = r.srv.log();
        let paths: Vec<&str> = log.iter().map(|q| q.path.as_str()).collect();
        assert_eq!(paths, ["/file", "/cdn?once=1", "/file", "/cdn?once=2"], "every request after the first starts at /file again");
        assert_eq!(log[3].range.as_deref(), Some("bytes=20000-"));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), r.body);
        r.end();
    }

    #[tokio::test]
    async fn starting_a_running_download_again_starts_nothing() {
        let r = Rig::with("twice", |c| c.policy.stall = Duration::from_secs(60)).await;
        r.srv.ctl.lock().unwrap().hold_after = Some(16_384);
        let a = r.mgr.start(&r.entry()).unwrap();
        let b = r.mgr.start(&r.entry()).unwrap();
        assert_eq!((a.state.as_str(), b.state.as_str()), ("downloading", "downloading"));
        until("16 KiB", || r.view().done >= 16_384).await;
        let c = r.mgr.start(&r.entry()).unwrap();
        assert_eq!(c.entry_id, "fixture");
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(r.requests(), 1, "one worker, one request stream");
        assert_eq!(r.mgr.list().len(), 1);
        assert_eq!(r.store.lock().unwrap().downloads().unwrap().len(), 1);
        r.mgr.cancel("fixture").unwrap();
        r.end();
    }

    #[tokio::test]
    async fn an_existing_file_is_never_overwritten() {
        let r = Rig::new("exists").await;
        std::fs::create_dir_all(&r.dir).unwrap();
        std::fs::write(r.dir.join(FILE), b"mine").unwrap();
        let err = r.mgr.start(&r.entry()).unwrap_err();
        assert_eq!(err, format!("{FILE} is already in {}", r.dir.display()));
        assert_eq!(std::fs::read(r.dir.join(FILE)).unwrap(), b"mine");
        assert_eq!(r.requests(), 0);
        assert!(r.mgr.list().is_empty());
        r.end();
    }

    #[tokio::test]
    async fn a_full_disk_fails_once_and_keeps_the_part() {
        struct Limited {
            inner: Box<dyn Write + Send>,
            left: usize,
        }
        impl Write for Limited {
            fn write(&mut self, b: &[u8]) -> io::Result<usize> {
                if self.left == 0 {
                    return Err(io::Error::from_raw_os_error(112));
                }
                let n = self.inner.write(&b[..b.len().min(self.left)])?;
                self.left -= n;
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                self.inner.flush()
            }
        }
        let r = Rig::with("full", |c| c.open_part = Arc::new(|p, off| Ok(Box::new(Limited { inner: open_part(p, off)?, left: 16_384 })))).await;
        r.mgr.start(&r.entry()).unwrap();
        r.until_state("failed").await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(r.view().error.as_deref(), Some("the disk is full"));
        assert_eq!(r.requests(), 1, "no retry loop on the network");
        assert_eq!(file_len(&r.dir.join(format!("{FILE}.part"))), 16_384);
        assert_eq!(r.store.lock().unwrap().downloads().unwrap()[0].state, "failed");
        r.end();
    }

    trait LockFree {
        fn lock_free(&self) -> Vec<ModelInfo>;
    }
    impl LockFree for Arc<RwLock<Vec<ModelInfo>>> {
        fn lock_free(&self) -> Vec<ModelInfo> {
            self.read().unwrap().clone()
        }
    }
}
