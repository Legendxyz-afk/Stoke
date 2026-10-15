//! Starts, watches and restarts the engine (spec §3.1). One engine process at a time; a model
//! switch is a restart. Status changes go out on a watch channel for the UI.

use crate::engine::job::Job;
use crate::engine::memory;
use crate::gguf;
use crate::monitor::Monitor;
use serde::Serialize;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, watch, Mutex};
use ts_rs::TS;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    Stopped,
    Starting,
    Serving,
    Restarting,
    Failed,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct EngineStatus {
    pub state: EngineState,
    pub model_id: Option<String>,
    pub model_path: Option<String>,
    pub port: Option<u16>,
    pub pid: Option<u32>,
    pub ctx: u32,
    pub seqs: u32,
    pub threads: Option<u32>,
    pub kv: String,
    #[ts(type = "number | null")]
    pub load_ms: Option<u64>,
    pub restarts: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub exe: PathBuf,
    pub ctx: u32,
    pub seqs: u32,
    pub threads: Option<u32>,
    pub kv: String,
    pub load_timeout: Duration,
    pub health_interval: Duration,
    pub health_timeout: Duration,
    pub health_misses: u32,
    pub backoff: Vec<Duration>,
    pub failure_window: Duration,
    pub max_failures: usize,
    pub env: Vec<(String, String)>,
    pub min_free_bytes: u64,
    pub check_memory: bool,
}

impl EngineConfig {
    pub fn defaults(exe: PathBuf) -> Self {
        EngineConfig {
            exe,
            ctx: gguf::ENGINE_MAX_CTX,
            seqs: gguf::ENGINE_SEQS,
            threads: None,
            kv: "q8".into(),
            load_timeout: Duration::from_secs(120),
            health_interval: Duration::from_secs(5),
            // The real engine answers /health only between scheduler steps. A prefill step is 256 tokens
            // and takes ~3-4 s at the app's history sizes, and a new connection can wait two steps.
            health_timeout: Duration::from_secs(30),
            health_misses: 3,
            backoff: vec![Duration::from_secs(1), Duration::from_secs(5), Duration::from_secs(30)],
            failure_window: Duration::from_secs(600),
            max_failures: 3,
            env: vec![],
            min_free_bytes: 1 << 30,
            check_memory: true,
        }
    }

    fn args(&self, model: &Path) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "serve".into(),
            model.to_string_lossy().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "0".into(),
            "--ctx".into(),
            self.ctx.to_string(),
            "--seqs".into(),
            self.seqs.to_string(),
            "--kv".into(),
            self.kv.clone(),
        ];
        if let Some(t) = self.threads {
            a.push("--threads".into());
            a.push(t.to_string());
        }
        a
    }
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub id: String,
    pub path: PathBuf,
    pub file_bytes: u64,
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    #[error("not enough memory to load {model}: it needs about {need_mb} MB and {free_mb} MB is free (1 GB is kept spare)")]
    NotEnoughMemory { model: String, need_mb: u64, free_mb: u64 },
    #[error("the engine stopped while loading: {0}")]
    LoadFailed(String),
    #[error("the engine did not start within {0} s")]
    LoadTimeout(u64),
    #[error("the engine has failed {0} times in 10 minutes; see Logs")]
    GaveUp(usize),
    #[error("could not start the engine: {0}")]
    Spawn(String),
}

pub fn parse_banner(line: &str) -> Option<u16> {
    let rest = line.split("listening on ").nth(1)?;
    rest.split_whitespace().next()?.rsplit(':').next()?.parse().ok()
}

/// How long a failed start waits for the thread reading the engine's stderr to reach the end of the pipe.
/// That takes microseconds once the process is gone; the cap is for a pipe something else keeps open.
const STDERR_DRAIN_CAP: Duration = Duration::from_secs(2);

/// Calls `f` with each line of `src` until it ends or fails. The decoding is lossy on purpose: `BufRead::lines`
/// stops at the first line that is not UTF-8, and the real engine writes the model path (in the ANSI code
/// page) into the very error lines that say why it stopped.
fn for_each_line(src: impl Read, mut f: impl FnMut(String)) {
    let mut reader = BufReader::new(src);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => return,
            Ok(_) => {
                while matches!(buf.last(), Some(b'\n' | b'\r')) {
                    buf.pop();
                }
                f(String::from_utf8_lossy(&buf).into_owned());
            }
        }
    }
}

/// Reads the engine's stderr on its own thread: every line goes to the log and into `tail` (the last 20).
/// Before the banner the engine is still loading, so a stderr line is the reason it is about to stop (the
/// real loader's errors carry no "fatal" prefix); after the banner it is ordinary output. The receiver
/// resolves once the pipe has reached its end and the last line is in `tail`.
fn spawn_stderr_reader(
    stderr: impl Read + Send + 'static,
    banner_seen: Arc<AtomicBool>,
    monitor: Arc<Monitor>,
    tail: Arc<StdMutex<Vec<String>>>,
) -> oneshot::Receiver<()> {
    let (done_tx, done_rx) = oneshot::channel();
    std::thread::spawn(move || {
        for_each_line(stderr, |line| {
            let level = if !banner_seen.load(Ordering::SeqCst) && !line.contains("warning") { "error" } else { "info" };
            monitor.log("engine", level, line.clone());
            let mut t = tail.lock().unwrap();
            t.push(line);
            if t.len() > 20 {
                t.remove(0);
            }
        });
        let _ = done_tx.send(());
    });
    done_rx
}

/// Waits up to `cap` for a reader from `spawn_stderr_reader` to finish. False when it did not.
async fn drained(done: oneshot::Receiver<()>, cap: Duration) -> bool {
    tokio::time::timeout(cap, done).await.is_ok()
}

/// The error for an engine that never reached its banner: it stopped (`exited`) or ran out of `load_timeout`.
/// A stopped engine's last non-empty stderr line is the reason, and it is read only after the stderr reader has
/// reached the end of the pipe (at most `drain_cap`), not after a guessed number of milliseconds: it must be in
/// `tail` however the threads happen to be scheduled (R35).
async fn load_error(
    exited: bool,
    load_timeout: Duration,
    stderr_done: oneshot::Receiver<()>,
    drain_cap: Duration,
    tail: &StdMutex<Vec<String>>,
    monitor: &Monitor,
) -> EngineError {
    if !drained(stderr_done, drain_cap).await {
        monitor.log(
            "app",
            "warn",
            format!("the engine's stderr did not close within {} s; its last lines may be missing", drain_cap.as_secs_f64()),
        );
    }
    if !exited {
        return EngineError::LoadTimeout(load_timeout.as_secs());
    }
    let reason = tail.lock().unwrap().iter().rev().find(|l| !l.trim().is_empty()).cloned();
    EngineError::LoadFailed(reason.unwrap_or_else(|| "no output".into()))
}

/// Puts `child` in a job object that kills it when the app's last handle closes (spec 3.1). When the job
/// cannot be made, or will not take the process, the engine still runs, but one `warn` line says it may
/// outlive the app (R35).
fn attach_job(monitor: &Monitor, child: &Child, make: impl FnOnce() -> std::io::Result<Job>) -> Option<Job> {
    match make().and_then(|job| job.assign(child).map(|()| job)) {
        Ok(job) => Some(job),
        Err(e) => {
            monitor.log("app", "warn", format!("could not put the engine in a job object ({e}); it may outlive the app if the app crashes"));
            None
        }
    }
}

struct Running {
    child: Child,
    _job: Option<Job>,
    model: ModelRef,
    port: u16,
    stderr_tail: Arc<StdMutex<Vec<String>>>,
}

struct Inner {
    running: Option<Running>,
    generation: u64,
    failures: Vec<Instant>,
    restarts: u32,
}

pub struct Supervisor {
    cfg: EngineConfig,
    monitor: Arc<Monitor>,
    tx: watch::Sender<EngineStatus>,
    inner: Mutex<Inner>,
    http: reqwest::Client,
}

impl Supervisor {
    pub fn new(cfg: EngineConfig, monitor: Arc<Monitor>) -> Arc<Supervisor> {
        let status = EngineStatus {
            state: EngineState::Stopped,
            model_id: None,
            model_path: None,
            port: None,
            pid: None,
            ctx: cfg.ctx,
            seqs: cfg.seqs,
            threads: cfg.threads,
            kv: cfg.kv.clone(),
            load_ms: None,
            restarts: 0,
            last_error: None,
        };
        let (tx, _rx) = watch::channel(status);
        Arc::new(Supervisor {
            cfg,
            monitor,
            tx,
            inner: Mutex::new(Inner { running: None, generation: 0, failures: vec![], restarts: 0 }),
            // The engine closes idle keep-alive connections after 60 s; reqwest's default would reuse a dead one.
            http: reqwest::Client::builder()
                .no_proxy()
                .pool_idle_timeout(Duration::from_secs(30))
                .build()
                .expect("http client"),
        })
    }

    pub fn status(&self) -> EngineStatus {
        self.tx.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<EngineStatus> {
        self.tx.subscribe()
    }

    fn set(&self, f: impl FnOnce(&mut EngineStatus)) {
        self.tx.send_modify(f);
    }

    /// Makes sure the engine is serving `model`, starting or switching it if needed.
    /// Waits for any start already in progress. Returns the port.
    pub async fn ensure_model(self: &Arc<Self>, model: &ModelRef) -> Result<u16, EngineError> {
        let mut inner = self.inner.lock().await;
        let status = self.status();
        if let Some(r) = inner.running.as_mut() {
            if r.model.id == model.id && status.state == EngineState::Serving && matches!(r.child.try_wait(), Ok(None)) {
                return Ok(r.port);
            }
        }
        if status.state == EngineState::Failed {
            inner.failures.clear(); // an explicit request retries after giving up
        }
        self.stop_locked(&mut inner);
        self.start_locked(&mut inner, model.clone(), false).await
    }

    pub async fn stop(&self) {
        let mut inner = self.inner.lock().await;
        self.stop_locked(&mut inner);
    }

    fn stop_locked(&self, inner: &mut Inner) {
        inner.generation += 1; // retires any watcher of the old process
        if let Some(mut r) = inner.running.take() {
            let _ = r.child.kill();
            let _ = r.child.wait();
            self.monitor.log("app", "info", format!("stopped the engine ({})", r.model.id));
        }
        self.set(|s| {
            s.state = EngineState::Stopped;
            s.port = None;
            s.pid = None;
        });
    }

    fn memory_check(&self, model: &ModelRef) -> Result<(), EngineError> {
        if !self.cfg.check_memory {
            return Ok(());
        }
        // The same number the Models tab shows as the memory to run; `min_free_bytes` is the spare on top.
        let need = gguf::run_bytes(model.file_bytes, model.kv_bytes_per_token, self.cfg.ctx, self.cfg.seqs);
        let free = memory::available_physical_bytes();
        // Saturating: `need` is u64::MAX for a hostile GGUF header, and adding the spare to that must not wrap
        // (release) or panic (debug) into "fits".
        if free < need.saturating_add(self.cfg.min_free_bytes) {
            return Err(EngineError::NotEnoughMemory { model: model.id.clone(), need_mb: need >> 20, free_mb: free >> 20 });
        }
        Ok(())
    }

    async fn start_locked(self: &Arc<Self>, inner: &mut Inner, model: ModelRef, restart: bool) -> Result<u16, EngineError> {
        if let Err(e) = self.memory_check(&model) {
            self.set(|s| {
                s.state = EngineState::Failed;
                s.last_error = Some(e.to_string());
            });
            self.monitor.log("app", "error", e.to_string());
            return Err(e);
        }
        inner.generation += 1;
        let gen = inner.generation;
        let path = model.path.to_string_lossy().to_string();
        self.set(|s| {
            s.state = if restart { EngineState::Restarting } else { EngineState::Starting };
            s.model_id = Some(model.id.clone());
            s.model_path = Some(path.clone());
            s.port = None;
            s.pid = None;
            s.last_error = None;
        });
        self.monitor.log("app", "info", format!("starting the engine with {path}"));
        let t0 = Instant::now();
        let mut cmd = Command::new(&self.cfg.exe);
        cmd.args(self.cfg.args(&model.path))
            .envs(self.cfg.env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                // The status above says Starting/Restarting; returning without updating it would leave the UI
                // waiting for an engine that was never started (R35).
                let err = EngineError::Spawn(e.to_string());
                self.set(|s| {
                    s.state = EngineState::Failed;
                    s.port = None;
                    s.pid = None;
                    s.last_error = Some(err.to_string());
                });
                self.monitor.log("app", "error", err.to_string());
                return Err(err);
            }
        };
        let job = attach_job(&self.monitor, &child, Job::kill_on_close);
        let (banner_tx, banner_rx) = mpsc::channel::<u16>();
        // Set by the stdout reader when the banner arrives; the stderr reader below reads it.
        let banner_seen = Arc::new(AtomicBool::new(false));
        let stdout = child.stdout.take().expect("piped stdout");
        let (mon, seen_w) = (self.monitor.clone(), banner_seen.clone());
        std::thread::spawn(move || {
            for_each_line(stdout, |line| {
                if let Some(p) = parse_banner(&line) {
                    seen_w.store(true, Ordering::SeqCst);
                    let _ = banner_tx.send(p);
                }
                mon.log("engine", "info", line);
            });
        });
        let tail = Arc::new(StdMutex::new(Vec::<String>::new()));
        let stderr = child.stderr.take().expect("piped stderr");
        let stderr_done = spawn_stderr_reader(stderr, banner_seen, self.monitor.clone(), tail.clone());
        let timeout = self.cfg.load_timeout;
        // Ok(port) | Err(true) = exited before the banner | Err(false) = timed out
        let waited = tokio::task::spawn_blocking(move || match banner_rx.recv_timeout(timeout) {
            Ok(p) => Ok(p),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(false),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(true),
        })
        .await
        .unwrap_or(Err(true));
        match waited {
            Ok(port) => {
                let load_ms = t0.elapsed().as_millis() as u64;
                let pid = child.id();
                let restarts = inner.restarts;
                inner.running = Some(Running { child, _job: job, model: model.clone(), port, stderr_tail: tail });
                self.set(|s| {
                    s.state = EngineState::Serving;
                    s.port = Some(port);
                    s.pid = Some(pid);
                    s.load_ms = Some(load_ms);
                    s.restarts = restarts;
                });
                self.monitor.log(
                    "app",
                    "info",
                    format!("engine serving {} on port {port} after {:.2} s", model.id, load_ms as f64 / 1000.0),
                );
                self.spawn_watch(gen);
                Ok(port)
            }
            Err(exited) => {
                let _ = child.kill();
                let _ = child.wait();
                let err = load_error(exited, timeout, stderr_done, STDERR_DRAIN_CAP, &tail, &self.monitor).await;
                self.set(|s| {
                    s.state = EngineState::Failed;
                    s.port = None;
                    s.pid = None;
                    s.last_error = Some(err.to_string());
                });
                self.monitor.log("app", "error", err.to_string());
                Err(err)
            }
        }
    }

    fn spawn_watch(self: &Arc<Self>, gen: u64) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut misses = 0u32;
            loop {
                tokio::time::sleep(me.cfg.health_interval).await;
                let (port, exited) = {
                    let mut inner = me.inner.lock().await;
                    if inner.generation != gen {
                        return;
                    }
                    let Some(r) = inner.running.as_mut() else { return };
                    (r.port, !matches!(r.child.try_wait(), Ok(None)))
                };
                if !exited && me.healthy(port).await {
                    misses = 0;
                    continue;
                }
                if !exited {
                    misses += 1;
                    if misses < me.cfg.health_misses {
                        continue;
                    }
                }
                let why = if exited { "the engine exited" } else { "the engine stopped answering /health" };
                me.recover(gen, why).await;
                return;
            }
        });
    }

    async fn healthy(&self, port: u16) -> bool {
        matches!(
            self.http.get(format!("http://127.0.0.1:{port}/health")).timeout(self.cfg.health_timeout).send().await,
            Ok(r) if r.status().is_success()
        )
    }

    /// A crash or a hang: restart after a backoff, or give up after `max_failures` in `failure_window`.
    async fn recover(self: &Arc<Self>, gen: u64, why: &str) {
        let (model, delay) = {
            let mut inner = self.inner.lock().await;
            if inner.generation != gen {
                return;
            }
            let Some(mut r) = inner.running.take() else { return };
            let _ = r.child.kill();
            let _ = r.child.wait();
            let last = r.stderr_tail.lock().unwrap().last().cloned();
            self.monitor.log(
                "app",
                "warn",
                format!("{why}{}", last.map(|l| format!(" (last engine line: {l})")).unwrap_or_default()),
            );
            let now = Instant::now();
            let window = self.cfg.failure_window;
            inner.failures.retain(|t| now.duration_since(*t) < window);
            inner.failures.push(now);
            if inner.failures.len() >= self.cfg.max_failures {
                let e = EngineError::GaveUp(inner.failures.len());
                self.set(|s| {
                    s.state = EngineState::Failed;
                    s.port = None;
                    s.pid = None;
                    s.last_error = Some(e.to_string());
                });
                self.monitor.log("app", "error", e.to_string());
                return;
            }
            inner.restarts += 1;
            let delay = self.cfg.backoff.get(inner.failures.len() - 1).copied().unwrap_or(Duration::from_secs(30));
            self.set(|s| {
                s.state = EngineState::Restarting;
                s.port = None;
                s.pid = None;
                s.last_error = Some(why.to_string());
            });
            (r.model, delay)
        };
        tokio::time::sleep(delay).await;
        let mut inner = self.inner.lock().await;
        if inner.generation != gen || inner.running.is_some() {
            return; // someone started an engine meanwhile
        }
        let _ = self.start_locked(&mut inner, model, true).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// What `for_each_line` makes of `bytes`.
    fn lines(bytes: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        for_each_line(Cursor::new(bytes.to_vec()), |l| out.push(l));
        out
    }

    #[test]
    fn lines_split_on_lf_and_crlf_and_the_last_one_needs_no_newline() {
        assert_eq!(lines(b"one\ntwo\r\n\nthree"), ["one", "two", "", "three"]);
        assert!(lines(b"").is_empty());
    }

    /// `BufRead::lines` ends the loop at the first line that is not UTF-8. The real engine writes the model
    /// path into its error lines in the ANSI code page (0xE9 is "e acute" there), so that line must survive
    /// and the lines after it must still arrive.
    #[test]
    fn a_line_that_is_not_utf8_is_kept_and_the_reading_goes_on() {
        let got = lines(b"model: 'C:/Jos\xE9/x.gguf' -- bad\nnext\n");
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got[0].starts_with("model: 'C:/Jos") && got[0].ends_with("/x.gguf' -- bad"), "{got:?}");
        assert_eq!(got[1], "next");
    }

    /// A pipe whose first read takes `delay`, then yields its text and ends: a stderr reader that is
    /// scheduled late.
    struct Slow {
        delay: Option<Duration>,
        rest: Cursor<Vec<u8>>,
    }

    impl Read for Slow {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if let Some(d) = self.delay.take() {
                std::thread::sleep(d);
            }
            self.rest.read(buf)
        }
    }

    /// A pipe that does not end until its sender is dropped: a grandchild holding stderr open.
    struct Stuck(mpsc::Receiver<()>);

    impl Read for Stuck {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            let _ = self.0.recv(); // blocks until the sender is dropped
            Ok(0)
        }
    }

    fn reader_state() -> (Arc<AtomicBool>, Arc<Monitor>, Arc<StdMutex<Vec<String>>>) {
        (Arc::new(AtomicBool::new(false)), Monitor::new(None), Arc::new(StdMutex::new(Vec::new())))
    }

    /// R35: a stderr reader that is late is waited for, whatever the scheduler does. The old fixed 100 ms
    /// sleep would have read an empty tail here.
    #[tokio::test]
    async fn the_drain_waits_for_a_stderr_reader_that_is_late() {
        let (seen, mon, tail) = reader_state();
        let pipe = Slow { delay: Some(Duration::from_millis(400)), rest: Cursor::new(b"stoke serve: model: 'x' -- bad\n".to_vec()) };
        let done = spawn_stderr_reader(pipe, seen, mon.clone(), tail.clone());
        assert!(drained(done, Duration::from_secs(5)).await);
        assert_eq!(*tail.lock().unwrap(), ["stoke serve: model: 'x' -- bad"]);
        let logged = mon.logs_after(0);
        assert_eq!(logged.len(), 1);
        assert_eq!((logged[0].source.as_str(), logged[0].level.as_str()), ("engine", "error"));
    }

    /// ... and not for ever: a pipe that stays open must not hang the start.
    #[tokio::test]
    async fn the_drain_gives_up_at_its_cap_when_the_pipe_never_ends() {
        let (open, held) = mpsc::channel::<()>();
        let (seen, mon, tail) = reader_state();
        let done = spawn_stderr_reader(Stuck(held), seen, mon, tail);
        let t0 = Instant::now();
        assert!(!drained(done, Duration::from_millis(200)).await);
        let took = t0.elapsed();
        assert!(took >= Duration::from_millis(200) && took < Duration::from_secs(5), "{took:?}");
        drop(open); // lets the reader thread end
    }

    /// R35: the reason of a failed start is read only after the stderr reader has finished -- also when the
    /// reader is late. (Reading it after a fixed 100 ms would have found "no output" here.)
    #[tokio::test]
    async fn a_failed_start_names_the_last_stderr_line_even_when_the_reader_is_late() {
        let (seen, mon, tail) = reader_state();
        let pipe = Slow {
            delay: Some(Duration::from_millis(400)),
            rest: Cursor::new(b"stoke serve: warning -- no chat template\nstoke serve: model: 'x' -- bad\n\n".to_vec()),
        };
        let done = spawn_stderr_reader(pipe, seen, mon.clone(), tail.clone());
        let err = load_error(true, Duration::from_secs(120), done, Duration::from_secs(5), &tail, &mon).await;
        assert!(matches!(&err, EngineError::LoadFailed(m) if m == "stoke serve: model: 'x' -- bad"), "{err}");
    }

    #[tokio::test]
    async fn a_failed_start_does_not_wait_for_ever_for_a_pipe_that_stays_open() {
        let (open, held) = mpsc::channel::<()>();
        let (seen, mon, tail) = reader_state();
        let done = spawn_stderr_reader(Stuck(held), seen, mon.clone(), tail.clone());
        let t0 = Instant::now();
        let err = load_error(true, Duration::from_secs(120), done, Duration::from_millis(200), &tail, &mon).await;
        assert!(matches!(&err, EngineError::LoadFailed(m) if m == "no output"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        let w = warnings(&mon);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].1.contains("stderr did not close"), "{}", w[0].1);
        drop(open); // lets the reader thread end
    }

    #[tokio::test]
    async fn a_start_that_timed_out_is_a_timeout_not_a_stderr_line() {
        let (seen, mon, tail) = reader_state();
        let done = spawn_stderr_reader(Cursor::new(b"stoke serve: loading...\n".to_vec()), seen, mon.clone(), tail.clone());
        let err = load_error(false, Duration::from_secs(120), done, Duration::from_secs(5), &tail, &mon).await;
        assert!(matches!(err, EngineError::LoadTimeout(120)), "{err}");
    }

    #[tokio::test]
    async fn before_the_banner_a_stderr_line_is_an_error_unless_it_is_a_warning_and_after_it_is_not() {
        let (seen, mon, tail) = reader_state();
        let text = b"stoke serve: warning -- this model has no chat template\nstoke serve: model: 'x' -- bad\n".to_vec();
        drained(spawn_stderr_reader(Cursor::new(text), seen.clone(), mon.clone(), tail), Duration::from_secs(5)).await;
        let levels: Vec<String> = mon.logs_after(0).into_iter().map(|l| l.level).collect();
        assert_eq!(levels, ["info", "error"]);
        // Once the banner has been seen the same kind of line is ordinary output.
        seen.store(true, Ordering::SeqCst);
        let (_, mon2, tail2) = reader_state();
        drained(spawn_stderr_reader(Cursor::new(b"stoke serve: model: 'x' -- bad\n".to_vec()), seen, mon2.clone(), tail2), Duration::from_secs(5)).await;
        assert_eq!(mon2.logs_after(0)[0].level, "info");
    }

    #[tokio::test]
    async fn the_tail_keeps_the_last_twenty_lines() {
        let (seen, mon, tail) = reader_state();
        let text: String = (1..=30).map(|i| format!("line {i}\n")).collect();
        drained(spawn_stderr_reader(Cursor::new(text.into_bytes()), seen, mon, tail.clone()), Duration::from_secs(5)).await;
        let t = tail.lock().unwrap();
        assert_eq!((t.len(), t.first().map(String::as_str), t.last().map(String::as_str)), (20, Some("line 11"), Some("line 30")));
    }

    /// A process that outlives the test body, standing in for the engine.
    fn sleeper() -> Child {
        Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    /// The `warn` lines in the Monitor, as (source, text).
    fn warnings(mon: &Monitor) -> Vec<(String, String)> {
        mon.logs_after(0).into_iter().filter(|l| l.level == "warn").map(|l| (l.source, l.text)).collect()
    }

    /// R35: a job that cannot be made is not silent.
    #[test]
    fn a_job_that_cannot_be_made_is_a_warning_not_silence() {
        let mon = Monitor::new(None);
        let mut child = sleeper();
        let job = attach_job(&mon, &child, || Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied")));
        let _ = child.kill();
        let _ = child.wait();
        assert!(job.is_none());
        let w = warnings(&mon);
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].0, "app");
        assert!(w[0].1.contains("access denied") && w[0].1.contains("may outlive the app"), "{}", w[0].1);
    }

    /// R35: nor is a process the job refuses (Windows will not assign a process that has already ended).
    #[test]
    fn a_process_the_job_refuses_is_a_warning_too() {
        let mon = Monitor::new(None);
        let mut child = sleeper();
        child.kill().unwrap();
        child.wait().unwrap();
        let job = attach_job(&mon, &child, Job::kill_on_close);
        assert!(job.is_none());
        let w = warnings(&mon);
        assert_eq!(w.len(), 1, "{w:?}");
        assert_eq!(w[0].0, "app");
        assert!(w[0].1.contains("may outlive the app"), "{}", w[0].1);
    }

    #[test]
    fn a_job_that_takes_the_process_is_kept_and_says_nothing() {
        let mon = Monitor::new(None);
        let mut child = sleeper();
        let job = attach_job(&mon, &child, Job::kill_on_close);
        assert!(job.is_some());
        assert!(mon.logs_after(0).is_empty(), "{:?}", mon.logs_after(0));
        drop(job); // closing the job ends the process
        let deadline = Instant::now() + Duration::from_secs(3);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() > deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the process outlived its job");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Global Constraints: `--threads N` only when set.
    #[test]
    fn threads_are_passed_only_when_set() {
        let mut c = EngineConfig::defaults(PathBuf::from("lorren-engine.exe"));
        let model = Path::new("M:/New Drive/ai models/m.gguf");
        let a = c.args(model);
        assert_eq!(a[..3], ["serve", "M:/New Drive/ai models/m.gguf", "--host"]);
        assert!(!a.iter().any(|x| x == "--threads"), "{a:?}");
        c.threads = Some(6);
        let a = c.args(model);
        assert_eq!(a[a.len() - 2..], ["--threads", "6"]);
    }
}
