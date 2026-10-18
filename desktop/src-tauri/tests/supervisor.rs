#![cfg(feature = "fake-engine")]

use lorren_lib::engine::job::Job;
use lorren_lib::engine::supervisor::*;
use lorren_lib::monitor::Monitor;
use std::path::PathBuf;
use std::time::Duration;

fn cfg(env: &[(&str, String)]) -> EngineConfig {
    let mut c = EngineConfig::defaults(PathBuf::from(env!("CARGO_BIN_EXE_fake-engine")));
    c.health_interval = Duration::from_millis(100);
    c.health_timeout = Duration::from_millis(300);
    c.backoff = vec![Duration::from_millis(50); 3];
    c.load_timeout = Duration::from_secs(10);
    c.check_memory = false;
    c.env = env.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
    c
}

fn model(id: &str, path: &str) -> ModelRef {
    ModelRef { id: id.into(), path: PathBuf::from(path), file_bytes: 0, kv_bytes_per_token: 0 }
}

fn events_file(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lorren-sup-{}-{}.log", std::process::id(), tag));
    let _ = std::fs::remove_file(&p);
    p
}

async fn wait_for(sup: &Supervisor, want: EngineState, secs: u64) -> EngineStatus {
    let mut rx = sup.subscribe();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let now = rx.borrow().clone();
        if now.state == want {
            return now;
        }
        if tokio::time::timeout_at(deadline, rx.changed()).await.is_err() {
            panic!("timed out waiting for {want:?}; status is {:?}", sup.status());
        }
    }
}

#[tokio::test]
async fn it_starts_the_engine_and_reports_serving() {
    let sup = Supervisor::new(cfg(&[]), Monitor::new(None));
    let port = sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let s = sup.status();
    assert_eq!(s.state, EngineState::Serving);
    assert_eq!(s.port, Some(port));
    assert!(s.pid.is_some() && s.load_ms.is_some());
    assert_eq!((s.ctx, s.seqs, s.kv.as_str()), (4096, 2, "q8"));
    sup.stop().await;
}

/// Review Focus 1: every one of Kabir's model paths has a space in it.
#[tokio::test]
async fn a_model_path_with_spaces_arrives_as_one_argument() {
    let ev = events_file("spaces");
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_EVENTS", ev.display().to_string())]), Monitor::new(None));
    sup.ensure_model(&model("n", "M:/New Drive/ai models/nexus.gguf")).await.unwrap();
    let text = std::fs::read_to_string(&ev).unwrap();
    let argv: Vec<String> = serde_json::from_str(text.lines().next().unwrap().strip_prefix("argv ").unwrap()).unwrap();
    assert_eq!(argv[1], "serve");
    assert_eq!(argv[2], "M:/New Drive/ai models/nexus.gguf");
    assert!(argv.windows(2).any(|w| w[0] == "--ctx" && w[1] == "4096"));
    assert!(argv.windows(2).any(|w| w[0] == "--seqs" && w[1] == "2"));
    assert!(argv.windows(2).any(|w| w[0] == "--kv" && w[1] == "q8"));
    sup.stop().await;
}

#[tokio::test]
async fn a_refused_model_reports_the_engines_own_reason() {
    let mon = Monitor::new(None);
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "refuse".into())]), mon.clone());
    let err = sup.ensure_model(&model("vl", "vl.gguf")).await.unwrap_err();
    assert!(matches!(err, EngineError::LoadFailed(Some(ref m)) if m.contains("general.architecture is 'qwen2vl'")), "{err}");
    assert_eq!(sup.status().state, EngineState::Failed);
    // The refusal is a stderr line that arrives before the banner: it is logged at `error`.
    let line = mon
        .logs_after(0)
        .into_iter()
        .find(|l| l.source == "engine" && l.text.contains("general.architecture is 'qwen2vl'"))
        .expect("the engine's stderr line is in the log");
    assert_eq!(line.level, "error");
}

#[tokio::test]
async fn a_crash_is_restarted_and_three_crashes_give_up() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "crash_after_ms:300".into())]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let first = sup.status().pid;
    wait_for(&sup, EngineState::Restarting, 5).await;
    let again = wait_for(&sup, EngineState::Serving, 5).await;
    assert_ne!(again.pid, first, "a new process");
    assert!(again.restarts >= 1);
    let failed = wait_for(&sup, EngineState::Failed, 10).await;
    assert!(failed.last_error.unwrap().contains("failed 3 times"));
}

#[tokio::test]
async fn a_hung_engine_is_detected_and_replaced() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "hang".into())]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let first = sup.status().pid;
    let s = wait_for(&sup, EngineState::Restarting, 5).await;
    assert!(s.last_error.unwrap().contains("/health"));
    let again = wait_for(&sup, EngineState::Serving, 5).await;
    assert_ne!(again.pid, first, "a new process replaced the hung one");
    assert!(again.restarts >= 1);
    sup.stop().await;
}

#[tokio::test]
async fn switching_models_restarts_and_the_same_model_does_not() {
    let sup = Supervisor::new(cfg(&[]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let pid_a = sup.status().pid;
    sup.ensure_model(&model("b", "b.gguf")).await.unwrap();
    let s = sup.status();
    assert_eq!(s.model_id.as_deref(), Some("b"));
    assert_ne!(s.pid, pid_a);
    sup.ensure_model(&model("b", "b.gguf")).await.unwrap();
    assert_eq!(sup.status().pid, s.pid, "no restart for the model already serving");
    sup.stop().await;
}

/// Review Focus 4: no orphan engine when the app dies.
#[test]
fn closing_the_job_kills_the_engine() {
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_fake-engine"))
        .args(["serve", "x.gguf", "--port", "0"])
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let job = Job::kill_on_close().unwrap();
    job.assign(&child).unwrap();
    drop(job);
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while child.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "the engine outlived its job");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn the_banner_gives_the_port() {
    assert_eq!(parse_banner("stoke serve: listening on 127.0.0.1:52341  ctx=4096"), Some(52341));
    assert_eq!(parse_banner("fake engine: serving x"), None);
}

/// The real engine answers /health only between scheduler steps (see `EngineConfig::defaults`).
#[test]
fn health_timeout_outlasts_two_prefill_steps() {
    let c = EngineConfig::defaults(PathBuf::from("lorren-engine.exe"));
    assert!(c.health_timeout >= Duration::from_secs(30), "{:?}", c.health_timeout);
}

// ---------------------------------------------------------------------------------------------
// Everything above is the brief's test file, verbatim. Below: the R35 test (RED first) and the
// coverage the brief's tests leave open -- the memory check, a load that times out, stop(), and the
// engine's output reaching the Monitor. Each starts an engine through the fake, and the ones that
// use a FAKE_ENGINE_* switch fail if the switch does not do what the test assumes.
// ---------------------------------------------------------------------------------------------

/// R35.1: a spawn that fails must leave the status at Failed, not at Starting for ever.
#[tokio::test]
async fn a_spawn_failure_marks_the_engine_failed() {
    let mon = Monitor::new(None);
    let mut c = EngineConfig::defaults(PathBuf::from("Z:/no/such/engine.exe"));
    c.check_memory = false;
    let sup = Supervisor::new(c, mon.clone());
    let err = sup.ensure_model(&model("a", "a.gguf")).await.unwrap_err();
    assert!(matches!(err, EngineError::Spawn(_)), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed, "{s:?}");
    assert_eq!(s.last_error.as_deref(), Some(err.to_string().as_str()), "{s:?}");
    assert!(s.port.is_none() && s.pid.is_none(), "{s:?}");
    let logged = mon.logs_after(0).into_iter().any(|l| l.source == "app" && l.level == "error" && l.text == err.to_string());
    assert!(logged, "the failure is in the log: {:?}", mon.logs_after(0));
}

/// An engine that does not print its banner within `load_timeout` is killed and reported as such.
/// (FAKE_ENGINE_LOAD_MS is spelled as a plain integer: the fake ignores a value it cannot parse.)
#[tokio::test]
async fn a_load_that_outlasts_the_timeout_is_reported_and_the_engine_killed() {
    let mon = Monitor::new(None);
    let mut c = cfg(&[("FAKE_ENGINE_LOAD_MS", "2500".into())]);
    c.load_timeout = Duration::from_secs(1);
    let sup = Supervisor::new(c, mon.clone());
    let err = sup.ensure_model(&model("a", "a.gguf")).await.unwrap_err();
    assert!(matches!(err, EngineError::LoadTimeout(1)), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed);
    assert_eq!(s.last_error.as_deref(), Some("the engine did not start within 1 s"));
    assert!(s.port.is_none() && s.pid.is_none());
    assert!(mon.logs_after(0).iter().any(|l| l.source == "app" && l.level == "error" && l.text.contains("did not start within 1 s")));
    // Killed means dead: a live engine would end its 2.5 s load, bind and print its banner (the stdout
    // reader would log it). Wait past that moment -- 1.5 s from now at the earliest -- and look.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    let logs = mon.logs_after(0);
    assert!(!logs.iter().any(|l| l.source == "engine" && l.text.contains("listening on")), "the engine outlived its failed start: {logs:?}");
}

/// The memory check (spec 3.1) refuses before anything is started.
#[tokio::test]
async fn a_model_that_does_not_fit_is_refused_before_an_engine_starts() {
    let ev = events_file("nofit");
    let mut c = cfg(&[("FAKE_ENGINE_EVENTS", ev.display().to_string())]);
    c.check_memory = true;
    let sup = Supervisor::new(c, Monitor::new(None));
    // 64 TiB of model file: more than any PC has.
    let huge = ModelRef { id: "huge".into(), path: PathBuf::from("huge.gguf"), file_bytes: 64u64 << 40, kv_bytes_per_token: 0 };
    let err = sup.ensure_model(&huge).await.unwrap_err();
    assert!(matches!(err, EngineError::NotEnoughMemory { .. }), "{err}");
    assert!(err.to_string().starts_with("not enough memory to load huge"), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed);
    assert_eq!(s.last_error.as_deref(), Some(err.to_string().as_str()));
    assert!(!ev.exists(), "no engine process was started (the fake records its argv at startup)");
}

/// A GGUF header is hostile input: its sizes can saturate `run_bytes`, and adding the spare to a
/// saturated number must not wrap it back into "fits" (or panic).
#[tokio::test]
async fn a_saturated_size_cannot_wrap_the_memory_check_into_a_pass() {
    let ev = events_file("wrap");
    let mut c = cfg(&[("FAKE_ENGINE_EVENTS", ev.display().to_string())]);
    c.check_memory = true;
    let sup = Supervisor::new(c, Monitor::new(None));
    let hostile = ModelRef { id: "hostile".into(), path: PathBuf::from("hostile.gguf"), file_bytes: 1 << 30, kv_bytes_per_token: u64::MAX };
    let err = sup.ensure_model(&hostile).await.unwrap_err();
    assert!(matches!(err, EngineError::NotEnoughMemory { .. }), "{err}");
    assert_eq!(sup.status().state, EngineState::Failed);
    assert!(!ev.exists(), "no engine process was started");
}

/// The other side of the check: a model that fits is started. With no spare and an empty file the
/// need is only the runtime's own 256 MiB, so this fails only if the memory reading is broken.
#[tokio::test]
async fn a_model_that_fits_passes_the_memory_check() {
    let mut c = cfg(&[]);
    c.check_memory = true;
    c.min_free_bytes = 0;
    let sup = Supervisor::new(c, Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    assert_eq!(sup.status().state, EngineState::Serving);
    sup.stop().await;
}

/// `stop()` ends the process (the listener goes with it) and the status says so.
#[tokio::test]
async fn stop_kills_the_engine_and_reports_stopped() {
    let sup = Supervisor::new(cfg(&[]), Monitor::new(None));
    let port = sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    assert!(std::net::TcpStream::connect(("127.0.0.1", port)).is_ok(), "the engine is listening before stop()");
    sup.stop().await;
    let s = sup.status();
    assert_eq!((s.state, s.port, s.pid), (EngineState::Stopped, None, None));
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
        assert!(std::time::Instant::now() < deadline, "the engine is still listening after stop()");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// "Waits for any start already in progress": two requests for one model start one engine.
#[tokio::test]
async fn two_requests_for_one_model_start_one_engine() {
    let ev = events_file("twice");
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_EVENTS", ev.display().to_string())]), Monitor::new(None));
    let m = model("a", "a.gguf");
    let (p1, p2) = tokio::join!(sup.ensure_model(&m), sup.ensure_model(&m));
    assert_eq!(p1.unwrap(), p2.unwrap());
    let starts = std::fs::read_to_string(&ev).unwrap().lines().filter(|l| l.starts_with("argv ")).count();
    let _ = std::fs::remove_file(&ev);
    assert_eq!(starts, 1, "one process was started");
    sup.stop().await;
}

/// "An explicit request retries after giving up": the old failures stop counting, so the new engine's
/// first crash is a restart, not a fourth failure that gives up at once.
#[tokio::test]
async fn an_explicit_request_retries_after_the_supervisor_has_given_up() {
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "crash_after_ms:300".into())]), Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    wait_for(&sup, EngineState::Failed, 15).await;
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    assert_eq!(sup.status().state, EngineState::Serving);
    let s = wait_for(&sup, EngineState::Restarting, 5).await;
    assert_eq!(s.last_error.as_deref(), Some("the engine exited"));
    sup.stop().await;
}

/// The engine's own output reaches the Monitor under source "engine" (the Logs tab), and its banner is there BEFORE the
/// supervisor's own "engine serving ..." line: the stdout reader logs a line first and announces the banner after (the
/// order itself is pinned by the unit test in supervisor.rs), so by the time `ensure_model` returns the line is in the
/// log and no waiting for it is needed.
#[tokio::test]
async fn the_engines_banner_is_logged_under_source_engine() {
    let mon = Monitor::new(None);
    let sup = Supervisor::new(cfg(&[]), mon.clone());
    let port = sup.ensure_model(&model("a", "a.gguf")).await.unwrap();
    let logs = mon.logs_after(0);
    let banner = logs
        .iter()
        .find(|l| l.source == "engine" && l.text.contains(&format!("listening on 127.0.0.1:{port}")))
        .unwrap_or_else(|| panic!("the banner is not in the log yet: {logs:?}"));
    assert_eq!(banner.level, "info");
    let serving = logs
        .iter()
        .find(|l| l.source == "app" && l.text.starts_with("engine serving"))
        .unwrap_or_else(|| panic!("no 'engine serving' line: {logs:?}"));
    assert!(banner.seq < serving.seq, "the engine's banner comes before the app's 'engine serving' line: {logs:?}");
    sup.stop().await;
}

/// R47: an engine that exits during its load without writing a line has given no reason of its own. The error says so
/// as data (`LoadFailed(None)`, which the chat service does not mark a model CAN'T RUN for), and in the same words as
/// before ("no output"); the supervisor is Failed with those words, and nothing was logged as the engine's output.
#[tokio::test]
async fn an_engine_that_exits_silently_during_its_load_gives_no_reason() {
    let mon = Monitor::new(None);
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_MODE", "silent_exit".into())]), mon.clone());
    let err = sup.ensure_model(&model("a", "a.gguf")).await.unwrap_err();
    assert!(matches!(err, EngineError::LoadFailed(None)), "{err:?}");
    assert_eq!(err.to_string(), "the engine stopped while loading: no output");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed);
    assert_eq!(s.last_error.as_deref(), Some("the engine stopped while loading: no output"));
    let engine_lines: Vec<_> = mon.logs_after(0).into_iter().filter(|l| l.source == "engine").collect();
    assert!(engine_lines.is_empty(), "the silent engine wrote nothing: {engine_lines:?}");
}

// ---------------------------------------------------------------------------------------------
// Task 10 rulings R38 / R40: what a refused or failed model switch leaves behind. Each was run
// RED against the unfixed supervisor first.
// ---------------------------------------------------------------------------------------------

/// R40: the memory refusal sets Failed and last_error, so model_id/model_path must name the model it
/// refused. Left alone they name the previous model (or none), and a Models row keyed on model_id would
/// mark the wrong model CAN'T RUN.
#[tokio::test]
async fn a_memory_refusal_names_the_model_it_refused() {
    let mut c = cfg(&[]);
    c.check_memory = true;
    c.min_free_bytes = 0;
    let sup = Supervisor::new(c, Monitor::new(None));
    let huge = ModelRef { id: "huge".into(), path: PathBuf::from("M:/New Drive/ai models/huge.gguf"), file_bytes: 1 << 60, kv_bytes_per_token: 0 };
    let err = sup.ensure_model(&huge).await.unwrap_err();
    assert!(matches!(err, EngineError::NotEnoughMemory { .. }), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed, "{s:?}");
    assert_eq!(s.model_id.as_deref(), Some("huge"), "{s:?}");
    assert_eq!(s.model_path.as_deref(), Some("M:/New Drive/ai models/huge.gguf"), "{s:?}");
}

/// R38a, spec 5: "a model won't load ... and the previous model is started again". The fake refuses only
/// a path that contains FAKE_ENGINE_REFUSE_MODEL, so the first model loads and the second does not.
#[tokio::test]
async fn a_model_that_will_not_load_brings_the_previous_one_back() {
    let mon = Monitor::new(None);
    let sup = Supervisor::new(cfg(&[("FAKE_ENGINE_REFUSE_MODEL", "refusable".into())]), mon.clone());
    sup.ensure_model(&model("prev-model", "prev.gguf")).await.unwrap();
    let first_pid = sup.status().pid;
    assert!(first_pid.is_some());
    let err = sup.ensure_model(&model("refused-model", "refusable-b.gguf")).await.unwrap_err();
    assert!(
        matches!(&err, EngineError::LoadFailed(Some(line)) if line.contains("general.architecture is 'qwen2vl'") && line.contains("refusable-b.gguf")),
        "the original error comes back: {err}"
    );
    let s = sup.status();
    assert_eq!(s.state, EngineState::Serving, "{s:?}");
    assert_eq!(s.model_id.as_deref(), Some("prev-model"), "{s:?}");
    assert_eq!(s.model_path.as_deref(), Some("prev.gguf"), "{s:?}");
    assert!(s.pid.is_some() && s.pid != first_pid, "a new process serves the previous model: {s:?}");
    let port = s.port.expect("a serving engine has a port");
    assert!(std::net::TcpStream::connect(("127.0.0.1", port)).is_ok(), "the previous model's engine is really listening");
    let warns: Vec<String> = mon.logs_after(0).into_iter().filter(|l| l.level == "warn").map(|l| l.text).collect();
    assert_eq!(warns, ["refused-model did not start; starting prev-model again"], "{warns:?}");
    sup.stop().await;
}

/// R38a again, for a refusal that comes before any process is started: the memory check runs after the
/// old engine has been stopped, so without the fallback the user is left with no engine at all.
#[tokio::test]
async fn a_memory_refusal_brings_the_previous_one_back() {
    let mut c = cfg(&[]);
    c.check_memory = true;
    c.min_free_bytes = 0;
    let sup = Supervisor::new(c, Monitor::new(None));
    sup.ensure_model(&model("a", "a.gguf")).await.unwrap(); // file_bytes 0: needs only the runtime's own 256 MiB
    let first_pid = sup.status().pid;
    let huge = ModelRef { id: "huge".into(), path: PathBuf::from("huge.gguf"), file_bytes: 1 << 60, kv_bytes_per_token: 0 };
    let err = sup.ensure_model(&huge).await.unwrap_err();
    assert!(matches!(err, EngineError::NotEnoughMemory { .. }), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Serving, "{s:?}");
    assert_eq!(s.model_id.as_deref(), Some("a"), "{s:?}");
    assert!(s.pid.is_some() && s.pid != first_pid, "a new process serves the previous model: {s:?}");
    sup.stop().await;
}

// ---------------------------------------------------------------------------------------------
// Pre-review addition (task 10): the clause of R38a that no ruling test pinned. Shown not vacuous by
// running it against the mutant that drops the `p.id != model.id` filter (see task-10-report.md).
// ---------------------------------------------------------------------------------------------

/// R38a: only a DIFFERENT running model is brought back. Asking again for the model that was running (after
/// a crash, or from Failed) has no fallback: without the `p.id != model.id` filter this request would start
/// the old model once more after the refusal and end Serving, and a request that keeps failing (a load
/// timeout, say) would be tried twice under the supervisor's lock before its error came back.
#[tokio::test]
async fn a_request_for_the_model_that_was_running_has_no_fallback() {
    let mon = Monitor::new(None);
    // The engine dies 200 ms after its banner and the health watcher (60 s) does not get there first, so the
    // supervisor still holds the dead process as "running" when the next request arrives.
    let mut c = cfg(&[("FAKE_ENGINE_MODE", "crash_after_ms:200".into())]);
    c.health_interval = Duration::from_secs(60);
    c.check_memory = true;
    c.min_free_bytes = 0;
    let sup = Supervisor::new(c, mon.clone());
    let a = model("a", "a.gguf");
    let port = sup.ensure_model(&a).await.unwrap();
    // Wait for the crash itself (the port stops accepting), not for a guessed number of milliseconds.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
        assert!(std::time::Instant::now() < deadline, "the fake engine never crashed");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    tokio::time::sleep(Duration::from_millis(200)).await; // the process reads as exited a moment after its socket closes
    // The same id again, and it no longer fits.
    let grown = ModelRef { file_bytes: 1 << 60, ..a.clone() };
    let err = sup.ensure_model(&grown).await.unwrap_err();
    assert!(matches!(err, EngineError::NotEnoughMemory { .. }), "{err}");
    let s = sup.status();
    assert_eq!(s.state, EngineState::Failed, "no fallback to the model that was just asked for: {s:?}");
    let warns: Vec<String> = mon.logs_after(0).into_iter().filter(|l| l.level == "warn").map(|l| l.text).collect();
    assert!(warns.iter().all(|w| !w.contains("did not start")), "{warns:?}");
    sup.stop().await;
}
