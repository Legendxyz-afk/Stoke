#![cfg(feature = "fake-engine")]

use lorren_lib::chat::*;
use lorren_lib::engine::supervisor::{EngineConfig, EngineState, Supervisor};
use lorren_lib::gguf::fixture::{self, Spec};
use lorren_lib::library::{self, LibraryConfig};
use lorren_lib::monitor::Monitor;
use lorren_lib::store::{MsgStatus, Role, Store};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

struct World {
    svc: Arc<ChatService>,
    events_file: PathBuf,
    qwen: String,
    nexus: String,
}

fn world(tag: &str, env: &[(&str, &str)], ctx: u32, settings: ChatSettings) -> World {
    let dir = std::env::temp_dir().join(format!("lorren chat {}-{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    fixture::write(&dir.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
    fixture::write(&dir.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
    let models = library::scan(&LibraryConfig {
        folders: vec![dir.clone()],
        ollama_models_dir: None,
        default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
    });
    let qwen = models.iter().find(|m| m.is_default).unwrap().id.clone();
    let nexus = models.iter().find(|m| m.name == "nexus").unwrap().id.clone();
    let events_file = dir.join("events.log");
    let mut cfg = EngineConfig::defaults(PathBuf::from(env!("CARGO_BIN_EXE_fake-engine")));
    cfg.ctx = ctx;
    cfg.check_memory = false;
    cfg.health_interval = Duration::from_millis(100);
    cfg.backoff = vec![Duration::from_millis(50); 3];
    cfg.env = env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    cfg.env.push(("FAKE_ENGINE_EVENTS".into(), events_file.display().to_string()));
    let monitor = Monitor::new(None);
    let sup = Supervisor::new(cfg, monitor.clone());
    let store = Arc::new(Mutex::new(Store::open_in_memory().unwrap()));
    let svc = ChatService::new(store, sup, Arc::new(RwLock::new(models)), monitor, settings);
    World { svc, events_file, qwen, nexus }
}

fn fast() -> ChatSettings {
    ChatSettings { liveness: Duration::from_secs(5), save_every: Duration::from_millis(50), ..ChatSettings::default() }
}

fn collector() -> (Arc<Mutex<Vec<StreamEvent>>>, impl Fn(StreamEvent) + Send + Sync + Clone) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s2 = seen.clone();
    (seen, move |e| s2.lock().unwrap().push(e))
}

fn events(w: &World) -> String {
    std::fs::read_to_string(&w.events_file).unwrap_or_default()
}

fn new_chat(w: &World, model: &str) -> String {
    w.svc.store.lock().unwrap().create_chat(model, None).unwrap().id
}

fn last_reply(w: &World, chat: &str) -> lorren_lib::store::Message {
    w.svc.store.lock().unwrap().active_path(chat).unwrap().pop().unwrap()
}

#[tokio::test]
async fn a_reply_streams_into_the_ui_and_the_store() {
    let w = world("stream", &[("FAKE_ENGINE_REPLY", "Verdict: HOLD."), ("FAKE_ENGINE_TOKEN_MS", "5")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    w.svc.send(&chat, "Long or wait?", sink).await.unwrap();
    let seen = seen.lock().unwrap();
    // `queued` comes first (demo D2), then `started` once the engine has taken the request.
    assert!(matches!(seen.first(), Some(StreamEvent::Queued { .. })), "{seen:?}");
    assert!(matches!(seen.get(1), Some(StreamEvent::Started { dropped_turns: 0, .. })), "{seen:?}");
    let text: String = seen.iter().filter_map(|e| match e { StreamEvent::Token { text } => Some(text.clone()), _ => None }).collect();
    assert_eq!(text, "Verdict: HOLD.");
    assert!(matches!(seen.last(), Some(StreamEvent::Finished { stats }) if stats.completion_tokens == Some(2)));
    let reply = last_reply(&w, &chat);
    assert_eq!((reply.role, reply.status, reply.content.as_str()), (Role::Assistant, MsgStatus::Done, "Verdict: HOLD."));
    let req = &w.svc.monitor.requests()[0];
    assert_eq!(req.status, "ok");
    assert!(req.ttft_ms.is_some());
    let body = events(&w);
    assert!(body.contains("\"max_tokens\":1024") && body.contains("\"include_usage\":true"), "{body}");
}

/// Review Focus 2, end to end.
#[tokio::test]
async fn multibyte_text_arrives_intact() {
    let w = world("utf8", &[("FAKE_ENGINE_REPLY", "Buy at ₹68,400 🚀 now"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&chat, "₹?", sink).await.unwrap();
    assert_eq!(last_reply(&w, &chat).content, "Buy at ₹68,400 🚀 now");
}

#[tokio::test]
async fn stop_cancels_the_generation_in_the_engine() {
    let long = "word ".repeat(60);
    let w = world("stop", &[("FAKE_ENGINE_REPLY", long.as_str()), ("FAKE_ENGINE_TOKEN_MS", "100")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let svc = w.svc.clone();
    let c2 = chat.clone();
    let task = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    for _ in 0..100 {
        if seen.lock().unwrap().iter().any(|e| matches!(e, StreamEvent::Token { .. })) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    w.svc.stop(&chat);
    task.await.unwrap().unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Stopped { .. })));
    let reply = last_reply(&w, &chat);
    assert_eq!(reply.status, MsgStatus::Stopped);
    assert!(!reply.content.is_empty() && reply.content.len() < long.len());
    for _ in 0..30 {
        if events(&w).contains("cancelled") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the engine never saw the connection close:\n{}", events(&w));
}

#[tokio::test]
async fn an_engine_crash_marks_the_reply_interrupted_and_the_next_send_works() {
    let marker = std::env::temp_dir().join(format!("lorren-crash-once-{}", std::process::id()));
    let _ = std::fs::remove_file(&marker);
    let long = "word ".repeat(40);
    let m = marker.display().to_string();
    let w = world(
        "crash",
        &[("FAKE_ENGINE_MODE", "crash_after_ms:400"), ("FAKE_ENGINE_CRASH_ONCE", m.as_str()),
          ("FAKE_ENGINE_REPLY", long.as_str()), ("FAKE_ENGINE_TOKEN_MS", "50")],
        4096,
        fast(),
    );
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    w.svc.send(&chat, "go", sink.clone()).await.unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Interrupted { .. })));
    assert_eq!(last_reply(&w, &chat).status, MsgStatus::Interrupted);
    for _ in 0..100 {
        let st = w.svc.supervisor.status();
        if st.restarts >= 1 && st.state == EngineState::Serving {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    w.svc.send(&chat, "again", sink).await.unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Finished { .. })));
}

#[tokio::test]
async fn a_full_context_is_trimmed_and_retried_once() {
    let settings = ChatSettings { history_budget: 100_000, ..fast() };
    let w = world("ctx", &[("FAKE_ENGINE_REPLY", "ok"), ("FAKE_ENGINE_TOKEN_MS", "1")], 512, settings);
    let chat = new_chat(&w, &w.qwen);
    {
        let s = w.svc.store.lock().unwrap();
        let mut parent: Option<String> = None;
        for i in 0..6 {
            let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
            let m = s.append_message(&chat, parent.as_deref(), role, &"y".repeat(400), MsgStatus::Done, None).unwrap();
            parent = Some(m.id);
        }
    }
    let (seen, sink) = collector();
    w.svc.send(&chat, "and now?", sink).await.unwrap();
    let seen = seen.lock().unwrap();
    // (`started` is the second event: `queued` is the first, demo D2.)
    assert!(matches!(seen.get(1), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns > 0), "{seen:?}");
    assert!(matches!(seen.last(), Some(StreamEvent::Finished { .. })), "{seen:?}");
    assert_eq!(events(&w).matches("context_length_exceeded").count(), 1);
}

#[tokio::test]
async fn the_history_budget_drops_the_oldest_turns() {
    let settings = ChatSettings { history_budget: 60, ..fast() };
    let w = world("budget", &[("FAKE_ENGINE_REPLY", "ok"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, settings);
    let chat = new_chat(&w, &w.qwen);
    {
        let s = w.svc.store.lock().unwrap();
        let mut parent: Option<String> = None;
        for i in 0..4 {
            let role = if i % 2 == 0 { Role::User } else { Role::Assistant };
            let m = s.append_message(&chat, parent.as_deref(), role, &format!("turn {i} {}", "z".repeat(100)), MsgStatus::Done, None).unwrap();
            parent = Some(m.id);
        }
    }
    let (seen, sink) = collector();
    w.svc.send(&chat, "latest", sink).await.unwrap();
    // (`started` is the second event: `queued` is the first, demo D2.)
    assert!(matches!(seen.lock().unwrap().get(1), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns >= 2));
    let body = events(&w);
    assert!(!body.contains("turn 0") && body.contains("latest"), "{body}");
}

#[tokio::test]
async fn regenerate_adds_a_sibling_and_shows_it() {
    let w = world("regen", &[("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&chat, "hi", sink.clone()).await.unwrap();
    let first = last_reply(&w, &chat);
    w.svc.regenerate(&chat, sink).await.unwrap();
    let second = last_reply(&w, &chat);
    assert_ne!(first.id, second.id);
    assert_eq!(second.parent_id, first.parent_id);
    assert_eq!(w.svc.store.lock().unwrap().siblings(&second.id).unwrap().len(), 2);
}

/// Review Focus 3: a send while the engine is switching models waits instead of failing.
#[tokio::test]
async fn a_send_during_a_model_switch_waits_for_the_engine() {
    let w = world("switch", &[("FAKE_ENGINE_LOAD_MS", "400"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.nexus);
    let lib = w.svc.library.read().unwrap().clone();
    let qwen = lib.iter().find(|m| m.id == w.qwen).unwrap().clone();
    let sup = w.svc.supervisor.clone();
    let switching = tokio::spawn(async move { sup.ensure_model(&model_ref(&qwen)).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (seen, sink) = collector();
    w.svc.send(&chat, "hello", sink).await.unwrap();
    switching.await.unwrap().unwrap();
    assert!(matches!(seen.lock().unwrap().last(), Some(StreamEvent::Finished { .. })));
    assert_eq!(w.svc.supervisor.status().model_id.as_deref(), Some(w.nexus.as_str()));
}

/// Review Focus 5: refused before any request is sent.
#[tokio::test]
async fn a_message_longer_than_the_context_is_refused_up_front() {
    let w = world("toolong", &[], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    let err = w.svc.send(&chat, &"word ".repeat(5_000), sink).await.unwrap_err();
    assert!(matches!(err, ChatError::TooLong(est, 4096) if est > 4096), "{err}");
    assert!(!events(&w).contains("request"));
    assert!(w.svc.store.lock().unwrap().active_path(&chat).unwrap().is_empty());
}

// ---------------------------------------------------------------------------------------------
// Everything above is the brief's test file, verbatim. Below: the tests of rulings R38 and R39
// (Task 10 dispatch, 2026-10-16), each run RED against the plan's chat.rs first.
// ---------------------------------------------------------------------------------------------

/// R38 (spec 5): the engine refuses a model's file. That reply ends in an Error that carries the engine's own
/// line, the previous model serves again, the model's row turns CAN'T RUN with that line, and a later send in
/// that chat is refused up front -- without stopping and restarting the previous engine once more.
#[tokio::test]
async fn a_model_the_engine_refuses_is_marked_cant_run_and_the_previous_model_serves() {
    // The fake refuses every model path that contains "nexus", so the world's tag must not.
    let w = world("refuse", &[("FAKE_ENGINE_REFUSE_MODEL", "nexus"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let qchat = new_chat(&w, &w.qwen);
    let (qseen, qsink) = collector();
    w.svc.send(&qchat, "hi", qsink).await.unwrap();
    let qevents = qseen.lock().unwrap().clone();
    assert!(matches!(qevents.last(), Some(StreamEvent::Finished { .. })), "{qevents:?}");

    let nchat = new_chat(&w, &w.nexus);
    let (nseen, nsink) = collector();
    w.svc.send(&nchat, "hello", nsink).await.unwrap();
    let nevents = nseen.lock().unwrap().clone();
    let message = match nevents.last() {
        Some(StreamEvent::Error { message }) => message.clone(),
        other => panic!("the last event of a refused model is an Error, got {other:?}"),
    };
    assert!(message.contains("general.architecture is 'qwen2vl'"), "{message}");
    assert_eq!(last_reply(&w, &nchat).status, MsgStatus::Error);

    // The previous model serves again.
    let s = w.svc.supervisor.status();
    assert_eq!(s.state, EngineState::Serving, "{s:?}");
    assert_eq!(s.model_id.as_deref(), Some(w.qwen.as_str()), "{s:?}");

    // The row turns CAN'T RUN with the engine's own line (the one its stderr reader logged), and only that row.
    let engine_line = w
        .svc
        .monitor
        .logs_after(0)
        .into_iter()
        .find(|l| l.source == "engine" && l.text.contains("general.architecture is 'qwen2vl'"))
        .expect("the engine's refusal is in the log")
        .text;
    assert!(engine_line.contains("nexus.gguf"), "{engine_line}");
    let (nexus, qwen) = {
        let lib = w.svc.library.read().unwrap();
        (
            lib.iter().find(|m| m.id == w.nexus).cloned().unwrap(),
            lib.iter().find(|m| m.id == w.qwen).cloned().unwrap(),
        )
    };
    assert!(!nexus.runnable);
    assert_eq!(nexus.reason.as_deref(), Some(engine_line.as_str()));
    assert!(qwen.runnable && qwen.reason.is_none(), "the previous model's row is untouched: {qwen:?}");

    // A second send in that chat is refused up front, nothing is appended, and the previous engine is left alone
    // (the same process: a stop-and-restart would also end Serving qwen, but under a new pid).
    let before = w.svc.store.lock().unwrap().active_path(&nchat).unwrap().len();
    let pid_before = w.svc.supervisor.status().pid;
    assert!(pid_before.is_some());
    let (_s2, sink2) = collector();
    let err = w.svc.send(&nchat, "again", sink2).await.unwrap_err();
    assert!(matches!(err, ChatError::NotRunnable(..)), "{err}");
    assert!(err.to_string().contains("cannot run: ") && err.to_string().contains("general.architecture"), "{err}");
    assert_eq!(w.svc.store.lock().unwrap().active_path(&nchat).unwrap().len(), before);
    let s = w.svc.supervisor.status();
    assert_eq!(
        (s.state, s.model_id.as_deref(), s.pid),
        (EngineState::Serving, Some(w.qwen.as_str()), pid_before),
        "the refused send left the previous engine alone: {s:?}"
    );
}

/// R39: Stop while the engine is still loading must not abandon the load half-way. The supervisor's future
/// runs to its end in a task of its own, so the engine ends up Serving the model that was asked for.
#[tokio::test]
async fn stop_during_a_model_load_leaves_the_engine_serving() {
    // A plain integer: the fake refuses a *_MS value it cannot parse (exit 2 at startup), so a mistyped one cannot pass for the default.
    let w = world("loadstop", &[("FAKE_ENGINE_LOAD_MS", "1500"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let svc = w.svc.clone();
    let c2 = chat.clone();
    let task = tokio::spawn(async move { svc.send(&c2, "hello", sink).await });
    let mut starting = false;
    for _ in 0..300 {
        if w.svc.supervisor.status().state == EngineState::Starting {
            starting = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(starting, "the engine never reported Starting: {:?}", w.svc.supervisor.status());
    w.svc.stop(&chat);
    task.await.unwrap().unwrap();
    // Stop returned PROMPTLY: the load takes 1.5 s and is still going on. A send that waited for the load to end before it
    // acted on Stop would only have returned once the engine was Serving.
    let s = w.svc.supervisor.status();
    assert_eq!(s.state, EngineState::Starting, "Stop must not wait for the load: {s:?}");
    let evs = seen.lock().unwrap().clone();
    assert!(matches!(evs.last(), Some(StreamEvent::Stopped { .. })), "{evs:?}");
    assert_eq!(last_reply(&w, &chat).status, MsgStatus::Stopped);

    // Within 4 s the load that Stop left behind ends: Serving, on the model that was asked for ...
    let deadline = tokio::time::Instant::now() + Duration::from_secs(4);
    loop {
        let s = w.svc.supervisor.status();
        if s.state == EngineState::Serving {
            assert_eq!(s.model_id.as_deref(), Some(w.qwen.as_str()), "{s:?}");
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "after Stop during the load the engine never reached Serving: {s:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // ... and it stays off Starting.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let s = w.svc.supervisor.status();
    assert_eq!((s.state, s.model_id.as_deref()), (EngineState::Serving, Some(w.qwen.as_str())), "{s:?}");
}

// ---------------------------------------------------------------------------------------------
// Pre-review addition (task 10): the clause of R38b that no ruling test pinned. Shown not vacuous by running it
// against the mutant that marks a model CAN'T RUN on ANY start error (see task-10-report.md).
// ---------------------------------------------------------------------------------------------

/// R38b: ONLY the engine's own refusal of the file (LoadFailed) marks a model CAN'T RUN. A memory refusal is not
/// the file's fault and can pass later: the row stays runnable, the previous model serves again, and the next
/// send in that chat tries again instead of being refused up front. `world()` switches the memory check off, so
/// this test builds its own world with it on, and makes the nexus file look larger than any PC's memory.
#[tokio::test]
async fn a_memory_refusal_does_not_mark_the_model_cant_run() {
    let dir = std::env::temp_dir().join(format!("lorren chat {}-memrefuse", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    fixture::write(&dir.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
    fixture::write(&dir.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
    let mut models = library::scan(&LibraryConfig {
        folders: vec![dir.clone()],
        ollama_models_dir: None,
        default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
    });
    let qwen = models.iter().find(|m| m.is_default).unwrap().id.clone();
    let nexus = models.iter().find(|m| m.name == "nexus").unwrap().id.clone();
    models.iter_mut().find(|m| m.id == nexus).unwrap().size_bytes = 1 << 60; // no PC has this much memory
    let mut cfg = EngineConfig::defaults(PathBuf::from(env!("CARGO_BIN_EXE_fake-engine")));
    cfg.check_memory = true;
    cfg.min_free_bytes = 0;
    cfg.env = vec![("FAKE_ENGINE_TOKEN_MS".into(), "1".into())];
    let monitor = Monitor::new(None);
    let sup = Supervisor::new(cfg, monitor.clone());
    let store = Arc::new(Mutex::new(Store::open_in_memory().unwrap()));
    let svc = ChatService::new(store, sup, Arc::new(RwLock::new(models)), monitor, fast());

    let qchat = svc.store.lock().unwrap().create_chat(&qwen, None).unwrap().id;
    let (qseen, qsink) = collector();
    svc.send(&qchat, "hi", qsink).await.unwrap();
    let qevents = qseen.lock().unwrap().clone();
    assert!(matches!(qevents.last(), Some(StreamEvent::Finished { .. })), "{qevents:?}");

    // Twice: the second send must reach the engine again, not be refused up front with NotRunnable.
    let nchat = svc.store.lock().unwrap().create_chat(&nexus, None).unwrap().id;
    for round in 0..2 {
        let (nseen, nsink) = collector();
        svc.send(&nchat, "hello", nsink)
            .await
            .unwrap_or_else(|e| panic!("round {round}: a memory refusal must not make the chat refuse up front: {e}"));
        let events = nseen.lock().unwrap().clone();
        assert!(
            matches!(events.last(), Some(StreamEvent::Error { message }) if message.contains("not enough memory")),
            "round {round}: {events:?}"
        );
        let entry = svc.library.read().unwrap().iter().find(|m| m.id == nexus).cloned().unwrap();
        assert!(entry.runnable && entry.reason.is_none(), "round {round}: the row is still runnable: {entry:?}");
        let s = svc.supervisor.status();
        assert_eq!((s.state, s.model_id.as_deref()), (EngineState::Serving, Some(qwen.as_str())), "round {round}: {s:?}");
    }
    svc.supervisor.stop().await;
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// Final wave (R47): the case of R38b that the "refuses" and "memory" tests above leave open.
// ---------------------------------------------------------------------------------------------

/// R47: an engine that is killed, or crashes, DURING a load and prints nothing has given no reason of its own, so
/// it is not a verdict on the file: the row stays runnable and the next send tries again. (An engine that REFUSES the
/// file, as in the first R38 test, marks the row CAN'T RUN with its own words.) The fake's `silent_exit` mode exits 1
/// before the banner without writing a line to either stream.
#[tokio::test]
async fn an_engine_that_dies_silently_during_a_load_does_not_mark_the_model_cant_run() {
    let w = world("silentdie", &[("FAKE_ENGINE_MODE", "silent_exit"), ("FAKE_ENGINE_LOAD_MS", "100")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    // Twice: the second send must reach the engine again, not be refused up front with NotRunnable.
    for round in 0..2 {
        let (seen, sink) = collector();
        w.svc
            .send(&chat, "hello", sink)
            .await
            .unwrap_or_else(|e| panic!("round {round}: a silent death must not make the chat refuse up front: {e}"));
        let events = seen.lock().unwrap().clone();
        assert!(
            matches!(events.last(), Some(StreamEvent::Error { message }) if message == "the engine stopped while loading: no output"),
            "round {round}: {events:?}"
        );
        let entry = w.svc.library.read().unwrap().iter().find(|m| m.id == w.qwen).cloned().unwrap();
        assert!(entry.runnable && entry.reason.is_none(), "round {round}: the row is still runnable: {entry:?}");
        assert_eq!(w.svc.supervisor.status().state, EngineState::Failed, "round {round}");
    }
}

// ---------------------------------------------------------------------------------------------
// Demo wave (2026-10-20), D2 and D3. Each test was run RED first: the event did not exist, and the warm-up was not sent.
// ---------------------------------------------------------------------------------------------

fn kind(e: &StreamEvent) -> &'static str {
    match e {
        StreamEvent::Queued { .. } => "queued",
        StreamEvent::Started { .. } => "started",
        StreamEvent::Token { .. } => "token",
        StreamEvent::Finished { .. } => "finished",
        StreamEvent::Stopped { .. } => "stopped",
        StreamEvent::Interrupted { .. } => "interrupted",
        StreamEvent::Error { .. } => "error",
    }
}

/// One event, with what the host looked like the moment it was sent: the engine's state and model, and the chat's stored
/// path as (id, role, status).
type Snapshot = (StreamEvent, EngineState, Option<String>, Vec<(String, Role, MsgStatus)>);

fn recorder(w: &World, chat: &str) -> (Arc<Mutex<Vec<Snapshot>>>, impl Fn(StreamEvent) + Send + Sync) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (record, svc, chat) = (seen.clone(), w.svc.clone(), chat.to_string());
    (seen, move |e| {
        let engine = svc.supervisor.status();
        let path = svc.store.lock().unwrap().active_path(&chat).unwrap().into_iter().map(|m| (m.id, m.role, m.status)).collect();
        record.lock().unwrap().push((e, engine.state, engine.model_id, path));
    })
}

/// D2: while the engine starts or switches (a minute, for a big model) the UI heard nothing, and the message the user had
/// just sent was not on screen. `queued` is the first event, sent as soon as the user's message and the reply's placeholder
/// are stored and BEFORE the engine is asked for anything. The load takes 400 ms here, so "before" is a state the test can
/// read off the supervisor: Stopped at `queued`, Serving at `started`.
#[tokio::test]
async fn a_send_begins_with_queued_before_the_engine_is_asked_for_anything() {
    let w = world("queued-send", &[("FAKE_ENGINE_LOAD_MS", "400"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = recorder(&w, &chat);
    w.svc.send(&chat, "hello", sink).await.unwrap();
    let seen = seen.lock().unwrap();
    let kinds: Vec<_> = seen.iter().map(|s| kind(&s.0)).collect();
    assert_eq!(kinds[..2], ["queued", "started"], "{kinds:?}");
    assert!(kinds.len() > 3 && kinds[2..kinds.len() - 1].iter().all(|k| *k == "token") && kinds.last() == Some(&"finished"), "{kinds:?}");

    let stored = w.svc.store.lock().unwrap().active_path(&chat).unwrap();
    let (user, reply) = (&stored[0], &stored[1]);
    let (queued, state, _model, path) = &seen[0];
    match queued {
        StreamEvent::Queued { message_id, user_message_id } => assert_eq!((message_id, user_message_id), (&reply.id, &user.id)),
        other => panic!("{other:?}"),
    }
    assert_eq!(*state, EngineState::Stopped, "the engine had not been asked to start when `queued` was sent");
    assert_eq!(
        path,
        &[(user.id.clone(), Role::User, MsgStatus::Done), (reply.id.clone(), Role::Assistant, MsgStatus::Streaming)],
        "both messages were stored when `queued` was sent: the user's, and the reply as a streaming placeholder"
    );
    assert_eq!(seen[1].1, EngineState::Serving, "`started` comes once the engine serves");
    assert_eq!(
        serde_json::to_value(&seen[0].0).unwrap(),
        serde_json::json!({"kind": "queued", "message_id": reply.id, "user_message_id": user.id}),
        "the wire form the UI reads"
    );
}

/// The same for a regenerate, which is the case that switches models: the chat has moved to another model, the engine
/// still serves the first, and `queued` is sent while it does. The new placeholder is a sibling of the first reply.
#[tokio::test]
async fn a_regenerate_begins_with_queued_before_the_engine_is_switched() {
    let w = world("queued-regen", &[("FAKE_ENGINE_LOAD_MS", "300"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_first, first_sink) = collector();
    w.svc.send(&chat, "hi", first_sink).await.unwrap();
    let first_reply = last_reply(&w, &chat);
    w.svc.store.lock().unwrap().set_chat_model(&chat, &w.nexus).unwrap();

    let (seen, sink) = recorder(&w, &chat);
    w.svc.regenerate(&chat, sink).await.unwrap();
    let seen = seen.lock().unwrap();
    let kinds: Vec<_> = seen.iter().map(|s| kind(&s.0)).collect();
    assert_eq!(kinds[..2], ["queued", "started"], "{kinds:?}");
    assert_eq!(kinds.last(), Some(&"finished"), "{kinds:?}");

    let new_reply = last_reply(&w, &chat);
    let user_id = first_reply.parent_id.clone().expect("the first reply answers the user's message");
    assert_ne!(new_reply.id, first_reply.id);
    match &seen[0].0 {
        StreamEvent::Queued { message_id, user_message_id } => assert_eq!((message_id, user_message_id), (&new_reply.id, &user_id)),
        other => panic!("{other:?}"),
    }
    assert_eq!((seen[0].1, seen[0].2.as_deref()), (EngineState::Serving, Some(w.qwen.as_str())), "at `queued` the engine still serves the old model");
    assert_eq!(
        seen[0].3,
        [(user_id.clone(), Role::User, MsgStatus::Done), (new_reply.id.clone(), Role::Assistant, MsgStatus::Streaming)],
        "the new placeholder is stored when `queued` is sent"
    );
    assert_eq!((seen[1].1, seen[1].2.as_deref()), (EngineState::Serving, Some(w.nexus.as_str())), "`started` comes once nexus serves");
}

/// D2 for a reply that cannot start: the UI has been told `queued`, and then hears why.
#[tokio::test]
async fn a_send_the_engine_refuses_is_queued_first_and_then_ends_in_an_error() {
    let w = world("queued-refused", &[("FAKE_ENGINE_MODE", "refuse")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = recorder(&w, &chat);
    w.svc.send(&chat, "hello", sink).await.unwrap();
    let kinds: Vec<_> = seen.lock().unwrap().iter().map(|s| kind(&s.0)).collect();
    assert_eq!(kinds, ["queued", "error"]);
}

/// D3: the warm-up that follows the engine's banner goes through the supervisor's own client. The chat service does not
/// see it, and neither do the Requests list nor the request counter: the engine got two requests, the app knows of one.
#[tokio::test]
async fn the_engines_warm_up_is_not_a_request_the_chat_service_or_the_list_sees() {
    let w = world("warmup-hidden", &[("FAKE_ENGINE_REPLY", "Verdict: HOLD."), ("FAKE_ENGINE_TOKEN_MS", "5")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    w.svc.send(&chat, "Long or wait?", sink).await.unwrap();
    let bodies: Vec<serde_json::Value> = events(&w)
        .lines()
        .filter_map(|l| l.strip_prefix("request "))
        .map(|b| serde_json::from_str(b).unwrap())
        .collect();
    assert_eq!(bodies.len(), 2, "the warm-up, then the reply: {bodies:?}");
    assert_eq!(bodies[0]["max_tokens"], 1, "{}", bodies[0]);
    assert!(!bodies[0].to_string().contains("Long or wait?"), "{}", bodies[0]);
    assert_eq!(bodies[1]["max_tokens"], 1024, "{}", bodies[1]);
    assert!(bodies[1].to_string().contains("Long or wait?"), "{}", bodies[1]);
    assert_eq!(w.svc.monitor.requests().len(), 1);
    assert_eq!(w.svc.monitor.request_count(), 1);
    let text: String = seen.lock().unwrap().iter().filter_map(|e| match e { StreamEvent::Token { text } => Some(text.clone()), _ => None }).collect();
    assert_eq!(text, "Verdict: HOLD.", "the warm-up's token is not in the reply");
}

// ---------------------------------------------------------------------------------------------
// Demo D9: the engine loads a chat's model when the user starts typing in it, not when a chat or a model is picked. The engine
// moves ONLY on: (a) the app's start (the default model), (b) `prepare_model`, the first key of a draft, and (c) a send.
// Each test was run RED first.
// ---------------------------------------------------------------------------------------------

/// Waits for the engine to be serving `model`.
async fn wait_serving(w: &World, model: &str, secs: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    loop {
        let s = w.svc.supervisor.status();
        if s.state == EngineState::Serving && s.model_id.as_deref() == Some(model) {
            return;
        }
        assert!(tokio::time::Instant::now() < deadline, "the engine never served {model}: {s:?}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn starts(w: &World) -> usize {
    events(w).lines().filter(|l| l.starts_with("argv ")).count()
}

/// Picking a model for a chat stores it and nothing more. The engine used to switch at once (the demo's log shows the switch
/// 20 s before the next send): now it stays where it was, and the chat has the model all the same.
#[tokio::test]
async fn setting_a_chats_model_stores_it_and_leaves_the_engine_where_it_was() {
    let w = world("d9-set", &[("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let first = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&first, "hi", sink).await.unwrap(); // the engine serves qwen now
    let before = w.svc.supervisor.status();
    let other = new_chat(&w, &w.qwen);
    lorren_lib::commands::switch_chat_model(&w.svc, &other, &w.nexus).unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await; // far longer than the fake takes to load
    let after = w.svc.supervisor.status();
    assert_eq!((after.state, after.model_id.as_deref(), after.pid), (EngineState::Serving, Some(w.qwen.as_str()), before.pid), "{after:?}");
    assert_eq!(starts(&w), 1, "{}", events(&w));
    assert_eq!(w.svc.store.lock().unwrap().get_chat(&other).unwrap().model_id, w.nexus, "the chat has the model it was given");
}

/// ... and a model that does not exist, or cannot run, is refused as before.
#[tokio::test]
async fn setting_a_chats_model_refuses_what_a_send_would_refuse() {
    let w = world("d9-set-refused", &[], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    assert_eq!(lorren_lib::commands::switch_chat_model(&w.svc, &chat, "no-such-model").unwrap_err(), "no such model");
    {
        let mut lib = w.svc.library.write().unwrap();
        let nexus = lib.iter_mut().find(|m| m.id == w.nexus).unwrap();
        nexus.runnable = false;
        nexus.reason = Some("the engine refused it".into());
    }
    let err = lorren_lib::commands::switch_chat_model(&w.svc, &chat, &w.nexus).unwrap_err();
    assert_eq!(err, "nexus cannot run: the engine refused it");
    assert_eq!(w.svc.store.lock().unwrap().get_chat(&chat).unwrap().model_id, w.qwen, "nothing was stored");
}

/// The first key of a draft: the engine, idle, starts the model, and the call does not wait for the load.
#[tokio::test]
async fn preparing_a_model_starts_it_when_the_engine_is_idle_without_waiting_for_the_load() {
    let w = world("d9-start", &[("FAKE_ENGINE_LOAD_MS", "500"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let t0 = std::time::Instant::now();
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::Started);
    assert!(t0.elapsed() < Duration::from_millis(300), "it came back without waiting for the 500 ms load: {:?}", t0.elapsed());
    wait_serving(&w, &w.nexus, 10).await;
    assert_eq!(starts(&w), 1, "{}", events(&w));
    let logs = w.svc.monitor.logs_after(0);
    let line = logs.iter().find(|l| l.text.starts_with("preparing ")).unwrap_or_else(|| panic!("no 'preparing' line: {logs:?}"));
    assert_eq!((line.source.as_str(), line.level.as_str(), line.text.as_str()), ("app", "info", "preparing nexus: the user is typing"));
}

/// It switches the engine from the model it has to the one the chat has, and marks the model that the engine refuses CAN'T RUN
/// (it goes through `start_model`, the app's only way to start an engine).
#[tokio::test]
async fn preparing_a_model_switches_from_the_model_that_is_serving() {
    let w = world("d9-switch", &[("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (_seen, sink) = collector();
    w.svc.send(&chat, "hi", sink).await.unwrap();
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::Started);
    wait_serving(&w, &w.nexus, 10).await;
    assert_eq!(starts(&w), 2, "{}", events(&w));
}

#[tokio::test]
async fn preparing_a_model_the_engine_refuses_marks_it_cant_run() {
    let w = world("d9-refuse", &[("FAKE_ENGINE_REFUSE_MODEL", "nexus"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::Started);
    for _ in 0..200 {
        if !w.svc.library.read().unwrap().iter().find(|m| m.id == w.nexus).unwrap().runnable {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("the refused model was never marked: {:?}", w.svc.library.read().unwrap().iter().find(|m| m.id == w.nexus));
}

/// Never under a reply: while any chat has a reply being written, the call does nothing to the engine (a send switches it as
/// before). The reply is the one the engine is writing, so it goes on, and the engine is the same process until it has ended.
/// The model that was asked for is not lost (R76): a stopped reply releases it too, and the engine moves to it then.
#[tokio::test]
async fn preparing_a_model_while_a_reply_streams_does_nothing_until_the_reply_ends() {
    let long = "word ".repeat(60);
    let w = world("d9-busy", &[("FAKE_ENGINE_REPLY", long.as_str()), ("FAKE_ENGINE_TOKEN_MS", "50")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let task = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    for _ in 0..200 {
        if seen.lock().unwrap().iter().any(|e| matches!(e, StreamEvent::Token { .. })) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let before = w.svc.supervisor.status();
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let during = w.svc.supervisor.status();
    assert_eq!((during.state, during.model_id.as_deref(), during.pid), (EngineState::Serving, Some(w.qwen.as_str()), before.pid), "{during:?}");
    w.svc.stop(&chat);
    task.await.unwrap().unwrap();
    // R76: the reply is over, and the model that was asked for during it loads now.
    wait_serving(&w, &w.nexus, 10).await;
    assert_eq!(starts(&w), 2, "{}", events(&w));
}

/// ... also while the reply is still waiting for the engine to load its own model: that load is not taken over (the reply is
/// written by the model it waited for), and the model that was asked for meanwhile loads after the reply (R76).
#[tokio::test]
async fn preparing_a_model_while_a_reply_waits_for_its_own_model_waits_for_the_reply_too() {
    let w = world("d9-waiting", &[("FAKE_ENGINE_LOAD_MS", "600"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let task = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    for _ in 0..200 {
        if w.svc.supervisor.status().state == EngineState::Starting {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);
    task.await.unwrap().unwrap();
    let evs = seen.lock().unwrap().clone();
    assert!(matches!(evs.last(), Some(StreamEvent::Finished { .. })), "{evs:?}");
    wait_serving(&w, &w.nexus, 10).await;
    let argv: Vec<String> = events(&w).lines().filter(|l| l.starts_with("argv ")).map(|l| l.to_lowercase()).collect();
    assert_eq!(argv.len(), 2, "{}", events(&w));
    assert!(argv[0].contains("qwen3-4b") && argv[1].contains("nexus.gguf"), "the reply's model first, then the one asked for: {argv:?}");
}

/// The engine already serves the model, or is starting it, and no reply is under way: Ok, and nothing new is started. (Under a
/// reply the model is kept all the same, R77: see the tests at the end.)
#[tokio::test]
async fn preparing_a_model_that_is_serving_or_starting_starts_nothing() {
    let w = world("d9-there", &[("FAKE_ENGINE_LOAD_MS", "500"), ("FAKE_ENGINE_TOKEN_MS", "1")], 4096, fast());
    let starting = w.svc.start_model(&w.svc.library.read().unwrap().iter().find(|m| m.id == w.qwen).cloned().unwrap());
    for _ in 0..200 {
        if w.svc.supervisor.status().state == EngineState::Starting {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(w.svc.supervisor.status().state, EngineState::Starting);
    assert_eq!(w.svc.prepare_model(&w.qwen).unwrap(), Prepared::AlreadyThere, "while it is starting");
    starting.await.unwrap().unwrap();
    let pid = w.svc.supervisor.status().pid;
    assert_eq!(w.svc.prepare_model(&w.qwen).unwrap(), Prepared::AlreadyThere, "once it serves");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!((w.svc.supervisor.status().state, w.svc.supervisor.status().pid), (EngineState::Serving, pid));
    assert_eq!(starts(&w), 1, "{}", events(&w));
    assert!(!w.svc.monitor.logs_after(0).iter().any(|l| l.text.starts_with("preparing ")), "nothing was prepared");
}

/// A model that is not in the library, or cannot run, gives the error a send would give.
#[tokio::test]
async fn preparing_a_model_that_is_unknown_or_cannot_run_gives_the_sends_error() {
    let w = world("d9-refused", &[], 4096, fast());
    assert!(matches!(w.svc.prepare_model("no-such-model"), Err(ChatError::UnknownModel(id)) if id == "no-such-model"));
    {
        let mut lib = w.svc.library.write().unwrap();
        let nexus = lib.iter_mut().find(|m| m.id == w.nexus).unwrap();
        nexus.runnable = false;
        nexus.reason = Some("the engine refused it".into());
    }
    let err = w.svc.prepare_model(&w.nexus).unwrap_err();
    assert!(matches!(err, ChatError::NotRunnable(..)), "{err}");
    assert_eq!(err.to_string(), "nexus cannot run: the engine refused it");
    assert_eq!(starts(&w), 0, "no engine was started");
}

// ---------------------------------------------------------------------------------------------
// R76 (D9): a first key typed while a reply is being written is KEPT, not dropped. The reply is never interrupted for it; when the
// LAST reply has ended, the engine moves to the model that was asked for last, unless a send or a regenerate began meanwhile (they
// start their own model, so the user has moved on) or the engine has it by then. R77: the model asked for last is kept EVEN WHEN
// the engine already has it, which is how a first key in the streaming chat's own model takes the place of a model kept for another
// chat. Each test was run RED first, against a service that answers `ReplyInProgress` and forgets the model, except the negative
// ones (a send or a regenerate lets go of the kept model; a kept model the engine has is not started): "nothing is loaded" is true
// of that service too, so they are proved by mutation of the code they guard.
// ---------------------------------------------------------------------------------------------

/// What the engine says in these tests: twelve words, 60 ms apart, about 700 ms a reply, so that the calls of a test land in the
/// middle of it.
const TWELVE_WORDS: &str = "word word word word word word word word word word word word ";

fn slow_world(tag: &str) -> World {
    world(tag, &[("FAKE_ENGINE_REPLY", TWELVE_WORDS), ("FAKE_ENGINE_TOKEN_MS", "60")], 4096, fast())
}

/// Waits until the reply whose events `seen` collects has begun to stream.
async fn wait_for_a_token(seen: &Arc<Mutex<Vec<StreamEvent>>>) {
    for _ in 0..200 {
        if seen.lock().unwrap().iter().any(|e| matches!(e, StreamEvent::Token { .. })) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let evs = seen.lock().unwrap().clone();
    panic!("the reply never began to stream: {evs:?}");
}

/// The lines the engine was started with (its `argv` events), in order, lower-cased: one for each start, naming the model file.
fn started_on(w: &World) -> Vec<String> {
    events(w).lines().filter(|l| l.starts_with("argv ")).map(|l| l.to_lowercase()).collect()
}

/// One more model in the world's library, for the tests that need two models other than the one the engine serves: the world's
/// folder is scanned again with one more file in it. `tag` is the one `world()` was given.
fn add_a_model(w: &World, tag: &str, file_stem: &str) -> String {
    let dir = std::env::temp_dir().join(format!("lorren chat {}-{}", std::process::id(), tag));
    fixture::write(&dir.join(format!("{file_stem}.gguf")), &Spec::llama("Another_Model")).unwrap();
    let models = library::scan(&LibraryConfig {
        folders: vec![dir],
        ollama_models_dir: None,
        default_prefix: "qwen3-4b-instruct-2507-q4_k_m".into(),
    });
    let id = models.iter().find(|m| m.name == file_stem).unwrap().id.clone();
    *w.svc.library.write().unwrap() = models;
    id
}

/// The first key of a draft during a reply: the reply is left alone, the answer is `ReplyInProgress`, and the model is kept. When the
/// reply has ended, by itself and whole, the engine moves to that model, and the log says why.
#[tokio::test]
async fn a_model_asked_for_during_a_reply_loads_when_the_reply_ends() {
    let w = slow_world("r76-kept");
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let reply = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    wait_for_a_token(&seen).await;
    let before = w.svc.supervisor.status();
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);

    // The reply goes on, written by the engine it began on.
    tokio::time::sleep(Duration::from_millis(250)).await;
    let during = w.svc.supervisor.status();
    assert_eq!((during.state, during.model_id.as_deref(), during.pid), (EngineState::Serving, Some(w.qwen.as_str()), before.pid), "{during:?}");
    assert!(!w.svc.monitor.logs_after(0).iter().any(|l| l.text.starts_with("preparing ")), "nothing is prepared while the reply goes on");

    // It ends by itself, whole ...
    reply.await.unwrap().unwrap();
    let evs = seen.lock().unwrap().clone();
    assert!(matches!(evs.last(), Some(StreamEvent::Finished { .. })), "{evs:?}");
    let stored = last_reply(&w, &chat);
    assert_eq!((stored.status, stored.content.split_whitespace().count()), (MsgStatus::Done, 12));
    // ... and then the engine moves to the model that was asked for during it.
    wait_serving(&w, &w.nexus, 10).await;
    assert_eq!(starts(&w), 2, "{}", events(&w));
    let logs = w.svc.monitor.logs_after(0);
    let lines: Vec<_> = logs.iter().filter(|l| l.text.starts_with("preparing ")).collect();
    assert_eq!(lines.len(), 1, "{logs:?}");
    assert_eq!(
        (lines[0].source.as_str(), lines[0].level.as_str(), lines[0].text.as_str()),
        ("app", "info", "preparing nexus: the user typed during a reply")
    );
}

/// Several first keys during one reply: the model asked for last wins, and the others are not loaded before or after it.
#[tokio::test]
async fn the_model_asked_for_last_during_a_reply_is_the_one_that_loads() {
    let w = slow_world("r76-latest");
    let another = add_a_model(&w, "r76-latest", "another");
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let reply = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    wait_for_a_token(&seen).await;
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);
    assert_eq!(w.svc.prepare_model(&another).unwrap(), Prepared::ReplyInProgress);
    reply.await.unwrap().unwrap();
    wait_serving(&w, &another, 10).await;
    tokio::time::sleep(Duration::from_millis(600)).await; // time for a model kept by mistake to be started as well
    let s = w.svc.supervisor.status();
    assert_eq!((s.state, s.model_id.as_deref()), (EngineState::Serving, Some(another.as_str())), "{s:?}");
    let started = started_on(&w);
    assert_eq!(started.len(), 2, "{started:?}");
    assert!(started[0].contains("qwen3-4b") && started[1].contains("another.gguf"), "{started:?}");
}

/// A send, or a regenerate, starts the model of its own chat, so the model kept for after the replies is let go: the user has moved
/// on from it. Chats A and B are on the model the engine serves; a reply is being written in A; a first key asks for nexus; then a
/// reply begins in B. When both have ended the engine is the same process on the same model, and nexus was never started. (The same
/// flow without B's reply is the test above, where nexus does load.)
async fn a_reply_that_begins_lets_go_of_the_kept_model(tag: &str, regenerate: bool) {
    let w = slow_world(tag);
    let (a, b) = (new_chat(&w, &w.qwen), new_chat(&w, &w.qwen));
    if regenerate {
        let (_seen, sink) = collector();
        w.svc.send(&b, "hi", sink).await.unwrap(); // B has a reply to regenerate; this starts the engine
    }
    let (seen_a, sink_a) = collector();
    let (svc, a2) = (w.svc.clone(), a.clone());
    let first = tokio::spawn(async move { svc.send(&a2, "go", sink_a).await });
    wait_for_a_token(&seen_a).await;
    let before = w.svc.supervisor.status();
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);

    let (seen_b, sink_b) = collector();
    let (svc, b2) = (w.svc.clone(), b.clone());
    let second = tokio::spawn(async move {
        if regenerate {
            svc.regenerate(&b2, sink_b).await
        } else {
            svc.send(&b2, "and you?", sink_b).await
        }
    });
    wait_for_a_token(&seen_b).await;
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    for seen in [&seen_a, &seen_b] {
        let evs = seen.lock().unwrap().clone();
        assert!(matches!(evs.last(), Some(StreamEvent::Finished { .. })), "{evs:?}");
    }
    tokio::time::sleep(Duration::from_millis(800)).await; // time for the model that was kept to be started, if it had been
    let after = w.svc.supervisor.status();
    assert_eq!((after.state, after.model_id.as_deref(), after.pid), (EngineState::Serving, Some(w.qwen.as_str()), before.pid), "{after:?}");
    assert_eq!(starts(&w), 1, "{}", events(&w));
    assert!(!w.svc.monitor.logs_after(0).iter().any(|l| l.text.starts_with("preparing ")), "nothing was prepared");
}

#[tokio::test]
async fn a_send_during_a_reply_lets_go_of_the_model_kept_for_after_it() {
    a_reply_that_begins_lets_go_of_the_kept_model("r76-send", false).await;
}

#[tokio::test]
async fn a_regenerate_during_a_reply_lets_go_of_the_model_kept_for_after_it() {
    a_reply_that_begins_lets_go_of_the_kept_model("r76-regen", true).await;
}

/// The kept model is loaded only if the engine does not have it by then. The test after this one has the engine on the kept model all
/// along (R77); here the engine GETS it meanwhile: something else moves it to the model during the reply (directly; the reply under way
/// is cut short by that). When the reply has ended there is nothing to start, and no line says there was.
#[tokio::test]
async fn a_kept_model_the_engine_already_has_is_not_started_again() {
    let w = slow_world("r76-there");
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let reply = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    wait_for_a_token(&seen).await;
    assert_eq!(w.svc.prepare_model(&w.nexus).unwrap(), Prepared::ReplyInProgress);
    let nexus = w.svc.library.read().unwrap().iter().find(|m| m.id == w.nexus).cloned().unwrap();
    w.svc.start_model(&nexus).await.unwrap().unwrap();
    reply.await.unwrap().unwrap();
    let evs = seen.lock().unwrap().clone();
    assert!(matches!(evs.last(), Some(StreamEvent::Interrupted { .. })), "the switch cut the reply short: {evs:?}");
    tokio::time::sleep(Duration::from_millis(600)).await; // time for a second start of nexus, if one were made
    let s = w.svc.supervisor.status();
    assert_eq!((s.state, s.model_id.as_deref()), (EngineState::Serving, Some(w.nexus.as_str())), "{s:?}");
    assert_eq!(starts(&w), 2, "{}", events(&w));
    assert!(!w.svc.monitor.logs_after(0).iter().any(|l| l.text.starts_with("preparing ")), "nothing was prepared: the engine had the model");
}

/// R77: the model asked for last during a reply is kept EVEN WHEN the engine already has it. Only the host knows that a reply is under
/// way in some other chat, so a first key in the model of the chat that is streaming has to reach it as a kept model: it takes the
/// place of the one kept for another chat's draft, which would otherwise be loaded at the end of the reply, with the user back in
/// the streaming chat, and the next send there would switch the engine back. Here: a reply in chat A on qwen; a first key in a chat
/// on nexus; then a first key in a chat on qwen. When the reply has ended the engine is the same process on qwen, nothing was started
/// and no line says so. With no reply under way the same call is answered `AlreadyThere`, as before.
#[tokio::test]
async fn a_first_key_in_the_streaming_chats_own_model_takes_the_place_of_the_model_kept_for_another_chat() {
    let w = slow_world("r77-own");
    let chat = new_chat(&w, &w.qwen);
    let (seen, sink) = collector();
    let (svc, c2) = (w.svc.clone(), chat.clone());
    let reply = tokio::spawn(async move { svc.send(&c2, "go", sink).await });
    wait_for_a_token(&seen).await;
    let before = w.svc.supervisor.status();
    let other = w.svc.prepare_model(&w.nexus).unwrap(); // a first key in a chat on nexus
    let own = w.svc.prepare_model(&w.qwen).unwrap(); // ... then a first key in a chat on the model that is being served
    reply.await.unwrap().unwrap();
    let evs = seen.lock().unwrap().clone();
    assert!(matches!(evs.last(), Some(StreamEvent::Finished { .. })), "{evs:?}");
    tokio::time::sleep(Duration::from_millis(800)).await; // time for the model kept for the other chat to be started, if it were

    // What the user would have seen: the engine did not move. (Asserted before the answers, so that a failure shows this first.)
    let after = w.svc.supervisor.status();
    assert_eq!((after.state, after.model_id.as_deref(), after.pid), (EngineState::Serving, Some(w.qwen.as_str()), before.pid), "{after:?}");
    assert_eq!(starts(&w), 1, "{}", events(&w));
    assert!(!w.svc.monitor.logs_after(0).iter().any(|l| l.text.starts_with("preparing ")), "nothing was prepared");
    // The answers: both calls were made during the reply, so both kept their model, the second although the engine had it.
    assert_eq!((other, own), (Prepared::ReplyInProgress, Prepared::ReplyInProgress));
    // With the reply over, the same call is answered as it always was.
    assert_eq!(w.svc.prepare_model(&w.qwen).unwrap(), Prepared::AlreadyThere);
}
