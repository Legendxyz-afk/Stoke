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
    assert!(matches!(seen.first(), Some(StreamEvent::Started { dropped_turns: 0, .. })));
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
        if w.svc.supervisor.status().state == EngineState::Serving {
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
    assert!(matches!(seen.first(), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns > 0), "{seen:?}");
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
    assert!(matches!(seen.lock().unwrap().first(), Some(StreamEvent::Started { dropped_turns, .. }) if *dropped_turns >= 2));
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
