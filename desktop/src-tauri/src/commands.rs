//! The Tauri commands the UI calls (spec §3.8): thin shells over the services.

use crate::chat::{ChatService, StreamEvent};
use crate::engine::memory;
use crate::engine::supervisor::{EngineStatus, Supervisor};
use crate::library::ModelInfo;
use crate::monitor::{LogLine, Monitor, RequestRecord};
use crate::store::{Chat, Message, Store};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, RwLock};
use tauri::ipc::Channel;
use tauri::{Manager, State};
use ts_rs::TS;

pub struct AppState {
    pub store: Arc<Mutex<Store>>,
    pub library: Arc<RwLock<Vec<ModelInfo>>>,
    /// Turns `true` once the scan thread has written the library; `models` waits on it.
    pub library_ready: tokio::sync::watch::Receiver<bool>,
    pub monitor: Arc<Monitor>,
    pub supervisor: Arc<Supervisor>,
    pub chat: Arc<ChatService>,
    pub updates: Arc<crate::updater::UpdateService>,
    pub catalog: Arc<crate::catalog::CatalogService>,
    pub downloads: Arc<crate::downloads::DownloadManager>,
    pub started_ms: i64,
    /// Epoch ms at the top of `run()`: what the opening is measured from.
    pub launched_ms: i64,
    /// Epoch ms right after the window was shown.
    pub shown_ms: i64,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct MemoryInfo {
    /// Lorren plus its WebView2 processes (working set).
    #[ts(type = "number")]
    pub app_bytes: u64,
    #[ts(type = "number")]
    pub engine_bytes: u64,
    #[ts(type = "number")]
    pub used_bytes: u64,
    #[ts(type = "number")]
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Overview {
    pub engine: EngineStatus,
    pub memory: MemoryInfo,
    pub app_version: String,
    #[ts(type = "number")]
    pub started_ms: i64,
    pub request_count: u32,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct MessageView {
    pub message: Message,
    /// 1-based position among its siblings (‹ version/versions ›).
    pub version: u32,
    pub versions: u32,
    pub sibling_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ChatView {
    pub chat: Chat,
    pub messages: Vec<MessageView>,
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub fn memory_info(engine_pid: Option<u32>) -> MemoryInfo {
    let total = memory::total_physical_bytes();
    MemoryInfo {
        app_bytes: memory::tree_working_set(std::process::id(), engine_pid),
        engine_bytes: engine_pid.and_then(memory::process_memory).map(|(ws, _)| ws).unwrap_or(0),
        used_bytes: total.saturating_sub(memory::available_physical_bytes()),
        total_bytes: total,
    }
}

pub fn chat_view(store: &Store, chat_id: &str) -> rusqlite::Result<ChatView> {
    let chat = store.get_chat(chat_id)?;
    let mut messages = Vec::new();
    for m in store.active_path(chat_id)? {
        let sibs = store.siblings(&m.id)?;
        let version = sibs.iter().position(|s| s.id == m.id).map(|i| i as u32 + 1).unwrap_or(1);
        messages.push(MessageView {
            versions: sibs.len() as u32,
            version,
            sibling_ids: sibs.into_iter().map(|s| s.id).collect(),
            message: m,
        });
    }
    Ok(ChatView { chat, messages })
}

/// The version comes from `tauri.conf.json` (which a 0.0.2 build overrides), not from Cargo.toml.
#[tauri::command]
pub fn overview(app: tauri::AppHandle, state: State<'_, AppState>) -> Overview {
    let engine = state.supervisor.status();
    Overview {
        memory: memory_info(engine.pid),
        engine,
        app_version: app.package_info().version.to_string(),
        started_ms: state.started_ms,
        // Requests started this session; the list the Requests tab shows holds only the newest 500.
        request_count: u32::try_from(state.monitor.request_count()).unwrap_or(u32::MAX),
    }
}

#[tauri::command]
pub async fn models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    debug_hold(state.launched_ms).await;
    let mut rx = state.library_ready.clone();
    wait_library(&mut rx).await;
    Ok(state.library.read().unwrap().clone())
}

/// Returns once the library scan has been published (at once if it already has), or when the scan's sender is gone.
pub async fn wait_library(rx: &mut tokio::sync::watch::Receiver<bool>) {
    let _ = rx.wait_for(|ready| *ready).await;
}

/// How long `models` still has to wait, counted from launch: `max(0, launched + hold - now)`; 0 without a hold.
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub fn hold_left(launched_ms: i64, hold_ms: Option<u64>, now_ms: i64) -> u64 {
    match hold_ms {
        Some(h) => u64::try_from((launched_ms + i64::try_from(h).unwrap_or(i64::MAX / 2) - now_ms).max(0)).unwrap_or(0),
        None => 0,
    }
}

/// Debug builds only: the e2e sets `LORREN_DEBUG_HOLD_MODELS_MS` to make the host look slow to answer `models`.
#[cfg(debug_assertions)]
async fn debug_hold(launched_ms: i64) {
    let hold = std::env::var("LORREN_DEBUG_HOLD_MODELS_MS").ok().and_then(|v| v.trim().parse::<u64>().ok());
    let left = hold_left(launched_ms, hold, crate::store::now_ms());
    if left > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(left)).await;
    }
}

#[cfg(not(debug_assertions))]
async fn debug_hold(_launched_ms: i64) {}

/// What the page's opening measured (`public/opening.js`), field for field, as sent.
#[derive(Debug, Clone, Deserialize)]
pub struct OpeningReport {
    pub navigation_ms: f64,
    pub first_frame_ms: Option<f64>,
    pub ready_ms: Option<f64>,
    pub finished_ms: Option<f64>,
    pub handover_start_ms: f64,
    pub handover_end_ms: f64,
    pub plan_ms: f64,
    pub remembered_ms: Option<f64>,
    pub longest_gap_ms: f64,
    pub longest_task_ms: Option<f64>,
    pub longest_task_at_ms: Option<f64>,
    pub landing_error_px: Option<f64>,
    pub font_ms: Option<f64>,
    pub reduced_motion: bool,
    pub valve: bool,
    pub error: Option<String>,
}

/// The one log line for an opening (spec 2026-10-23 section 7): (level, text).
pub fn opening_line(r: &OpeningReport, launched_ms: i64, shown_ms: i64) -> (&'static str, String) {
    let l = launched_ms as f64;
    let first = match r.first_frame_ms {
        Some(v) => format!("first frame {:.0} ms", v - l),
        None => "first frame never".to_string(),
    };
    let ready = match r.ready_ms {
        Some(v) => format!("ready {:.0} ms", v - l),
        None => "ready never came (the 12 s valve)".to_string(),
    };
    let start = match r.remembered_ms {
        Some(v) => format!("last start {:.1} s", v / 1000.0),
        None => "first start".to_string(),
    };
    let mut text = format!(
        "opening: window {} ms, page {:.0} ms, {first}, {ready}, in the top bar {:.0} ms after launch; planned {:.1} s ({start}); longest frame gap {:.0} ms",
        shown_ms - launched_ms,
        r.navigation_ms - l,
        r.handover_end_ms - l,
        r.plan_ms / 1000.0,
        r.longest_gap_ms,
    );
    if let (Some(t), Some(at)) = (r.longest_task_ms, r.longest_task_at_ms) {
        text.push_str(&format!("; longest task {t:.0} ms at {at:.0} ms"));
    }
    if let Some(e) = r.landing_error_px {
        text.push_str(&format!("; landed within {e:.2} px"));
    } else if r.reduced_motion {
        text.push_str("; reduced motion: faded");
    } else {
        text.push_str("; no wordmark to land on: faded");
    }
    if let Some(e) = &r.error {
        text.push_str(&format!("; the opening failed: {e}"));
    }
    (if r.valve || r.error.is_some() { "warn" } else { "info" }, text)
}

#[tauri::command]
pub fn opening_report(state: State<'_, AppState>, report: OpeningReport) {
    let (level, text) = opening_line(&report, state.launched_ms, state.shown_ms);
    state.monitor.log("app", level, text);
}

#[tauri::command]
pub fn chats(state: State<'_, AppState>) -> Result<Vec<Chat>, String> {
    state.store.lock().unwrap().list_chats().map_err(err)
}

#[tauri::command]
pub fn chat(state: State<'_, AppState>, chat_id: String) -> Result<ChatView, String> {
    chat_view(&state.store.lock().unwrap(), &chat_id).map_err(err)
}

/// The model a new chat is bound to (R50). With an id, that model: it has to exist and be runnable, and is never
/// swapped for another. Without one, the default if it can run, else the first model that can; `is_default` stays on a
/// model the engine refused (R38 marks it CAN'T RUN), so it cannot be taken on its own word.
pub fn pick_model(models: &[ModelInfo], id: Option<&str>) -> Result<ModelInfo, String> {
    if let Some(id) = id {
        let m = models.iter().find(|m| m.id == id).ok_or("no such model")?;
        if !m.runnable {
            return Err(crate::chat::cannot_run_message(&m.name, m.reason.as_deref().unwrap_or_default()));
        }
        return Ok(m.clone());
    }
    models
        .iter()
        .find(|m| m.is_default && m.runnable)
        .or_else(|| models.iter().find(|m| m.runnable))
        .cloned()
        .ok_or_else(|| "there is no runnable model in the library".to_string())
}

#[tauri::command]
pub fn new_chat(state: State<'_, AppState>, model_id: Option<String>) -> Result<Chat, String> {
    let model = pick_model(&state.library.read().unwrap(), model_id.as_deref())?;
    state.store.lock().unwrap().create_chat(&model.id, None).map_err(err)
}

#[tauri::command]
pub async fn send(state: State<'_, AppState>, chat_id: String, text: String, on_event: Channel<StreamEvent>) -> Result<(), String> {
    state.chat.send(&chat_id, &text, move |e| {
        let _ = on_event.send(e);
    })
    .await
    .map_err(err)
}

#[tauri::command]
pub async fn regenerate(state: State<'_, AppState>, chat_id: String, on_event: Channel<StreamEvent>) -> Result<(), String> {
    state.chat.regenerate(&chat_id, move |e| {
        let _ = on_event.send(e);
    })
    .await
    .map_err(err)
}

#[tauri::command]
pub fn stop(state: State<'_, AppState>, chat_id: String) {
    state.chat.stop(&chat_id);
}

/// Stores the chat's model, and nothing more (demo D9). It used to start the engine on it at once, so picking a model in the
/// panel loaded it (the demo's log shows the switch 20 s before the next send); now the engine moves to a chat's model only
/// when the user starts typing in the chat (`ChatService::prepare_model`) or sends. The model has to exist and be able to run,
/// as for a send.
pub fn switch_chat_model(svc: &ChatService, chat_id: &str, model_id: &str) -> Result<(), String> {
    let model = svc.library.read().unwrap().iter().find(|m| m.id == model_id).cloned().ok_or("no such model")?;
    if !model.runnable {
        return Err(crate::chat::cannot_run_message(&model.name, model.reason.as_deref().unwrap_or_default()));
    }
    svc.store.lock().unwrap().set_chat_model(chat_id, model_id).map_err(err)
}

#[tauri::command]
pub fn set_chat_model(state: State<'_, AppState>, chat_id: String, model_id: String) -> Result<(), String> {
    switch_chat_model(&state.chat, &chat_id, &model_id)
}

/// The user has started typing in a chat (or in the box that makes one): see `ChatService::prepare_model`. `async` so that it
/// runs inside the tokio runtime that `start_model` spawns into.
#[tauri::command]
pub async fn prepare_model(state: State<'_, AppState>, model_id: String) -> Result<(), String> {
    state.chat.prepare_model(&model_id).map(|_| ()).map_err(err)
}

#[tauri::command]
pub fn select_version(state: State<'_, AppState>, chat_id: String, message_id: String) -> Result<(), String> {
    state.store.lock().unwrap().set_active_leaf(&chat_id, &message_id).map_err(err)
}

#[tauri::command]
pub fn requests(state: State<'_, AppState>) -> Vec<RequestRecord> {
    state.monitor.requests()
}

#[tauri::command]
pub fn logs(state: State<'_, AppState>, after: u64) -> Vec<LogLine> {
    state.monitor.logs_after(after)
}

#[tauri::command]
pub fn update_state(state: State<'_, AppState>) -> crate::updater::UpdateState {
    state.updates.state()
}

#[tauri::command]
pub async fn check_update(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let updates = state.updates.clone();
    updates.check(&app).await;
    Ok(())
}

#[tauri::command]
pub async fn install_update(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    // R64: with no verified update waiting there is nothing to install, so say so and leave the engine running.
    if state.updates.ready_version().is_none() {
        return Err(crate::updater::NO_VERIFIED_UPDATE.to_string());
    }
    // R37: the sidecar must not be running when NSIS replaces it.
    state.supervisor.stop().await;
    let result = state.updates.install();
    if result.is_err() {
        // R64d: the plugin's pre-install hook (cleanup_before_exit) hid every window; show it again so the notice can be read.
        if let Some(w) = app.get_webview_window("main") {
            let _ = w.show();
            let _ = w.unminimize();
            let _ = w.set_focus();
        }
    }
    result
}

#[cfg(test)]
mod library_gate_tests {
    use super::wait_library;
    use std::time::Duration;
    use tokio::sync::watch;
    use tokio::time::timeout;

    #[tokio::test]
    async fn wait_library_returns_once_the_scan_is_published() {
        let (tx, mut rx) = watch::channel(false);
        let task = tokio::spawn(async move { wait_library(&mut rx).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished(), "returned before the scan was published");
        tx.send_replace(true);
        timeout(Duration::from_secs(1), task).await.expect("did not return after the publish").unwrap();
    }

    #[tokio::test]
    async fn wait_library_returns_at_once_when_already_published() {
        let (_tx, mut rx) = watch::channel(true);
        timeout(Duration::from_millis(50), wait_library(&mut rx)).await.expect("waited though the library was ready");
    }

    #[tokio::test]
    async fn wait_library_returns_when_the_scan_is_gone() {
        let (tx, mut rx) = watch::channel(false);
        let task = tokio::spawn(async move { wait_library(&mut rx).await });
        drop(tx);
        timeout(Duration::from_secs(1), task).await.expect("did not return after the sender was dropped").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{MsgStatus, Role};

    #[test]
    fn a_regenerated_reply_shows_as_version_two_of_two() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        let u = s.append_message(&chat.id, None, Role::User, "hi", MsgStatus::Done, None).unwrap();
        let a = s.append_message(&chat.id, Some(&u.id), Role::Assistant, "one", MsgStatus::Done, None).unwrap();
        let b = s.append_message(&chat.id, Some(&u.id), Role::Assistant, "two", MsgStatus::Done, None).unwrap();
        let v = chat_view(&s, &chat.id).unwrap();
        assert_eq!(v.messages.len(), 2);
        let last = &v.messages[1];
        assert_eq!((last.version, last.versions), (2, 2));
        assert_eq!(last.sibling_ids, vec![a.id, b.id]);
        assert_eq!(v.messages[0].versions, 1);
    }

    #[test]
    fn memory_info_counts_this_process() {
        let m = memory_info(None);
        assert!(m.app_bytes > 0 && m.total_bytes > m.used_bytes && m.engine_bytes == 0);
    }

    /// A library entry with only the fields `pick_model` reads set.
    fn entry(id: &str, runnable: bool, is_default: bool) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            name: format!("{id}-name"),
            path: format!("M:/models/{id}.gguf"),
            source: "folder".into(),
            size_bytes: 1,
            arch: "llama".into(),
            quant: "Q4_K_M".into(),
            size_label: String::new(),
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 32,
            vocab: 1,
            ctx_file: 4096,
            ctx_engine: 4096,
            kv_bytes_per_token: 1,
            run_bytes: 1,
            template: "chatml".into(),
            runnable,
            reason: (!runnable).then(|| format!("{id} was refused")),
            system_prompt: None,
            system_source: None,
            is_default,
        }
    }

    /// R50: an explicit id is never swapped for the default. A model that cannot run is refused with the same
    /// words a send and a model switch use; one that is not in the library is "no such model".
    #[test]
    fn an_explicit_model_that_cannot_run_is_refused_not_swapped_for_the_default() {
        let lib = [entry("default", true, true), entry("broken", false, false), entry("other", true, false)];
        let err = pick_model(&lib, Some("broken")).unwrap_err();
        assert_eq!(err, crate::chat::cannot_run_message("broken-name", "broken was refused"));
        assert_eq!(pick_model(&lib, Some("missing")).unwrap_err(), "no such model");
        assert_eq!(pick_model(&lib, Some("other")).unwrap().id, "other");
    }

    /// R50: after the engine refuses the default (R38 marks it CAN'T RUN, `is_default` stays), "+ NEW" binds to
    /// the first model that can run, not to the dead default.
    #[test]
    fn a_default_that_cannot_run_gives_way_to_the_first_runnable_model() {
        let lib = [entry("default", false, true), entry("broken", false, false), entry("first", true, false), entry("second", true, false)];
        assert_eq!(pick_model(&lib, None).unwrap().id, "first");
        // A default that can run is taken even when it is not first.
        let lib = [entry("first", true, false), entry("default", true, true)];
        assert_eq!(pick_model(&lib, None).unwrap().id, "default");
    }

    #[test]
    fn with_nothing_runnable_there_is_no_model_to_pick() {
        let lib = [entry("default", false, true), entry("broken", false, false)];
        assert_eq!(pick_model(&lib, None).unwrap_err(), "there is no runnable model in the library");
        assert_eq!(pick_model(&[], None).unwrap_err(), "there is no runnable model in the library");
    }
}

#[cfg(test)]
mod opening_tests {
    use super::*;

    fn normal() -> OpeningReport {
        OpeningReport {
            navigation_ms: 1_000_380.4,
            first_frame_ms: Some(1_000_455.2),
            ready_ms: Some(1_001_430.7),
            finished_ms: Some(1_002_455.0),
            handover_start_ms: 1_002_756.0,
            handover_end_ms: 1_003_515.9,
            plan_ms: 2000.0,
            remembered_ms: None,
            longest_gap_ms: 18.4,
            longest_task_ms: Some(140.2),
            longest_task_at_ms: Some(80.0),
            landing_error_px: Some(0.0213),
            font_ms: Some(60.0),
            reduced_motion: false,
            valve: false,
            error: None,
        }
    }

    const HEAD: &str = "opening: window 12 ms, page 380 ms, first frame 455 ms, ready 1431 ms, in the top bar 3516 ms after launch; planned 2.0 s (first start); longest frame gap 18 ms";

    #[test]
    fn opening_line_reads_a_normal_start() {
        let (level, text) = opening_line(&normal(), 1_000_000, 1_000_012);
        assert_eq!(level, "info");
        assert_eq!(text, format!("{HEAD}; longest task 140 ms at 80 ms; landed within 0.02 px"));
    }

    #[test]
    fn opening_line_reads_the_valve() {
        let mut r = normal();
        r.ready_ms = None;
        r.valve = true;
        r.plan_ms = 5000.0;
        r.remembered_ms = Some(7200.0);
        r.handover_end_ms = 1_013_080.0;
        r.longest_task_ms = None;
        r.landing_error_px = Some(0.03);
        let (level, text) = opening_line(&r, 1_000_000, 1_000_012);
        assert_eq!(level, "warn");
        assert_eq!(text, "opening: window 12 ms, page 380 ms, first frame 455 ms, ready never came (the 12 s valve), in the top bar 13080 ms after launch; planned 5.0 s (last start 7.2 s); longest frame gap 18 ms; landed within 0.03 px");
    }

    #[test]
    fn opening_line_reads_reduced_motion() {
        let mut r = normal();
        r.landing_error_px = None;
        r.reduced_motion = true;
        r.longest_task_ms = None;
        let (level, text) = opening_line(&r, 1_000_000, 1_000_012);
        assert_eq!(level, "info");
        assert_eq!(text, format!("{HEAD}; reduced motion: faded"));
    }

    #[test]
    fn opening_line_reads_a_failure() {
        let mut r = normal();
        r.landing_error_px = None;
        r.longest_task_ms = None;
        r.error = Some("TypeError: x is undefined".into());
        let (level, text) = opening_line(&r, 1_000_000, 1_000_012);
        assert_eq!(level, "warn");
        assert_eq!(text, format!("{HEAD}; no wordmark to land on: faded; the opening failed: TypeError: x is undefined"));
    }

    #[test]
    fn hold_left_counts_from_launch() {
        assert_eq!(hold_left(1000, None, 1500), 0);
        assert_eq!(hold_left(1000, Some(6000), 1500), 5500);
        assert_eq!(hold_left(1000, Some(6000), 9000), 0);
    }
}

/// What Discover shows now: memory, else the saved copy verified again. Makes no request.
#[tauri::command]
pub fn catalog(state: State<'_, AppState>) -> crate::catalog::CatalogView {
    state.catalog.view()
}

/// Reads the catalog from its address (Discover opening, or the user asking again).
#[tauri::command]
pub async fn refresh_catalog(state: State<'_, AppState>) -> Result<crate::catalog::CatalogView, String> {
    Ok(state.catalog.refresh().await)
}

/// Opens a VERIFIED catalog entry's licence page in the default browser (§6: links open only when clicked). The UI cannot
/// name an address: it names an entry, and the address is the one the signed catalog holds.
#[tauri::command]
pub fn open_licence(app: tauri::AppHandle, state: State<'_, AppState>, entry_id: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let entry = state.catalog.entry(&entry_id).ok_or_else(|| format!("no such model in the catalog: {entry_id}"))?;
    app.opener().open_url(entry.licence.url, None::<&str>).map_err(err)
}

/// Every download the manager knows: running, paused, failed, and the done ones (Discover shows them as on disk).
#[tauri::command]
pub fn downloads(state: State<'_, AppState>) -> Vec<crate::downloads::DownloadView> {
    state.downloads.list()
}

/// The downloads folder and the room on its disk, for the size line before a download.
#[tauri::command]
pub fn disk_room(state: State<'_, AppState>) -> crate::downloads::DiskRoom {
    crate::downloads::disk_room(state.downloads.dir())
}

/// Starts (or resumes) a download of a VERIFIED catalog entry; the UI names the entry, never an address.
#[tauri::command]
pub fn start_download(state: State<'_, AppState>, entry_id: String) -> Result<crate::downloads::DownloadView, String> {
    let entry = state.catalog.entry(&entry_id).ok_or_else(|| format!("no such model in the catalog: {entry_id}"))?;
    if state.catalog.view().needs_app.contains(&entry_id) {
        return Err(format!("{} needs a newer Lorren", entry.name));
    }
    state.downloads.start(&entry)
}

#[tauri::command]
pub fn pause_download(state: State<'_, AppState>, entry_id: String) -> Result<(), String> {
    state.downloads.pause(&entry_id)
}

#[tauri::command]
pub fn resume_download(state: State<'_, AppState>, entry_id: String) -> Result<(), String> {
    state.downloads.resume(&entry_id)
}

#[tauri::command]
pub fn cancel_download(state: State<'_, AppState>, entry_id: String) -> Result<(), String> {
    state.downloads.cancel(&entry_id)
}
