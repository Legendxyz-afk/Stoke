//! The Tauri commands the UI calls (spec §3.8): thin shells over the services.

use crate::chat::{ChatService, StreamEvent};
use crate::engine::memory;
use crate::engine::supervisor::{EngineStatus, Supervisor};
use crate::library::ModelInfo;
use crate::monitor::{LogLine, Monitor, RequestRecord};
use crate::store::{Chat, Message, Store};
use serde::Serialize;
use std::sync::{Arc, Mutex, RwLock};
use tauri::ipc::Channel;
use tauri::{Manager, State};
use ts_rs::TS;

pub struct AppState {
    pub store: Arc<Mutex<Store>>,
    pub library: Arc<RwLock<Vec<ModelInfo>>>,
    pub monitor: Arc<Monitor>,
    pub supervisor: Arc<Supervisor>,
    pub chat: Arc<ChatService>,
    pub updates: Arc<crate::updater::UpdateService>,
    pub started_ms: i64,
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
pub fn models(state: State<'_, AppState>) -> Vec<ModelInfo> {
    state.library.read().unwrap().clone()
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

/// Switches the chat's model and starts the engine on it right away, so the switch is visible. The start goes through
/// `ChatService::start_model` (R38/R39), the app's only way to start or switch the engine: it marks a model the engine
/// refuses as CAN'T RUN and keeps the load running when this command's handle is dropped. The command is `async` so
/// it runs inside the tokio runtime that `start_model` spawns into.
#[tauri::command]
pub async fn set_chat_model(state: State<'_, AppState>, chat_id: String, model_id: String) -> Result<(), String> {
    let model = state.library.read().unwrap().iter().find(|m| m.id == model_id).cloned().ok_or("no such model")?;
    if !model.runnable {
        return Err(crate::chat::cannot_run_message(&model.name, model.reason.as_deref().unwrap_or_default()));
    }
    state.store.lock().unwrap().set_chat_model(&chat_id, &model_id).map_err(err)?;
    drop(state.chat.start_model(&model));
    Ok(())
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
