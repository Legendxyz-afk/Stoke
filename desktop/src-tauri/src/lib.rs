pub mod budget;
pub mod chat;
pub mod commands;
pub mod engine;
pub mod gguf;
pub mod library;
pub mod monitor;
pub mod paths;
pub mod sse;
pub mod store;
pub mod updater;

use commands::AppState;
use std::sync::{Arc, Mutex, RwLock};
use tauri::{Emitter, Manager};

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                // `show` first (R64d): tao's `set_focus` does nothing for a window that is not visible, and a failed
                // update install leaves the window hidden.
                let _ = w.show();
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let monitor = monitor::Monitor::new(Some(paths::log_dir().join("lorren.log")));
            monitor.log("app", "info", format!("Lorren {} starting", app.package_info().version));
            std::fs::create_dir_all(paths::data_dir())?;
            let store = store::Store::open(&paths::data_dir().join("lorren.db"))?;
            let models = library::scan(&library::LibraryConfig::from_env(paths::downloads_dir()));
            monitor.log(
                "app",
                "info",
                format!("library: {} models, {} runnable", models.len(), models.iter().filter(|m| m.runnable).count()),
            );
            let default_model = models.iter().find(|m| m.is_default).cloned();
            let library = Arc::new(RwLock::new(models));
            let supervisor = engine::supervisor::Supervisor::new(
                engine::supervisor::EngineConfig::defaults(paths::engine_exe()),
                monitor.clone(),
            );
            let store = Arc::new(Mutex::new(store));
            let chat = chat::ChatService::new(store.clone(), supervisor.clone(), library.clone(), monitor.clone(), chat::ChatSettings::default());
            // `AppState` takes `chat`; the startup load below keeps its own handle on the service (R38/R39).
            let startup_chat = chat.clone();
            let updates = updater::UpdateService::new(monitor.clone());
            app.manage(AppState { store, library, monitor, supervisor: supervisor.clone(), chat, updates: updates.clone(), started_ms: store::now_ms() });

            // Updates (spec §3.7): a check 5 s after start, then every 6 h. Without an endpoint the first check says
            // "not configured" and the loop ends.
            let (updates_bg, handle_bg) = (updates.clone(), app.handle().clone());
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                loop {
                    updates_bg.check(&handle_bg).await;
                    if updates_bg.state() == updater::UpdateState::NotConfigured {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(6 * 3600)).await;
                }
            });

            let handle = app.handle().clone();
            let mut rx = supervisor.subscribe();
            tauri::async_runtime::spawn(async move {
                while rx.changed().await.is_ok() {
                    let status = rx.borrow().clone();
                    let _ = handle.emit("engine-status", status);
                }
            });
            // Start the engine on the default model so the first reply is quick. Through `start_model`, the app's
            // only way to start or switch the engine: it marks a model the engine refuses as CAN'T RUN. The task
            // wrapper stays, because `start_model` uses `tokio::spawn` and this closure runs outside a runtime.
            if let Some(m) = default_model {
                tauri::async_runtime::spawn(async move {
                    let _ = startup_chat.start_model(&m).await;
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::overview,
            commands::models,
            commands::chats,
            commands::chat,
            commands::new_chat,
            commands::send,
            commands::regenerate,
            commands::stop,
            commands::set_chat_model,
            commands::select_version,
            commands::requests,
            commands::logs,
            commands::update_state,
            commands::check_update,
            commands::install_update
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lorren");
}
