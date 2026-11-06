pub mod budget;
pub mod catalog;
pub mod chat;
pub mod commands;
pub mod downloads;
pub mod engine;
pub mod gguf;
pub mod library;
pub mod monitor;
pub mod paths;
pub mod sse;
pub mod store;
pub mod updater;
pub mod window_state;

use commands::AppState;
use std::sync::{Arc, Mutex, RwLock};
use tauri::{Emitter, Manager};

/// The data folder and the database in it. A failure here ends the setup, and with it the app before any window: the log
/// file is the only place that can say why, so the reason is written there before the error is handed back.
fn open_store(monitor: &monitor::Monitor, dir: &std::path::Path) -> Result<store::Store, Box<dyn std::error::Error>> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        monitor.log("app", "error", format!("startup failed: {e}"));
        return Err(e.into());
    }
    match store::Store::open(&dir.join("lorren.db")) {
        Ok(store) => Ok(store),
        Err(e) => {
            monitor.log("app", "error", format!("startup failed: {e}"));
            Err(e.into())
        }
    }
}

pub fn run() {
    let launched_ms = store::now_ms();
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
        // Opens a catalog entry's licence page in the browser; the plugin's JS API is not granted to the UI (capabilities).
        .plugin(tauri_plugin_opener::init())
        // The main window's state is saved when it is asked to close (demo D8).
        .on_window_event(window_state::on_window_event)
        .setup(move |app| {
            // First of all, before the store (the library scan runs on a thread of its own after it): the window is
            // created hidden (tauri.conf.json), placed as the last run left it (maximized the first time), and shown,
            // so it appears as early as it can (demo D8).
            let window_note = window_state::restore(app);
            let shown_ms = store::now_ms();
            let monitor = monitor::Monitor::new(Some(paths::log_dir().join("lorren.log")));
            monitor.log("app", "info", format!("Lorren {} starting", app.package_info().version));
            monitor.log("app", "info", window_note);
            let store = open_store(&monitor, &paths::data_dir())?;
            // The library starts empty and is filled by the scan thread below; `models` waits on `library_ready`.
            let library = Arc::new(RwLock::new(Vec::new()));
            let (library_tx, library_ready) = tokio::sync::watch::channel(false);
            let supervisor = engine::supervisor::Supervisor::new(
                engine::supervisor::EngineConfig::defaults(paths::engine_exe()),
                monitor.clone(),
            );
            let store = Arc::new(Mutex::new(store));
            let chat = chat::ChatService::new(store.clone(), supervisor.clone(), library.clone(), monitor.clone(), chat::ChatSettings::default());
            // `AppState` takes `chat`; the startup load after the scan keeps its own handle on the service (R38/R39).
            let startup_chat = chat.clone();
            // No endpoint in the build's config (every milestone-1 build but the test-feed ones): the service starts
            // NotConfigured, so the badge shows nothing from the first frame, and no check is ever scheduled.
            let configured = updater::endpoints_configured(app.config().plugins.0.get("updater"));
            let updates = updater::UpdateService::new(monitor.clone(), configured);
            // The catalog is read only when Discover opens (`refresh_catalog`): nothing here touches the network.
            let catalog_source = catalog::source_from_config(app.config().plugins.0.get(catalog::CATALOG_PLUGIN));
            let catalog_insecure = catalog_source.as_ref().map(|s| s.insecure).unwrap_or(false);
            let app_version = semver::Version::parse(&app.package_info().version.to_string()).unwrap_or_else(|_| semver::Version::new(0, 0, 0));
            let catalog = catalog::CatalogService::new(catalog_source, store.clone(), monitor.clone(), app_version);
            let scan_monitor = monitor.clone();
            // Downloads: the manager shares the library the scan fills and adds a finished file to it once the scan is
            // done (`library_ready`). A finished model tells the page (`library-changed`); when the library had no
            // runnable model, it is started as the startup default is, so the stranger's first chat is quick.
            let mut dl_cfg = downloads::DownloadConfig::new(
                paths::downloads_dir(),
                catalog_insecure,
                library::LibraryConfig::from_env(paths::downloads_dir()).default_prefix,
            );
            dl_cfg.library_ready = Some(library_ready.clone());
            let (dl_emit, dl_model, dl_chat) = (app.handle().clone(), app.handle().clone(), chat.clone());
            let download_manager = downloads::DownloadManager::new(
                dl_cfg,
                store.clone(),
                library.clone(),
                Arc::new(move |v| {
                    let _ = dl_emit.emit("download-state", v.clone());
                }),
                Arc::new(move |m, none_before| {
                    let _ = dl_model.emit("library-changed", ());
                    if none_before && m.runnable {
                        let chat = dl_chat.clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = chat.start_model(&m).await;
                        });
                    }
                }),
            );
            // Nothing runs at start: running rows become paused, a `.part` no row owns goes (before any window event).
            download_manager.reconcile_at_start();
            app.manage(AppState { store, library: library.clone(), library_ready, monitor, supervisor: supervisor.clone(), chat, updates: updates.clone(), catalog, downloads: download_manager, started_ms: store::now_ms(), launched_ms, shown_ms });

            // Updates (spec §3.7): a check 5 s after start, then every 6 h. A configured endpoint can still answer
            // "not configured" (the plugin's own EmptyEndpoints), and then the loop ends.
            if configured {
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
            }

            let handle = app.handle().clone();
            let mut rx = supervisor.subscribe();
            tauri::async_runtime::spawn(async move {
                while rx.changed().await.is_ok() {
                    let status = rx.borrow().clone();
                    let _ = handle.emit("engine-status", status);
                }
            });
            // The scan runs off the main thread: wry serves the page's requests on it, so a scan here would hold the
            // page back for the scan's whole time (R9). `models` waits for `library_ready`.
            std::thread::spawn(move || {
                let models = library::scan(&library::LibraryConfig::from_env(paths::downloads_dir()));
                scan_monitor.log(
                    "app",
                    "info",
                    format!("library: {} models, {} runnable", models.len(), models.iter().filter(|m| m.runnable).count()),
                );
                let default_model = models.iter().find(|m| m.is_default).cloned();
                *library.write().unwrap() = models;
                library_tx.send_replace(true);
                // Start the engine on the default model so the first reply is quick. Through `start_model`, the app's
                // only way to start or switch the engine: it marks a model the engine refuses as CAN'T RUN. The task
                // wrapper stays, because `start_model` uses `tokio::spawn` and this thread runs outside a runtime.
                if let Some(m) = default_model {
                    tauri::async_runtime::spawn(async move {
                        let _ = startup_chat.start_model(&m).await;
                    });
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::overview,
            commands::models,
            commands::opening_report,
            commands::chats,
            commands::chat,
            commands::new_chat,
            commands::send,
            commands::regenerate,
            commands::stop,
            commands::set_chat_model,
            commands::prepare_model,
            commands::select_version,
            commands::requests,
            commands::logs,
            commands::update_state,
            commands::check_update,
            commands::install_update,
            commands::catalog,
            commands::refresh_catalog,
            commands::open_licence,
            commands::downloads,
            commands::disk_room,
            commands::start_download,
            commands::pause_download,
            commands::resume_download,
            commands::cancel_download
        ])
        .run(tauri::generate_context!())
        .expect("error while running Lorren");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lorren-startup-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&dir);
        dir
    }

    /// The window never appears when the setup fails, so lorren.log is the only place that can say why: one `error`
    /// line goes there, in the app's own words, before the error is handed back.
    #[test]
    fn a_data_folder_that_cannot_be_made_is_logged_before_the_error_is_returned() {
        let blocker = scratch("folder");
        std::fs::write(&blocker, "a file where the data folder's parent should be").unwrap();
        let mon = monitor::Monitor::new(None);
        let err = open_store(&mon, &blocker.join("data")).err().expect("a folder cannot be made under a file");
        let _ = std::fs::remove_file(&blocker);
        let logged = mon.logs_after(0);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert_eq!((logged[0].source.as_str(), logged[0].level.as_str()), ("app", "error"));
        assert_eq!(logged[0].text, format!("startup failed: {err}"));
    }

    #[test]
    fn a_database_that_cannot_be_opened_is_logged_before_the_error_is_returned() {
        let dir = scratch("db");
        std::fs::create_dir_all(dir.join("lorren.db")).unwrap(); // a folder where the database file should be
        let mon = monitor::Monitor::new(None);
        let err = open_store(&mon, &dir).err().expect("a folder is not a database");
        let _ = std::fs::remove_dir_all(&dir);
        let logged = mon.logs_after(0);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert_eq!((logged[0].source.as_str(), logged[0].level.as_str()), ("app", "error"));
        assert_eq!(logged[0].text, format!("startup failed: {err}"));
    }

    #[test]
    fn a_data_folder_that_works_is_opened_and_says_nothing() {
        let dir = scratch("ok");
        let mon = monitor::Monitor::new(None);
        open_store(&mon, &dir.join("nested")).unwrap();
        assert!(dir.join("nested").join("lorren.db").is_file());
        let _ = std::fs::remove_dir_all(&dir);
        assert!(mon.logs_after(0).is_empty(), "{:?}", mon.logs_after(0));
    }
}
