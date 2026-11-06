//! The update setting on a mock Tauri app (see `updater_plugin.rs` for why these are integration tests and not unit tests:
//! a mock app needs the comctl32 version 6 manifest, which `build.rs` gives the integration tests only). The updater plugin
//! is registered with an endpoint nobody listens on (a refused connection, no network), so a check that asks the plugin
//! ends `Failed`; one that returns before asking ends as the mode says.

use lorren_lib::monitor::Monitor;
use lorren_lib::updater::{UpdateMode, UpdateService, UpdateState};
use std::time::Duration;
use tauri::test::{mock_builder, mock_context, noop_assets, MockRuntime};
use tauri::App;

fn app_with_updater() -> App<MockRuntime> {
    let mut context = mock_context::<MockRuntime, _>(noop_assets());
    context.config_mut().plugins.0.insert(
        "updater".into(),
        serde_json::json!({"pubkey": "dW50cnVzdGVk", "endpoints": ["http://127.0.0.1:1/latest.json"]}),
    );
    mock_builder().plugin(tauri_plugin_updater::Builder::new().build()).build(context).expect("a mock app")
}

/// Asking the plugin would end `Failed` (the control below shows it): `Off` proves no request was made.
#[tokio::test]
async fn off_makes_no_request() {
    let app = app_with_updater();
    let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Off);
    service.check(app.handle()).await;
    assert_eq!(service.state(), UpdateState::Off);
    // Turned Off after a start in Auto: the same.
    let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Auto);
    service.set_mode(app.handle(), UpdateMode::Off);
    service.check(app.handle()).await;
    assert_eq!(service.state(), UpdateState::Off);
}

/// The control for the test above: the same app in Auto does reach for the plugin, and fails on the refused connection.
#[tokio::test]
async fn the_control_for_off_a_check_in_auto_does_reach_for_the_plugin() {
    let app = app_with_updater();
    let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Auto);
    service.check(app.handle()).await;
    assert!(matches!(service.state(), UpdateState::Failed { .. }), "{:?}", service.state());
}

async fn woke(service: &UpdateService) -> bool {
    tokio::time::timeout(Duration::from_millis(100), service.woken()).await.is_ok()
}

#[tokio::test]
async fn turning_off_back_on_wakes_the_loop() {
    let app = tauri::test::mock_app();
    let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Auto);
    service.set_mode(app.handle(), UpdateMode::Notify);
    assert!(!woke(&service).await, "Auto to Notify does not wake the loop");
    service.set_mode(app.handle(), UpdateMode::Off);
    assert_eq!(service.state(), UpdateState::Off);
    assert!(!woke(&service).await, "turning off does not wake it");
    service.set_mode(app.handle(), UpdateMode::Auto);
    assert_eq!(service.state(), UpdateState::Idle);
    assert!(woke(&service).await, "Off to Auto wakes it");
    assert_eq!(service.mode(), UpdateMode::Auto);
}

#[tokio::test]
async fn a_service_that_is_not_configured_keeps_the_mode_and_the_state() {
    let app = tauri::test::mock_app();
    let service = UpdateService::new(Monitor::new(None), false, UpdateMode::Auto);
    service.set_mode(app.handle(), UpdateMode::Off);
    assert_eq!((service.state(), service.mode()), (UpdateState::NotConfigured, UpdateMode::Off));
}

#[tokio::test]
async fn download_offered_with_nothing_kept_says_so() {
    let app = tauri::test::mock_app();
    let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Notify);
    assert_eq!(service.download_offered(app.handle()).await, Err("no update is waiting to be downloaded".to_string()));
    assert_eq!(service.state(), UpdateState::Idle);
}
