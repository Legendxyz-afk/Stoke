//! App updates (spec §3.7) through tauri-plugin-updater: check, then download and verify in the
//! background; install only when the user restarts. The plugin checks the minisign signature
//! against the public key in tauri.conf.json; this module decides what a failure means and logs it.

use crate::monitor::Monitor;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};
use tauri_plugin_updater::{Update, UpdaterExt};
use ts_rs::TS;

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum UpdateState {
    /// No update endpoint is configured (every milestone-1 build except the test-feed ones).
    NotConfigured,
    Idle,
    Checking,
    UpToDate { version: String },
    Downloading {
        version: String,
        #[ts(type = "number")]
        done: u64,
        #[ts(type = "number | null")]
        total: Option<u64>,
    },
    /// Downloaded and its signature verified: installs when the user restarts.
    Ready { version: String },
    /// Downloaded, but its signature did not verify: thrown away, nothing installed.
    Refused { version: String, reason: String },
    Failed { message: String },
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    NotConfigured,
    Refused(String),
    Failed(String),
}

pub fn classify(e: &tauri_plugin_updater::Error) -> Verdict {
    use tauri_plugin_updater::Error as E;
    match e {
        E::EmptyEndpoints => Verdict::NotConfigured,
        // The last two are tauri-plugin-updater 2.14's version-binding checks, made by the same `Update::verify`.
        E::Minisign(_) | E::Base64(_) | E::SignatureUtf8(_) | E::SignedVersionMismatch { .. } | E::MissingSignedVersion => {
            Verdict::Refused(format!("the signature did not verify ({e})"))
        }
        _ => Verdict::Failed(e.to_string()),
    }
}

pub struct UpdateService {
    state: Mutex<UpdateState>,
    pending: Mutex<Option<(Update, Vec<u8>)>>,
    monitor: Arc<Monitor>,
}

impl UpdateService {
    pub fn new(monitor: Arc<Monitor>) -> Arc<UpdateService> {
        Arc::new(UpdateService { state: Mutex::new(UpdateState::Idle), pending: Mutex::new(None), monitor })
    }

    pub fn state(&self) -> UpdateState {
        self.state.lock().unwrap().clone()
    }

    fn set<R: Runtime>(&self, app: &AppHandle<R>, s: UpdateState) {
        *self.state.lock().unwrap() = s.clone();
        let _ = app.emit("update-state", s);
    }

    fn fail<R: Runtime>(&self, app: &AppHandle<R>, version: Option<String>, e: &tauri_plugin_updater::Error) {
        match classify(e) {
            Verdict::NotConfigured => self.set(app, UpdateState::NotConfigured),
            Verdict::Refused(reason) => {
                let v = version.unwrap_or_default();
                self.monitor.log("updater", "error", format!("update {v} REFUSED: {reason}; nothing was installed"));
                self.set(app, UpdateState::Refused { version: v, reason });
            }
            Verdict::Failed(message) => {
                self.monitor.log("updater", "warn", format!("update check failed: {message}"));
                self.set(app, UpdateState::Failed { message });
            }
        }
    }

    /// Checks the feed and, when it offers a newer version, downloads and verifies it. Never installs.
    pub async fn check<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) {
        if matches!(self.state(), UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }) {
            return;
        }
        self.set(app, UpdateState::Checking);
        let updater = match app.updater() {
            Ok(u) => u,
            Err(e) => return self.fail(app, None, &e),
        };
        let update = match updater.check().await {
            Ok(Some(u)) => u,
            Ok(None) => {
                let v = app.package_info().version.to_string();
                self.monitor.log("updater", "info", format!("up to date ({v})"));
                return self.set(app, UpdateState::UpToDate { version: v });
            }
            Err(e) => return self.fail(app, None, &e),
        };
        let version = update.version.clone();
        self.monitor.log("updater", "info", format!("update {version} offered at {}", update.download_url));
        self.set(app, UpdateState::Downloading { version: version.clone(), done: 0, total: None });
        let (me, progress_app, v) = (self.clone(), app.clone(), version.clone());
        let mut done = 0u64;
        let mut last = Instant::now();
        let result = update
            .download(
                move |chunk, total| {
                    done += chunk as u64;
                    if last.elapsed().as_millis() >= 250 {
                        last = Instant::now();
                        me.set(&progress_app, UpdateState::Downloading { version: v.clone(), done, total });
                    }
                },
                || {},
            )
            .await;
        match result {
            Ok(bytes) => {
                self.monitor.log(
                    "updater",
                    "info",
                    format!("update {version}: {} bytes downloaded and the signature verified; installs on restart", bytes.len()),
                );
                *self.pending.lock().unwrap() = Some((update, bytes));
                self.set(app, UpdateState::Ready { version });
            }
            Err(e) => self.fail(app, Some(version), &e),
        }
    }

    /// Installs the verified update. On Windows the plugin starts the NSIS installer and exits the
    /// app; the engine dies with it (job object, Task 8).
    pub fn install(&self) -> Result<(), String> {
        let (update, bytes) = self.pending.lock().unwrap().take().ok_or("no verified update is waiting")?;
        self.monitor.log("updater", "info", format!("installing {}", update.version));
        update.install(bytes).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri_plugin_updater::Error as E;

    #[test]
    fn a_signature_that_does_not_verify_is_a_refusal() {
        assert!(matches!(classify(&E::Minisign(minisign_verify::Error::InvalidSignature)), Verdict::Refused(_)));
        assert!(matches!(classify(&E::SignatureUtf8("not base64".into())), Verdict::Refused(_)));
    }

    // tauri-plugin-updater 2.14 added two more ways for `Update::verify` to reject a download: the signature is
    // valid but was made for another version than the feed announced, or carries no version where one is required.
    // Both are "the package is not what was announced": thrown away, not a failed check to retry.
    #[test]
    fn a_package_signed_for_another_version_is_a_refusal() {
        let mismatch = E::SignedVersionMismatch { signed: "0.0.1".into(), announced: "0.0.9".into() };
        assert!(matches!(classify(&mismatch), Verdict::Refused(_)));
        assert!(matches!(classify(&E::MissingSignedVersion), Verdict::Refused(_)));
    }

    #[test]
    fn a_network_failure_is_not_a_refusal_and_no_endpoint_is_not_configured() {
        assert!(matches!(classify(&E::Network("connection refused".into())), Verdict::Failed(_)));
        assert_eq!(classify(&E::EmptyEndpoints), Verdict::NotConfigured);
    }

    #[test]
    fn the_ui_reads_the_state_by_its_tag() {
        let j = serde_json::to_value(UpdateState::Ready { version: "0.0.2".into() }).unwrap();
        assert_eq!(j, serde_json::json!({"state": "ready", "version": "0.0.2"}));
        assert_eq!(serde_json::to_value(UpdateState::NotConfigured).unwrap(), serde_json::json!({"state": "not_configured"}));
    }
}
