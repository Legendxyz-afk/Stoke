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

/// What an install press is told when no downloaded, verified update is waiting. `install_update` checks for that
/// before it stops the engine and `UpdateService::install` checks again when it takes the update, so both use these
/// words and cannot drift apart (R64).
pub const NO_VERIFIED_UPDATE: &str = "no verified update is waiting";

pub struct UpdateService {
    state: Mutex<UpdateState>,
    pending: Mutex<Option<(Update, Vec<u8>)>>,
    monitor: Arc<Monitor>,
}

/// Whether the build's `plugins.updater` config names a feed to check: a non-empty `endpoints` array. Without one the
/// service starts NotConfigured and no check is scheduled.
pub fn endpoints_configured(updater_config: Option<&serde_json::Value>) -> bool {
    updater_config.and_then(|c| c.get("endpoints")).and_then(|e| e.as_array()).is_some_and(|e| !e.is_empty())
}

impl UpdateService {
    /// `configured` is `endpoints_configured` of the build's config: with no endpoint the badge starts NotConfigured
    /// (and shows nothing) instead of Idle ("check for updates") until a first check, 5 s later, says so.
    pub fn new(monitor: Arc<Monitor>, configured: bool) -> Arc<UpdateService> {
        let state = if configured { UpdateState::Idle } else { UpdateState::NotConfigured };
        Arc::new(UpdateService { state: Mutex::new(state), pending: Mutex::new(None), monitor })
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

    /// Takes the check, or says another one has it: false while a check or a download is running and while a verified
    /// update is waiting; otherwise the state becomes Checking and this is true. Test and set under ONE lock: with the
    /// read and the write apart, the startup check and a click could both pass the read and both download.
    fn begin_check(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if matches!(*state, UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }) {
            return false;
        }
        *state = UpdateState::Checking;
        true
    }

    /// Checks the feed and, when it offers a newer version, downloads and verifies it. Never installs.
    pub async fn check<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) {
        if !self.begin_check() {
            return;
        }
        let _ = app.emit("update-state", UpdateState::Checking);
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

    /// The version of the downloaded, verified update that is waiting for the user to restart, if there is one.
    /// `install_update` asks this before it stops the engine, so a press with nothing waiting leaves the engine alone.
    pub fn ready_version(&self) -> Option<String> {
        self.pending.lock().unwrap().as_ref().map(|(update, _)| update.version.clone())
    }

    /// Installs the verified update. On Windows the plugin starts the NSIS installer and exits the
    /// app; the engine dies with it (job object, Task 8).
    ///
    /// When the plugin cannot install, the update and its bytes go back into `pending` and the state stays `Ready`, so
    /// the badge still offers the install and the next press tries again; the reason is logged and returned (R64).
    /// `Update::install` borrows the bytes (it verifies them again, then starts the installer), which is what makes
    /// the hand-back possible. It returns only on failure on Windows: on success the app exits inside it.
    pub fn install(&self) -> Result<(), String> {
        let Some((update, bytes)) = self.pending.lock().unwrap().take() else {
            return Err(NO_VERIFIED_UPDATE.to_string());
        };
        self.monitor.log("updater", "info", format!("installing {}", update.version));
        match update.install(&bytes) {
            Ok(()) => Ok(()),
            Err(e) => {
                let message = format!("update {} could not be installed: {e}", update.version);
                self.monitor.log("updater", "error", message.clone());
                *self.pending.lock().unwrap() = Some((update, bytes));
                Err(message)
            }
        }
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

    // R64: the install command asks whether a verified update is waiting before it stops the engine, so that has to be
    // answerable without an `Update` in hand, and the command and `install()` have to say the same words when it is not.
    // Only this branch is tested: a plugin `Update` cannot be built outside the plugin, so a waiting update, and
    // `Update::install` failing and handing its bytes back, can be checked by reading only.
    #[test]
    fn with_nothing_downloaded_no_update_is_ready_and_install_says_so_in_the_shared_words() {
        let service = UpdateService::new(Monitor::new(None), true);
        assert_eq!(service.ready_version(), None);
        assert_eq!(NO_VERIFIED_UPDATE, "no verified update is waiting"); // the words the badge shows after "could not install the update: "
        assert_eq!(service.install(), Err(NO_VERIFIED_UPDATE.to_string()));
        assert_eq!(service.ready_version(), None); // asking, and being refused, used nothing up
    }

    // The badge starts NotConfigured when the build has no endpoint, so a run without one never shows "check for updates"
    // while the first check (5 s after start) is still to come. The decision is a pure function of the plugin's config.
    #[test]
    fn only_a_non_empty_endpoints_array_counts_as_configured() {
        use serde_json::json;
        assert!(!endpoints_configured(Some(&json!({"endpoints": []}))));
        assert!(endpoints_configured(Some(&json!({"endpoints": ["http://127.0.0.1:8765/latest.json"]}))));
        assert!(!endpoints_configured(Some(&json!({"pubkey": "abc", "windows": {"installMode": "passive"}}))), "no endpoints key");
        assert!(!endpoints_configured(Some(&json!({"endpoints": "http://127.0.0.1:8765/latest.json"}))), "not an array");
        assert!(!endpoints_configured(None), "no updater section at all");
    }

    #[test]
    fn a_service_without_an_endpoint_starts_not_configured_and_one_with_an_endpoint_starts_idle() {
        assert_eq!(UpdateService::new(Monitor::new(None), false).state(), UpdateState::NotConfigured);
        assert_eq!(UpdateService::new(Monitor::new(None), true).state(), UpdateState::Idle);
    }

    fn service_in(state: UpdateState) -> Arc<UpdateService> {
        let service = UpdateService::new(Monitor::new(None), true);
        *service.state.lock().unwrap() = state;
        service
    }

    // `check()` used to read the state and set Checking in two lock acquisitions, so the startup check and a click could
    // both pass the read and both download. `begin_check` is the one test-and-set.
    #[test]
    fn begin_check_takes_the_check_once() {
        let service = service_in(UpdateState::Idle);
        assert!(service.begin_check());
        assert_eq!(service.state(), UpdateState::Checking);
        assert!(!service.begin_check());
        assert_eq!(service.state(), UpdateState::Checking);
    }

    #[test]
    fn begin_check_is_refused_while_a_check_a_download_or_a_ready_update_is_there() {
        for busy in [
            UpdateState::Checking,
            UpdateState::Downloading { version: "0.0.2".into(), done: 10, total: Some(100) },
            UpdateState::Ready { version: "0.0.2".into() },
        ] {
            let service = service_in(busy.clone());
            assert!(!service.begin_check(), "{busy:?}");
            assert_eq!(service.state(), busy, "a refused call leaves the state as it was");
        }
    }

    #[test]
    fn begin_check_starts_from_every_other_state() {
        for from in [
            UpdateState::Idle,
            UpdateState::NotConfigured,
            UpdateState::UpToDate { version: "0.0.1".into() },
            UpdateState::Refused { version: "0.0.2".into(), reason: "the signature did not verify".into() },
            UpdateState::Failed { message: "connection refused".into() },
        ] {
            let service = service_in(from.clone());
            assert!(service.begin_check(), "{from:?}");
            assert_eq!(service.state(), UpdateState::Checking, "{from:?}");
        }
    }

    /// Callers that arrive together: in every round exactly one of them begins the check. Sequential calls cannot tell
    /// one lock acquisition from two; eight threads released by a barrier, 2,000 times over, can.
    #[test]
    fn of_callers_arriving_together_exactly_one_begins_the_check() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Barrier;
        const THREADS: usize = 8;
        const ROUNDS: usize = 2000;
        let service = service_in(UpdateState::Idle);
        let (go, finished) = (Arc::new(Barrier::new(THREADS + 1)), Arc::new(Barrier::new(THREADS + 1)));
        let winners = Arc::new(AtomicUsize::new(0));
        let callers: Vec<_> = (0..THREADS)
            .map(|_| {
                let (service, go, finished, winners) = (service.clone(), go.clone(), finished.clone(), winners.clone());
                std::thread::spawn(move || {
                    for _ in 0..ROUNDS {
                        go.wait();
                        if service.begin_check() {
                            winners.fetch_add(1, Ordering::SeqCst);
                        }
                        finished.wait();
                    }
                })
            })
            .collect();
        let mut bad = Vec::new();
        for round in 0..ROUNDS {
            *service.state.lock().unwrap() = UpdateState::Idle;
            winners.store(0, Ordering::SeqCst);
            go.wait();
            finished.wait();
            let n = winners.load(Ordering::SeqCst);
            if n != 1 {
                bad.push((round, n));
            }
        }
        for c in callers {
            c.join().unwrap();
        }
        assert!(
            bad.is_empty(),
            "{} of {ROUNDS} rounds did not have exactly one caller begin the check; the first (round, callers): {:?}",
            bad.len(),
            &bad[..bad.len().min(10)]
        );
    }
}
