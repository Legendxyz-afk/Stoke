//! App updates (spec §3.7) through tauri-plugin-updater: check, then download and verify in the
//! background; install only when the user restarts. The plugin checks the minisign signature
//! against the public key in tauri.conf.json; this module decides what a failure means and logs it.

use crate::monitor::Monitor;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};
use tauri_plugin_updater::{Update, UpdaterExt};
use ts_rs::TS;

/// The update setting (distribution design): auto checks and downloads, notify checks and downloads nothing, off makes
/// no request at all. Kept in the store under `MODE_KEY`; the default is auto (§3.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(rename_all = "lowercase")]
pub enum UpdateMode {
    Auto,
    Notify,
    Off,
}

/// The settings key the mode is kept under.
pub const MODE_KEY: &str = "updates.mode";

/// The state a mode change leaves, or None to leave it as it is. Work already running wins (a check, a download, a
/// ready update); NotConfigured outranks the mode; Off shows `Off`; leaving Off shows `Idle`.
pub fn state_after_mode_change(state: &UpdateState, mode: UpdateMode) -> Option<UpdateState> {
    match (state, mode) {
        (UpdateState::NotConfigured, _) => None,
        (UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }, _) => None,
        (UpdateState::Off, UpdateMode::Off) => None,
        (_, UpdateMode::Off) => Some(UpdateState::Off),
        (UpdateState::Off, _) => Some(UpdateState::Idle),
        _ => None,
    }
}

impl UpdateMode {
    /// Anything unknown (a missing, empty or damaged value) is the default, Auto.
    pub fn parse(s: &str) -> UpdateMode {
        match s {
            "notify" => UpdateMode::Notify,
            "off" => UpdateMode::Off,
            _ => UpdateMode::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            UpdateMode::Auto => "auto",
            UpdateMode::Notify => "notify",
            UpdateMode::Off => "off",
        }
    }
}

/// Whether the loop checks the feed at all in this mode: only Off does not.
pub fn wants_check(mode: UpdateMode) -> bool {
    mode != UpdateMode::Off
}

#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[ts(export)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum UpdateState {
    /// The user turned update checks off: no request is made.
    Off,
    /// Notify mode: a newer version is offered and nothing is downloaded until the user asks.
    Available { version: String },
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
    /// Downloaded (or announced), but not trustworthy: thrown away, nothing installed. `label` is a few words for the
    /// badge and `reason` one sentence for its tooltip: the same cause, said once each (demo D7).
    Refused { version: String, label: String, reason: String },
    Failed { message: String },
}

/// Why an update was refused, said once in plain words (demo D7): a few words for the badge, one sentence for its
/// tooltip and the log, and the error's Debug form as a code for support. The plugin's own Display text is never
/// repeated next to the sentence: it says the same thing in other words.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub label: String,
    pub sentence: String,
    pub code: String,
}

#[derive(Debug, PartialEq)]
pub enum Verdict {
    NotConfigured,
    Refused(Refusal),
    Failed(String),
}

/// What a failed download or check means. The words of a refusal are the brief's table (demo D7); the error itself goes
/// into `Refusal::code`. `Update::verify` (tauri-plugin-updater 2.14, src/updater.rs:901) makes every refusal here: the
/// download ends with it (:895), and it reads the key and the signature (`verify_signature`, :1893) before it binds the
/// signed version (`verify_signed_version`, :1925).
pub fn classify(e: &tauri_plugin_updater::Error) -> Verdict {
    use minisign_verify::Error as M;
    use tauri_plugin_updater::Error as E;
    let refused = |label: &str, sentence: &str| {
        Verdict::Refused(Refusal { label: label.into(), sentence: sentence.into(), code: format!("{e:?}") })
    };
    match e {
        E::EmptyEndpoints => Verdict::NotConfigured,
        // The signature is well formed and the key is ours, and the bytes do not match it.
        E::Minisign(M::InvalidSignature) => refused(
            "installer altered or damaged",
            "the downloaded installer does not match its signature, so it changed after it was signed",
        ),
        // A well formed signature made by another key (minisign compares the key ids before anything else).
        E::Minisign(M::UnexpectedKeyId) => refused("untrusted signing key", "the update is signed with a key this Lorren does not trust"),
        // Not base64, not UTF-8, or not a minisign signature (a wrong algorithm, a cut-short text, an empty one).
        E::Base64(_) | E::SignatureUtf8(_) | E::Minisign(_) => refused("unreadable signature", "the update's signature cannot be read"),
        // The two version-binding checks of the same `Update::verify`.
        E::SignedVersionMismatch { .. } => {
            refused("signature for another version", "the signature is for a different version than the one offered")
        }
        E::MissingSignedVersion => refused("signature names no version", "the signature does not say which version it is for"),
        _ => Verdict::Failed(e.to_string()),
    }
}

/// A feed entry with no signature at all (an empty or blank string): refused before anything is downloaded.
pub fn is_unsigned(signature: &str) -> bool {
    signature.trim().is_empty()
}

pub fn unsigned() -> Refusal {
    Refusal { label: "unsigned".into(), sentence: "the update has no signature".into(), code: "EmptySignature".into() }
}

/// The log line for a refusal: the sentence, then the error's Debug form in brackets. That is a code for whoever supports
/// the user, never prose, and it is the only place the plugin's error shows.
pub fn refusal_log_line(version: &str, r: &Refusal) -> String {
    format!("update {version} REFUSED: {}; nothing was installed [{}]", r.sentence, r.code)
}

/// What an install press is told when no downloaded, verified update is waiting. `install_update` checks for that
/// before it stops the engine and `UpdateService::install` checks again when it takes the update, so both use these
/// words and cannot drift apart (R64).
pub const NO_VERIFIED_UPDATE: &str = "no verified update is waiting";

pub struct UpdateService {
    state: Mutex<UpdateState>,
    pending: Mutex<Option<(Update, Vec<u8>)>>,
    /// Notify mode: the update the feed offered, kept for `download_offered`.
    offered: Mutex<Option<Update>>,
    mode: Mutex<UpdateMode>,
    /// Woken when the mode leaves Off, so the loop checks now instead of at its next turn.
    wake: tokio::sync::Notify,
    monitor: Arc<Monitor>,
}

/// What `download_offered` says when nothing was offered.
pub const NO_OFFERED_UPDATE: &str = "no update is waiting to be downloaded";

/// Whether the build's `plugins.updater` config names a feed to check: a non-empty `endpoints` array. Without one the
/// service starts NotConfigured and no check is scheduled.
pub fn endpoints_configured(updater_config: Option<&serde_json::Value>) -> bool {
    updater_config.and_then(|c| c.get("endpoints")).and_then(|e| e.as_array()).is_some_and(|e| !e.is_empty())
}

impl UpdateService {
    /// `configured` is `endpoints_configured` of the build's config: with no endpoint the badge starts NotConfigured
    /// (and shows nothing) instead of Idle ("check for updates") until a first check, 5 s later, says so.
    /// With no endpoint the state is NotConfigured whatever the mode; otherwise Off for mode Off, else Idle.
    pub fn new(monitor: Arc<Monitor>, configured: bool, mode: UpdateMode) -> Arc<UpdateService> {
        let state = if !configured {
            UpdateState::NotConfigured
        } else if mode == UpdateMode::Off {
            UpdateState::Off
        } else {
            UpdateState::Idle
        };
        Arc::new(UpdateService {
            state: Mutex::new(state),
            pending: Mutex::new(None),
            offered: Mutex::new(None),
            mode: Mutex::new(mode),
            wake: tokio::sync::Notify::new(),
            monitor,
        })
    }

    pub fn mode(&self) -> UpdateMode {
        *self.mode.lock().unwrap()
    }

    /// Resolves when `set_mode` turned checks back on (a permit is kept if nobody was waiting).
    pub async fn woken(&self) {
        self.wake.notified().await
    }

    /// Takes a new mode (the caller has stored it). Work already running wins: a check, a download or a ready update
    /// keeps the state. Off shows `Off`; leaving Off shows `Idle` and wakes the loop to check now.
    pub fn set_mode<R: Runtime>(&self, app: &AppHandle<R>, mode: UpdateMode) {
        let prev = std::mem::replace(&mut *self.mode.lock().unwrap(), mode);
        let new_state = {
            let mut state = self.state.lock().unwrap();
            let next = state_after_mode_change(&state, mode);
            if let Some(n) = &next {
                *state = n.clone();
            }
            next
        };
        if let Some(n) = new_state {
            let _ = app.emit("update-state", n);
        }
        if prev == UpdateMode::Off && mode != UpdateMode::Off {
            self.wake.notify_one();
        }
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
            Verdict::Refused(r) => self.refuse(app, version.unwrap_or_default(), r),
            Verdict::Failed(message) => {
                self.monitor.log("updater", "warn", format!("update check failed: {message}"));
                self.set(app, UpdateState::Failed { message });
            }
        }
    }

    /// Logs a refusal and shows it. Nothing was installed.
    fn refuse<R: Runtime>(&self, app: &AppHandle<R>, version: String, r: Refusal) {
        self.monitor.log("updater", "error", refusal_log_line(&version, &r));
        self.set(app, UpdateState::Refused { version, label: r.label, reason: r.sentence });
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

    /// Checks the feed. Auto: when it offers a newer version, downloads and verifies it. Notify: keeps the offer
    /// (`Available`) and downloads nothing. Off: returns before any request. Never installs.
    pub async fn check<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) {
        if self.mode() == UpdateMode::Off {
            let busy = matches!(self.state(), UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. });
            if !busy {
                self.set(app, UpdateState::Off);
            }
            return;
        }
        if !self.begin_check() {
            return;
        }
        let _ = app.emit("update-state", UpdateState::Checking);
        self.check_feed(app).await;
        // The mode may have turned Off while this ran: the work ran to its end, and unless a verified update is
        // waiting (or a download is running) the last thing shown is that checks are off.
        if self.mode() == UpdateMode::Off && !matches!(self.state(), UpdateState::Downloading { .. } | UpdateState::Ready { .. }) {
            *self.offered.lock().unwrap() = None;
            self.set(app, UpdateState::Off);
        }
    }

    async fn check_feed<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) {
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
        if is_unsigned(&update.signature) {
            return self.refuse(app, version, unsigned());
        }
        if self.mode() == UpdateMode::Notify {
            self.monitor.log("updater", "info", format!("update {version} offered; waiting for the user (notify)"));
            *self.offered.lock().unwrap() = Some(update);
            return self.set(app, UpdateState::Available { version });
        }
        self.run_download(app, update).await;
    }

    /// The download half, shared by Auto's check and `download_offered` so the two cannot drift: the unsigned refusal,
    /// Downloading with progress every 250 ms, then `pending` and Ready, or the failure.
    async fn run_download<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>, update: Update) {
        let version = update.version.clone();
        if is_unsigned(&update.signature) {
            return self.refuse(app, version, unsigned());
        }
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

    /// Notify mode: the user asked for the offered update. Downloads and verifies it as Auto does, ending `Ready`.
    /// A press while a check or download is running, or an update is ready, starts nothing.
    pub async fn download_offered<R: Runtime>(self: &Arc<Self>, app: &AppHandle<R>) -> Result<(), String> {
        let update = {
            let mut state = self.state.lock().unwrap();
            if matches!(*state, UpdateState::Checking | UpdateState::Downloading { .. } | UpdateState::Ready { .. }) {
                return Ok(());
            }
            let Some(update) = self.offered.lock().unwrap().take() else {
                return Err(NO_OFFERED_UPDATE.to_string());
            };
            *state = UpdateState::Downloading { version: update.version.clone(), done: 0, total: None };
            update
        };
        self.run_download(app, update).await;
        Ok(())
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

    // Demo D7: a refusal is said once, in plain words. The table is the brief's: the error, the label for the badge and
    // the sentence for the tooltip and the log. tauri-plugin-updater 2.14 has two version-binding refusals next to the
    // signature ones (the signature is valid but was made for another version than the feed announced, or names none
    // where one is required): both are "the package is not what was announced", thrown away, not a failed check to retry.
    fn refusal_rows() -> Vec<(E, &'static str, &'static str)> {
        use base64::Engine as _;
        use minisign_verify::Error as M;
        let unreadable = "the update's signature cannot be read";
        let not_base64 = base64::engine::general_purpose::STANDARD.decode("***").unwrap_err();
        vec![
            (
                E::Minisign(M::InvalidSignature),
                "installer altered or damaged",
                "the downloaded installer does not match its signature, so it changed after it was signed",
            ),
            (E::Minisign(M::UnexpectedKeyId), "untrusted signing key", "the update is signed with a key this Lorren does not trust"),
            (E::Base64(not_base64), "unreadable signature", unreadable),
            (E::SignatureUtf8("not base64".into()), "unreadable signature", unreadable),
            (E::Minisign(M::InvalidEncoding), "unreadable signature", unreadable),
            (E::Minisign(M::UnexpectedAlgorithm), "unreadable signature", unreadable),
            (E::Minisign(M::UnsupportedAlgorithm), "unreadable signature", unreadable),
            (E::Minisign(M::UnsupportedLegacyMode), "unreadable signature", unreadable),
            (E::Minisign(M::IoError(std::io::Error::other("disk"))), "unreadable signature", unreadable),
            (
                E::SignedVersionMismatch { signed: "0.0.1".into(), announced: "0.0.9".into() },
                "signature for another version",
                "the signature is for a different version than the one offered",
            ),
            (E::MissingSignedVersion, "signature names no version", "the signature does not say which version it is for"),
        ]
    }

    #[test]
    fn each_refusal_gets_its_label_and_its_sentence_and_the_debug_form_as_a_code() {
        for (e, label, sentence) in refusal_rows() {
            let want = Refusal { label: label.into(), sentence: sentence.into(), code: format!("{e:?}") };
            assert_eq!(classify(&e), Verdict::Refused(want), "{e:?}");
        }
    }

    /// The plugin's Display text says the same thing as the sentence in other words ("The signature verification failed"
    /// beside "did not verify"); neither the label nor the sentence carries it, nor any parenthesised error text.
    #[test]
    fn the_plugins_own_text_is_never_printed_next_to_the_sentence() {
        for (e, ..) in refusal_rows() {
            let Verdict::Refused(r) = classify(&e) else { panic!("{e:?} is a refusal") };
            let own = e.to_string();
            for (what, text) in [("label", &r.label), ("sentence", &r.sentence)] {
                assert!(!text.contains(&own), "{e:?}: the {what} {text:?} repeats {own:?}");
                assert!(!text.contains("verification failed") && !text.contains("did not verify"), "{e:?}: the {what} {text:?}");
                assert!(!text.contains('(') && !text.contains(&r.code), "{e:?}: the {what} {text:?} carries error text");
            }
        }
    }

    /// Anything that is not a refusal keeps its meaning: a network failure is a failed check, no endpoint is not configured.
    #[test]
    fn a_network_failure_is_not_a_refusal_and_no_endpoint_is_not_configured() {
        assert!(matches!(classify(&E::Network("connection refused".into())), Verdict::Failed(_)));
        assert_eq!(classify(&E::EmptyEndpoints), Verdict::NotConfigured);
    }

    /// A feed entry with no signature is caught from the entry itself, before a byte of the installer is downloaded.
    #[test]
    fn an_empty_or_blank_signature_in_the_feed_is_unsigned() {
        for blank in ["", " ", "\n", "  \t\r\n "] {
            assert!(is_unsigned(blank), "{blank:?}");
        }
        for there in ["dW50cnVzdGVkIGNvbW1lbnQ6", "x", " x "] {
            assert!(!is_unsigned(there), "{there:?}");
        }
        let u = unsigned();
        assert_eq!((u.label.as_str(), u.sentence.as_str(), u.code.as_str()), ("unsigned", "the update has no signature", "EmptySignature"));
    }

    /// The Debug form is a code for support: it goes at the end of the log line, in brackets, after the sentence.
    #[test]
    fn the_log_line_is_the_sentence_then_the_code_in_brackets() {
        let Verdict::Refused(r) = classify(&E::Minisign(minisign_verify::Error::InvalidSignature)) else { panic!("a refusal") };
        assert_eq!(
            refusal_log_line("0.0.2", &r),
            "update 0.0.2 REFUSED: the downloaded installer does not match its signature, so it changed after it was signed; nothing was installed [Minisign(InvalidSignature)]"
        );
        assert_eq!(
            refusal_log_line("0.0.2", &unsigned()),
            "update 0.0.2 REFUSED: the update has no signature; nothing was installed [EmptySignature]"
        );
    }

    #[test]
    fn the_refused_state_carries_the_label_and_the_sentence() {
        let refused = UpdateState::Refused { version: "0.0.2".into(), label: "unsigned".into(), reason: "the update has no signature".into() };
        assert_eq!(
            serde_json::to_value(refused).unwrap(),
            serde_json::json!({"state": "refused", "version": "0.0.2", "label": "unsigned", "reason": "the update has no signature"})
        );
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
        let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Auto);
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
        assert_eq!(UpdateService::new(Monitor::new(None), false, UpdateMode::Auto).state(), UpdateState::NotConfigured);
        assert_eq!(UpdateService::new(Monitor::new(None), true, UpdateMode::Auto).state(), UpdateState::Idle);
    }

    fn service_in(state: UpdateState) -> Arc<UpdateService> {
        let service = UpdateService::new(Monitor::new(None), true, UpdateMode::Auto);
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
            UpdateState::Refused { version: "0.0.2".into(), label: "unsigned".into(), reason: "the update has no signature".into() },
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

    // ---- the update setting ----

    #[test]
    fn the_mode_is_auto_unless_the_store_says_otherwise() {
        assert_eq!(UpdateMode::parse("auto"), UpdateMode::Auto);
        assert_eq!(UpdateMode::parse("notify"), UpdateMode::Notify);
        assert_eq!(UpdateMode::parse("off"), UpdateMode::Off);
        assert_eq!(UpdateMode::parse(""), UpdateMode::Auto);
        assert_eq!(UpdateMode::parse("OFF?"), UpdateMode::Auto);
        for m in [UpdateMode::Auto, UpdateMode::Notify, UpdateMode::Off] {
            assert_eq!(UpdateMode::parse(m.as_str()), m);
        }
        assert_eq!(MODE_KEY, "updates.mode");
    }

    #[test]
    fn wants_check_is_false_only_for_off() {
        assert!(wants_check(UpdateMode::Auto));
        assert!(wants_check(UpdateMode::Notify));
        assert!(!wants_check(UpdateMode::Off));
    }

    #[test]
    fn not_configured_outranks_off_and_off_shows_off() {
        for m in [UpdateMode::Auto, UpdateMode::Notify, UpdateMode::Off] {
            assert_eq!(UpdateService::new(Monitor::new(None), false, m).state(), UpdateState::NotConfigured);
        }
        assert_eq!(UpdateService::new(Monitor::new(None), true, UpdateMode::Off).state(), UpdateState::Off);
        assert_eq!(UpdateService::new(Monitor::new(None), true, UpdateMode::Notify).state(), UpdateState::Idle);
    }

    #[test]
    fn work_already_running_keeps_its_state_when_the_mode_changes() {
        for busy in [
            UpdateState::Checking,
            UpdateState::Downloading { version: "0.0.2".into(), done: 1, total: None },
            UpdateState::Ready { version: "0.0.2".into() },
        ] {
            for m in [UpdateMode::Auto, UpdateMode::Notify, UpdateMode::Off] {
                assert_eq!(state_after_mode_change(&busy, m), None, "{busy:?} {m:?}");
            }
        }
        for m in [UpdateMode::Auto, UpdateMode::Notify, UpdateMode::Off] {
            assert_eq!(state_after_mode_change(&UpdateState::NotConfigured, m), None);
        }
        let idle = UpdateState::UpToDate { version: "0.0.1".into() };
        assert_eq!(state_after_mode_change(&idle, UpdateMode::Off), Some(UpdateState::Off));
        assert_eq!(state_after_mode_change(&idle, UpdateMode::Notify), None);
        assert_eq!(state_after_mode_change(&UpdateState::Off, UpdateMode::Auto), Some(UpdateState::Idle));
        assert_eq!(state_after_mode_change(&UpdateState::Off, UpdateMode::Off), None);
    }

    #[test]
    fn the_new_states_read_by_their_tags_and_the_ts_export_carries_them() {
        assert_eq!(serde_json::to_value(UpdateState::Off).unwrap(), serde_json::json!({"state": "off"}));
        assert_eq!(
            serde_json::to_value(UpdateState::Available { version: "0.0.2".into() }).unwrap(),
            serde_json::json!({"state": "available", "version": "0.0.2"})
        );
        assert_eq!(serde_json::to_value(UpdateMode::Notify).unwrap(), serde_json::json!("notify"));
        let ts = UpdateState::export_to_string(&ts_rs::Config::default()).unwrap();
        assert!(ts.contains("\"off\"") && ts.contains("\"available\""), "{ts}");
    }
}
