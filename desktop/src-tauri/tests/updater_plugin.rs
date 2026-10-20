//! What the updater PLUGIN says when a download cannot be trusted (demo D7). The errors are not written by hand: each one is
//! what `Update::verify`, the check `download()` and `install()` both end with, returns for the case. An `Update` cannot be
//! built outside the plugin by hand, but `Updater::restore_update` builds one from a feed's JSON without a request, and the
//! updater is the real plugin running on `tauri::test`'s mock app. `lorren_lib::updater::classify` then says what each means.
//!
//! The vectors are the plugin's own (tauri-plugin-updater 2.14.0, src/updater.rs:2117-2123, `mod offline_tests`): a throw-away
//! key pair and the signature of the payload `test` made with `--app-version 2.0.0`, and a second key's unversioned
//! signature of the same payload.
//!
//! This is an integration test and not a unit test of the library on purpose: a program that builds a mock Tauri app links the
//! window code, which imports `TaskDialogIndirect`, which only comctl32 version 6 has, and only a program whose manifest asks
//! for version 6 gets that. `build.rs` puts the manifest into the integration tests (`rustc-link-arg-tests`, which does not reach
//! the library's own unit tests); without it the program does not start (STATUS_ENTRYPOINT_NOT_FOUND).

use base64::Engine as _;
use lorren_lib::updater::{classify, is_unsigned, unsigned, Refusal, Verdict};
use tauri::test::{mock_builder, mock_context, noop_assets, MockRuntime};
use tauri_plugin_updater::{Error as E, Update, UpdaterExt as _};

const PUBLIC_KEY: &str = "untrusted comment: minisign public key: A49EF4746FA6E6D7\nRWTX5qZvdPSepDGWg2bz8n6ArQnBMwgQK1geVosorcbLzcunD+KJ+cYe";
const SIGNATURE: &str = "untrusted comment: signature from tauri secret key\nRUTX5qZvdPSepPF2zsjvP/L/TQtBLRnWRvaq+4omK1UEugtQp/USrGiI1v762sDy3id3aor7bRC9ETyeot6lMHHrWoZ5hjUM1Qc=\ntrusted comment: timestamp:1791649051\tfile:payload\tversion:2.0.0\nDDq2CLOEl5cQmC2eJRHlnZPVViOZsu1CjpPKQoei4A/XBfkDqGO0B+QMrcrHgEQ5qV3mne75tdFWhJpqUXh9AA==";
const OTHER_PUBLIC_KEY: &str = "untrusted comment: minisign public key E7620F1842B4E81F\nRWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3";
const OTHER_SIGNATURE: &str = "untrusted comment: signature from minisign secret key\nRUQf6LRCGA9i559r3g7V1qNyJDApGip8MfqcadIgT9CuhV3EMhHoN1mGTkUidF/z7SrlQgXdy8ofjb7bNJJylDOocrCo8KLzZwo=\ntrusted comment: timestamp:1633700835\tfile:test\tprehashed\nwLMDjy9FLAuxZ3q4NlEvkgtyhrr0gtTu6KC4KBJdITbbOeAi1zBIYo0v4iTgt8jJpIidRJnp94ABQkJAgAooBQ==";
const PAYLOAD: &[u8] = b"test";

fn encode(text: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(text)
}

/// The update the plugin makes of a feed's JSON, checked against `pubkey` (the key's text, as the plugin's config holds it
/// after base64). Every release counts as an update, and `target` names the entry of `platforms` that is used.
fn update_from(feed: serde_json::Value, pubkey: &str, target: &str) -> Update {
    let mut context = mock_context::<MockRuntime, _>(noop_assets());
    context
        .config_mut()
        .plugins
        .0
        .insert("updater".into(), serde_json::json!({"pubkey": encode(pubkey), "endpoints": ["https://example.invalid/update"]}));
    let app = mock_builder().plugin(tauri_plugin_updater::Builder::new().build()).build(context).expect("a mock app");
    let updater = app.handle().updater_builder().target(target).version_comparator(|_, _| true).build().expect("an updater");
    updater.restore_update(feed).expect("a feed the plugin reads").expect("an update")
}

/// The update of a feed that announces `version` with `signature` (as the feed carries it: base64 of the signature text).
fn update(pubkey: &str, version: &str, signature: &str) -> Update {
    let feed = serde_json::json!({
        "version": version,
        "platforms": {"test-target": {"url": "https://example.invalid/package", "signature": signature}},
    });
    update_from(feed, pubkey, "test-target")
}

fn refusal(e: &E) -> Refusal {
    match classify(e) {
        Verdict::Refused(r) => r,
        other => panic!("{e:?} should be a refusal, got {other:?}"),
    }
}

/// The control: the vectors verify as they stand, so every case below fails for the one thing it changes.
#[test]
fn the_vectors_verify_as_they_stand() {
    update(PUBLIC_KEY, "2.0.0", &encode(SIGNATURE)).verify(PAYLOAD).expect("the plugin's own vector verifies");
}

/// The `tampered` feed: the installer's bytes changed after they were signed.
#[test]
fn changed_bytes_are_an_altered_installer() {
    let e = update(PUBLIC_KEY, "2.0.0", &encode(SIGNATURE)).verify(b"tampered").unwrap_err();
    assert!(matches!(e, E::Minisign(minisign_verify::Error::InvalidSignature)), "{e:?}");
    let r = refusal(&e);
    assert_eq!(r.label, "installer altered or damaged");
    assert_eq!(r.code, "Minisign(InvalidSignature)");
}

/// The `wrongkey` feed: signed, but by a key the app does not trust.
#[test]
fn a_signature_by_another_key_is_an_untrusted_signing_key() {
    let e = update(PUBLIC_KEY, "2.0.0", &encode(OTHER_SIGNATURE)).verify(PAYLOAD).unwrap_err();
    assert!(matches!(e, E::Minisign(minisign_verify::Error::UnexpectedKeyId)), "{e:?}");
    let r = refusal(&e);
    assert_eq!(r.label, "untrusted signing key");
    assert_eq!(r.code, "Minisign(UnexpectedKeyId)");
}

/// The `nosig` feed has `"signature": ""`. Had it reached the plugin it would read as an unreadable signature, which is why
/// `check()` asks `is_unsigned` of the entry first and says "unsigned".
#[test]
fn a_blank_signature_that_reached_the_plugin_would_be_unreadable() {
    let blank = update(PUBLIC_KEY, "2.0.0", "");
    assert!(is_unsigned(&blank.signature));
    let e = blank.verify(PAYLOAD).unwrap_err();
    assert!(matches!(e, E::Minisign(minisign_verify::Error::InvalidEncoding)), "{e:?}");
    assert_eq!(refusal(&e).label, "unreadable signature");
}

#[test]
fn a_signature_that_is_not_base64_is_unreadable() {
    let e = update(PUBLIC_KEY, "2.0.0", "invalid signature").verify(PAYLOAD).unwrap_err();
    assert!(matches!(e, E::Base64(_)), "{e:?}");
    assert_eq!(refusal(&e).label, "unreadable signature");
}

/// A valid signature of a package made for 2.0.0, offered as 3.0.0 (the feed is not signed, so it could say anything).
#[test]
fn a_signature_made_for_another_version_is_named_so() {
    let e = update(PUBLIC_KEY, "3.0.0", &encode(SIGNATURE)).verify(PAYLOAD).unwrap_err();
    assert!(matches!(e, E::SignedVersionMismatch { .. }), "{e:?}");
    assert_eq!(refusal(&e).label, "signature for another version");
}

/// A signature that names no version where the plugin requires one (`restore_update` always does).
#[test]
fn a_signature_that_names_no_version_is_named_so() {
    let e = update(OTHER_PUBLIC_KEY, "2.0.0", &encode(OTHER_SIGNATURE)).verify(PAYLOAD).unwrap_err();
    assert!(matches!(e, E::MissingSignedVersion), "{e:?}");
    assert_eq!(refusal(&e).label, "signature names no version");
}

/// Not run by `cargo test`: the four test feeds are built by `scripts/update-feed.py feeds` into `e2e/.work/feed`
/// (git-ignored). `cargo test --features fake-engine --test updater_plugin -- --ignored the_four_test_feeds --nocapture` feeds
/// each feed's signature and installer through the plugin's `Update::verify`, with the public key of `tauri.conf.json`, and
/// prints the error and the words it gets. The e2e then shows the same words in the badge of the real app.
#[test]
#[ignore = "needs desktop/e2e/.work/feed, built by scripts/update-feed.py feeds"]
fn the_four_test_feeds_get_the_words_the_e2e_expects() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../e2e/.work/feed");
    let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
    let pubkey = base64::engine::general_purpose::STANDARD
        .decode(conf["plugins"]["updater"]["pubkey"].as_str().expect("the updater's public key"))
        .unwrap();
    let pubkey = String::from_utf8(pubkey).unwrap();
    let want = [("good", None), ("tampered", Some("installer altered or damaged")), ("wrongkey", Some("untrusted signing key")), ("nosig", Some("unsigned"))];
    for (name, label) in want {
        let feed: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(name).join("latest.json")).unwrap()).unwrap();
        let installer = std::fs::read(dir.join(name).join("Lorren_0.0.2_x64-setup.exe")).unwrap();
        let update = update_from(feed.clone(), &pubkey, "windows-x86_64");
        let got = if is_unsigned(&update.signature) {
            println!("{name:9} signature {:?} -> {:?}: {}", update.signature, unsigned().label, unsigned().sentence);
            Some(unsigned().label)
        } else {
            match update.verify(&installer) {
                Ok(()) => {
                    println!("{name:9} verifies");
                    None
                }
                Err(e) => {
                    let r = refusal(&e);
                    println!("{name:9} {e:?} -> {:?}: {}", r.label, r.sentence);
                    Some(r.label)
                }
            }
        };
        assert_eq!(got.as_deref(), label, "{name}");
    }
}
