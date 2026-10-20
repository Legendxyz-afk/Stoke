fn main() {
    tauri_build::build();
    // Windows only. A test program that builds a mock Tauri app (tests/updater_plugin.rs) links Tauri's window code, which
    // imports TaskDialogIndirect, and that function exists only in comctl32 version 6, which a program gets only if its
    // manifest asks for it. tauri-build puts the manifest into the app (a bin); the integration tests get it here, or they
    // do not start (STATUS_ENTRYPOINT_NOT_FOUND). `-tests` does not reach the library's own unit tests, which do not need it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-arg-tests=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-tests=/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
        );
    }
}
