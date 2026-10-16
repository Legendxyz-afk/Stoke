//! Where things live (spec §3.5), with environment overrides for tests and the e2e run.

use std::path::PathBuf;

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var(k).ok().filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn known(k: &str) -> PathBuf {
    env_path(k).unwrap_or_else(std::env::temp_dir)
}

/// %APPDATA%\Lorren, unless `over` (LORREN_DATA_DIR) names another folder.
fn data_dir_from(over: Option<PathBuf>, appdata: PathBuf) -> PathBuf {
    over.unwrap_or_else(|| appdata.join("Lorren"))
}

/// %APPDATA%\Lorren: the database.
pub fn data_dir() -> PathBuf {
    data_dir_from(env_path("LORREN_DATA_DIR"), known("APPDATA"))
}

/// %LOCALAPPDATA%\Lorren\logs, or <LORREN_DATA_DIR>\logs.
pub fn log_dir() -> PathBuf {
    match env_path("LORREN_DATA_DIR") {
        Some(d) => d.join("logs"),
        None => known("LOCALAPPDATA").join("Lorren").join("logs"),
    }
}

/// %USERPROFILE%\Lorren\models: downloaded models (milestone 4 writes here).
pub fn downloads_dir() -> PathBuf {
    known("USERPROFILE").join("Lorren").join("models")
}

/// The engine sidecar: next to lorren.exe, where Tauri copies `externalBin`.
pub fn engine_exe() -> PathBuf {
    env_path("LORREN_ENGINE_EXE").unwrap_or_else(|| {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("lorren-engine.exe")))
            .unwrap_or_else(|| PathBuf::from("lorren-engine.exe"))
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn overrides_win_and_defaults_are_under_appdata() {
        std::env::set_var("LORREN_ENGINE_EXE", r"C:\x\fake-engine.exe");
        assert_eq!(super::engine_exe(), std::path::PathBuf::from(r"C:\x\fake-engine.exe"));
        std::env::remove_var("LORREN_ENGINE_EXE");
        assert!(super::engine_exe().ends_with("lorren-engine.exe"));
        // The default is a pure function of its arguments, so no environment variable can change this answer.
        let appdata = std::path::PathBuf::from("appdata");
        assert_eq!(super::data_dir_from(None, appdata.clone()), appdata.join("Lorren"));
        assert_eq!(super::data_dir_from(Some(std::path::PathBuf::from("over")), appdata), std::path::PathBuf::from("over"));
    }
}
