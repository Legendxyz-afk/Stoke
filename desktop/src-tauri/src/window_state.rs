//! The main window's last state (demo D8): it opens maximized the first time, and as it was left after that.
//!
//! The file is `window.json` in the data folder (`paths::data_dir()`, so `LORREN_DATA_DIR` keeps tests and the e2e away
//! from the owner's real state). The decision of what to do with it is a pure function (`place`); the window calls that
//! carry it out, and the hooks that save the state, are at the end of the file.

use serde::{Deserialize, Serialize};
use std::path::Path;

pub const FILE: &str = "window.json";

/// What is kept between runs: whether the window was maximized, and, when it was not, where it was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    pub maximized: bool,
    /// The window's outer rectangle in PHYSICAL pixels, with the scale factor it was taken at. Absent for a window that
    /// was maximized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<SavedRect>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SavedRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
}

/// A rectangle in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// A monitor: its area in physical pixels on the virtual desktop, and its scale factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Screen {
    pub area: Rect,
    pub scale: f64,
}

/// The window's configuration (`tauri.conf.json`), in logical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Defaults {
    pub width: f64,
    pub height: f64,
    pub min_width: f64,
    pub min_height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Placement {
    Maximize,
    /// The window's outer rectangle, in physical pixels.
    Window(Rect),
}

/// How much of the window has to be on a monitor, in each direction, for the window to count as being on it.
const MIN_OVERLAP_PX: i64 = 100;

/// Whether `a` and `b` share at least `MIN_OVERLAP_PX` pixels both across and down.
fn overlaps_enough(a: Rect, b: Rect) -> bool {
    let (ax, ay, bx, by) = (a.x as i64, a.y as i64, b.x as i64, b.y as i64);
    let across = (ax + a.width as i64).min(bx + b.width as i64) - ax.max(bx);
    let down = (ay + a.height as i64).min(by + b.height as i64) - ay.max(by);
    across >= MIN_OVERLAP_PX && down >= MIN_OVERLAP_PX
}

/// A scale factor that can scale: anything else (zero, negative, not a number, infinite) counts as 1.
fn sane(scale: f64) -> f64 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

/// `logical` pixels at `scale`, in whole physical pixels.
fn scaled(logical: f64, scale: f64) -> u32 {
    (logical * scale).round().clamp(0.0, u32::MAX as f64) as u32
}

/// The configuration's default size, centred on `on`; never larger than the monitor (a 1440 x 900 window on a 1366 x 768
/// laptop is the monitor's size, not a title bar above the top edge) and never below the configuration's minimum.
fn centred(on: Screen, d: Defaults) -> Rect {
    let scale = sane(on.scale);
    let width = scaled(d.width, scale).min(on.area.width).max(scaled(d.min_width, scale));
    let height = scaled(d.height, scale).min(on.area.height).max(scaled(d.min_height, scale));
    Rect {
        x: on.area.x + ((on.area.width as i64 - width as i64) / 2) as i32,
        y: on.area.y + ((on.area.height as i64 - height as i64) / 2) as i32,
        width,
        height,
    }
}

/// What to do with the window at startup. `screens` lists the monitors, the PRIMARY one first.
///
/// - no state, an unreadable one, or one that says nothing usable: maximize (the default is full screen);
/// - maximized: maximize;
/// - not maximized: the saved outer rectangle, raised to the configuration's minimum size (a logical size: the scale the
///   rectangle was taken at turns it into physical pixels), if at least 100 x 100 pixels of it are on a monitor;
/// - otherwise (the monitor it was on is gone, or the state is nonsense): the default size, centred on the primary monitor.
pub fn place(saved: Option<&Saved>, screens: &[Screen], defaults: Defaults) -> Placement {
    let Some(saved) = saved else { return Placement::Maximize };
    if saved.maximized {
        return Placement::Maximize;
    }
    let Some(r) = saved.rect else { return Placement::Maximize };
    if r.width == 0 || r.height == 0 || screens.is_empty() {
        return Placement::Maximize;
    }
    let scale = sane(r.scale);
    let rect = Rect {
        x: r.x,
        y: r.y,
        width: r.width.max(scaled(defaults.min_width, scale)),
        height: r.height.max(scaled(defaults.min_height, scale)),
    };
    if screens.iter().any(|s| overlaps_enough(rect, s.area)) {
        Placement::Window(rect)
    } else {
        Placement::Window(centred(screens[0], defaults))
    }
}

/// The saved state, or `None` when there is none or it cannot be read (no file, not JSON, the wrong shape).
pub fn read(dir: &Path) -> Option<Saved> {
    let text = std::fs::read_to_string(dir.join(FILE)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Writes the state atomically: a temporary file in the same folder, then a rename over the real one, so a reader (or a
/// crash half-way) never sees half a file. The folder is made if it is not there yet.
pub fn write(dir: &Path, saved: &Saved) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let json = serde_json::to_string_pretty(saved).map_err(std::io::Error::other)?;
    let temp = dir.join(format!("{FILE}.tmp"));
    std::fs::write(&temp, json)?;
    std::fs::rename(&temp, dir.join(FILE)).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

/// What a closing window leaves in the file. `None` leaves the file as it was: a minimized window has no placement of
/// its own, so the state it had before it was minimized stays.
pub fn to_save(maximized: bool, minimized: bool, outer: Option<SavedRect>) -> Option<Saved> {
    if minimized {
        return None;
    }
    if maximized {
        return Some(Saved { maximized: true, rect: None });
    }
    outer.map(|rect| Saved { maximized: false, rect: Some(rect) })
}

// ---------------------------------------------------------------------------------------------------------------------
// The window calls. They need a running app and cannot be tested; the decisions above can.
// ---------------------------------------------------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{Manager, PhysicalPosition, PhysicalSize, Runtime, WebviewWindow, WindowEvent};

/// Set when the main window's `CloseRequested` has been seen and the state saved, so that the `Destroyed` that follows it
/// does not save a second time, from a window that is already going away.
static SAVED_ON_CLOSE: AtomicBool = AtomicBool::new(false);

/// The monitors, the primary one first.
fn screens_of<R: Runtime>(window: &WebviewWindow<R>) -> Vec<Screen> {
    let screen = |m: &tauri::Monitor| Screen {
        area: Rect { x: m.position().x, y: m.position().y, width: m.size().width, height: m.size().height },
        scale: m.scale_factor(),
    };
    let mut screens: Vec<Screen> = window.available_monitors().unwrap_or_default().iter().map(screen).collect();
    if let Ok(Some(primary)) = window.primary_monitor() {
        let primary = screen(&primary);
        if let Some(at) = screens.iter().position(|s| s.area == primary.area) {
            screens.swap(0, at);
        }
    }
    screens
}

/// The configuration's main window, in logical pixels.
fn defaults_of(app: &tauri::App) -> Defaults {
    let main = app.config().app.windows.iter().find(|w| w.label == "main");
    Defaults {
        width: main.map_or(1440.0, |w| w.width),
        height: main.map_or(900.0, |w| w.height),
        min_width: main.and_then(|w| w.min_width).unwrap_or(1100.0),
        min_height: main.and_then(|w| w.min_height).unwrap_or(680.0),
    }
}

/// What the operating system draws around the window's content (title bar and borders): the outer size less the inner one.
/// `set_size` sets the INNER size, and the saved rectangle is the outer one.
fn frame<R: Runtime>(window: &WebviewWindow<R>) -> (u32, u32) {
    match (window.outer_size(), window.inner_size()) {
        (Ok(outer), Ok(inner)) => (outer.width.saturating_sub(inner.width), outer.height.saturating_sub(inner.height)),
        _ => (0, 0),
    }
}

/// First thing in `setup`: puts the main window where the last run left it, or maximizes it, and then shows it. The window
/// is created hidden (`"visible": false` in tauri.conf.json), so it is never seen at the wrong size first. A window that is
/// hidden is maximized without being shown (tao keeps the flag and does the `ShowWindow` when the window is shown). Nothing
/// here can leave the window hidden: every window call may fail, and the window is shown all the same. Returns a line for
/// the log.
pub fn restore(app: &tauri::App) -> String {
    let Some(window) = app.get_webview_window("main") else { return "window: there is no main window to place".into() };
    let saved = read(&crate::paths::data_dir());
    let placement = place(saved.as_ref(), &screens_of(&window), defaults_of(app));
    let note = match placement {
        Placement::Maximize => {
            let _ = window.maximize();
            format!("window: opened maximized ({})", if saved.is_some() { "as it was left" } else { "no saved state" })
        }
        Placement::Window(r) => {
            let (frame_width, frame_height) = frame(&window);
            let _ = window.set_position(PhysicalPosition::new(r.x, r.y));
            let _ = window.set_size(PhysicalSize::new(r.width.saturating_sub(frame_width), r.height.saturating_sub(frame_height)));
            let was_there = saved.as_ref().and_then(|s| s.rect).is_some_and(|s| (s.x, s.y) == (r.x, r.y));
            let how = if was_there { "restored to" } else { "centred (its saved place is off every monitor) at" };
            format!("window: {how} {}x{} at {},{}", r.width, r.height, r.x, r.y)
        }
    };
    let _ = window.show();
    let _ = window.set_focus();
    note
}

/// The window events (`Builder::on_window_event`): the main window's state is saved when it is asked to close, and, if that
/// was not seen, when it is destroyed.
pub fn on_window_event(window: &tauri::Window, event: &WindowEvent) {
    if window.label() != "main" {
        return;
    }
    match event {
        WindowEvent::CloseRequested { .. } => {
            SAVED_ON_CLOSE.store(true, Ordering::SeqCst);
            save(window);
        }
        WindowEvent::Destroyed => {
            if !SAVED_ON_CLOSE.load(Ordering::SeqCst) {
                save(window);
            }
        }
        _ => {}
    }
}

fn save(window: &tauri::Window) {
    let outer = match (window.outer_position(), window.outer_size(), window.scale_factor()) {
        (Ok(at), Ok(size), Ok(scale)) => Some(SavedRect { x: at.x, y: at.y, width: size.width, height: size.height, scale: sane(scale) }),
        _ => None,
    };
    let maximized = window.is_maximized().unwrap_or(false);
    let minimized = window.is_minimized().unwrap_or(false);
    let Some(state) = to_save(maximized, minimized, outer) else { return };
    if let Err(e) = write(&crate::paths::data_dir(), &state) {
        if let Some(app) = window.try_state::<crate::commands::AppState>() {
            app.monitor.log("app", "warn", format!("could not save the window's state: {e}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: i32, y: i32, width: u32, height: u32) -> Rect {
        Rect { x, y, width, height }
    }

    fn screen(x: i32, y: i32, width: u32, height: u32, scale: f64) -> Screen {
        Screen { area: rect(x, y, width, height), scale }
    }

    fn saved(x: i32, y: i32, width: u32, height: u32, scale: f64) -> Saved {
        Saved { maximized: false, rect: Some(SavedRect { x, y, width, height, scale }) }
    }

    /// tauri.conf.json's main window: 1440 x 900, never smaller than 1100 x 680 (logical pixels).
    const CONFIG: Defaults = Defaults { width: 1440.0, height: 900.0, min_width: 1100.0, min_height: 680.0 };

    fn full_hd() -> Vec<Screen> {
        vec![screen(0, 0, 1920, 1080, 1.0)]
    }

    #[test]
    fn with_no_saved_state_the_window_opens_maximized() {
        assert_eq!(place(None, &full_hd(), CONFIG), Placement::Maximize);
    }

    #[test]
    fn a_window_that_was_maximized_opens_maximized() {
        assert_eq!(place(Some(&Saved { maximized: true, rect: None }), &full_hd(), CONFIG), Placement::Maximize);
        // ... whatever else the file holds
        let both = Saved { maximized: true, rect: Some(SavedRect { x: 10, y: 10, width: 1300, height: 800, scale: 1.0 }) };
        assert_eq!(place(Some(&both), &full_hd(), CONFIG), Placement::Maximize);
    }

    #[test]
    fn a_state_that_says_not_maximized_but_has_no_rectangle_is_no_state_at_all() {
        assert_eq!(place(Some(&Saved { maximized: false, rect: None }), &full_hd(), CONFIG), Placement::Maximize);
        // a rectangle with no area cannot be a window
        assert_eq!(place(Some(&saved(0, 0, 0, 800, 1.0)), &full_hd(), CONFIG), Placement::Maximize);
        assert_eq!(place(Some(&saved(0, 0, 1300, 0, 1.0)), &full_hd(), CONFIG), Placement::Maximize);
    }

    #[test]
    fn with_no_monitor_to_check_against_the_window_opens_maximized() {
        assert_eq!(place(Some(&saved(100, 100, 1300, 800, 1.0)), &[], CONFIG), Placement::Maximize);
    }

    #[test]
    fn a_window_that_was_left_on_screen_comes_back_where_it_was() {
        assert_eq!(place(Some(&saved(120, 80, 1500, 950, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(120, 80, 1500, 950)));
    }

    #[test]
    fn a_window_on_the_second_monitor_comes_back_there_including_a_monitor_left_of_the_first() {
        let screens = vec![screen(0, 0, 1920, 1080, 1.0), screen(-1280, 0, 1280, 1024, 1.0), screen(1920, -200, 2560, 1440, 1.5)];
        assert_eq!(place(Some(&saved(-1200, 40, 1100, 700, 1.0)), &screens, CONFIG), Placement::Window(rect(-1200, 40, 1100, 700)));
        assert_eq!(place(Some(&saved(2000, 0, 1700, 1100, 1.5)), &screens, CONFIG), Placement::Window(rect(2000, 0, 1700, 1100)));
    }

    /// The monitor it was on is gone: centred, at the configuration's size, on the primary monitor.
    #[test]
    fn a_window_that_is_off_every_screen_is_centred_at_the_default_size() {
        // the 1920-wide monitor it was on has been unplugged
        assert_eq!(place(Some(&saved(2300, 100, 1300, 800, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(240, 90, 1440, 900)));
        // above the top edge, and far below the bottom one
        assert_eq!(place(Some(&saved(100, -5000, 1300, 800, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(240, 90, 1440, 900)));
        assert_eq!(place(Some(&saved(100, 4000, 1300, 800, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(240, 90, 1440, 900)));
    }

    /// The default size is in logical pixels: on a 125 % primary monitor it is 1800 x 1125, centred in 2560 x 1440.
    #[test]
    fn the_centred_default_is_scaled_by_the_primary_monitors_factor_and_never_larger_than_it() {
        let hidpi = vec![screen(0, 0, 2560, 1440, 1.25)];
        assert_eq!(place(Some(&saved(9000, 0, 1300, 800, 1.25)), &hidpi, CONFIG), Placement::Window(rect(380, 157, 1800, 1125)));
        // a small laptop screen: the window is the screen's size, not a title bar above the top edge
        let laptop = vec![screen(0, 0, 1366, 768, 1.0)];
        assert_eq!(place(Some(&saved(9000, 0, 1300, 800, 1.0)), &laptop, CONFIG), Placement::Window(rect(0, 0, 1366, 768)));
        // the primary monitor is the first one, wherever it is on the desktop
        let primary_right = vec![screen(1920, 0, 1920, 1080, 1.0), screen(0, 0, 1920, 1080, 1.0)];
        assert_eq!(place(Some(&saved(-9000, 0, 1300, 800, 1.0)), &primary_right, CONFIG), Placement::Window(rect(2160, 90, 1440, 900)));
    }

    /// At least 100 x 100 physical pixels of the window have to be on a monitor, in both directions.
    #[test]
    fn a_window_has_to_overlap_a_monitor_by_100_by_100_pixels_to_stay_where_it_was() {
        let screens = full_hd();
        // 100 wide, 100 high on the right edge: kept
        assert_eq!(place(Some(&saved(1820, 980, 1300, 800, 1.0)), &screens, CONFIG), Placement::Window(rect(1820, 980, 1300, 800)));
        // 99 wide: not
        assert_eq!(place(Some(&saved(1821, 100, 1300, 800, 1.0)), &screens, CONFIG), Placement::Window(rect(240, 90, 1440, 900)));
        // 99 high: not
        assert_eq!(place(Some(&saved(100, 981, 1300, 800, 1.0)), &screens, CONFIG), Placement::Window(rect(240, 90, 1440, 900)));
        // a window that straddles two monitors is on the one it mostly covers (70 px of the first, 1,230 of the second)
        let two = vec![screen(0, 0, 1920, 1080, 1.0), screen(1920, 0, 1920, 1080, 1.0)];
        assert_eq!(place(Some(&saved(1850, 100, 1300, 800, 1.0)), &two, CONFIG), Placement::Window(rect(1850, 100, 1300, 800)));
    }

    /// A size below the configuration's minimum is raised to it (the minimum is in logical pixels: the saved scale turns it
    /// into the physical pixels the rectangle is in).
    #[test]
    fn a_size_below_the_minimum_is_raised_to_it() {
        assert_eq!(place(Some(&saved(100, 100, 400, 300, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(100, 100, 1100, 680)));
        // one dimension only
        assert_eq!(place(Some(&saved(100, 100, 1600, 300, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(100, 100, 1600, 680)));
        // at 125 %: 1100 x 680 logical is 1375 x 850 physical
        let hidpi = vec![screen(0, 0, 2560, 1440, 1.25)];
        assert_eq!(place(Some(&saved(100, 100, 400, 300, 1.25)), &hidpi, CONFIG), Placement::Window(rect(100, 100, 1375, 850)));
        // exactly the minimum is not touched
        assert_eq!(place(Some(&saved(100, 100, 1100, 680, 1.0)), &full_hd(), CONFIG), Placement::Window(rect(100, 100, 1100, 680)));
    }

    /// A scale that is zero, negative or not a number cannot scale anything: it counts as 1.
    #[test]
    fn a_scale_that_means_nothing_counts_as_one() {
        for scale in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            assert_eq!(place(Some(&saved(100, 100, 400, 300, scale)), &full_hd(), CONFIG), Placement::Window(rect(100, 100, 1100, 680)), "{scale}");
        }
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lorren-window-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_missing_or_unreadable_file_is_no_state() {
        let dir = scratch("unreadable");
        assert_eq!(read(&dir), None, "no folder");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(read(&dir), None, "no file");
        for (what, text) in [
            ("empty", ""),
            ("not JSON", "this is not json"),
            ("cut short", "{\"maximized\": fal"),
            ("the wrong shape", "[1, 2, 3]"),
            ("a field of the wrong type", "{\"maximized\": \"yes\"}"),
            ("no maximized field", "{\"rect\": null}"),
            ("a rectangle of the wrong type", "{\"maximized\": false, \"rect\": {\"x\": \"left\"}}"),
        ] {
            std::fs::write(dir.join(FILE), text).unwrap();
            assert_eq!(read(&dir), None, "{what}");
        }
        // the file is a folder
        std::fs::remove_file(dir.join(FILE)).unwrap();
        std::fs::create_dir(dir.join(FILE)).unwrap();
        assert_eq!(read(&dir), None, "a folder where the file should be");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_is_written_is_read_back_and_replaces_what_was_there() {
        let dir = scratch("roundtrip");
        // the folder does not exist yet (the first close of a first run may be before anything else made it)
        let first = saved(120, 80, 1500, 950, 1.25);
        write(&dir, &first).unwrap();
        assert_eq!(read(&dir), Some(first));
        let maximized = Saved { maximized: true, rect: None };
        write(&dir, &maximized).unwrap();
        assert_eq!(read(&dir), Some(maximized));
        // the file is plain JSON a person can read, and nothing else is left in the folder
        write(&dir, &saved(-5, 7, 1200, 700, 1.0)).unwrap();
        let text = std::fs::read_to_string(dir.join(FILE)).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json, serde_json::json!({"maximized": false, "rect": {"x": -5, "y": 7, "width": 1200, "height": 700, "scale": 1.0}}));
        let names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(names, [FILE], "no temporary file is left behind");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_write_that_cannot_be_made_is_an_error_and_leaves_the_old_file_alone() {
        let dir = scratch("blocked");
        let old = saved(1, 2, 1300, 800, 1.0);
        write(&dir, &old).unwrap();
        // a folder where the temporary file would go
        std::fs::create_dir(dir.join(format!("{FILE}.tmp"))).unwrap();
        assert!(write(&dir, &saved(9, 9, 1400, 900, 1.0)).is_err());
        assert_eq!(read(&dir), Some(old), "the file that was there is untouched");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_closing_window_saves_its_state_and_a_minimized_one_saves_nothing() {
        let outer = SavedRect { x: 120, y: 80, width: 1500, height: 950, scale: 1.25 };
        assert_eq!(to_save(false, false, Some(outer)), Some(Saved { maximized: false, rect: Some(outer) }));
        // maximized: the flag only, the rectangle of a maximized window is the screen
        assert_eq!(to_save(true, false, Some(outer)), Some(Saved { maximized: true, rect: None }));
        assert_eq!(to_save(true, false, None), Some(Saved { maximized: true, rect: None }));
        // minimized: the file stays as it was (a minimized window reports a position off the desktop)
        assert_eq!(to_save(false, true, Some(outer)), None);
        assert_eq!(to_save(true, true, Some(outer)), None);
        // not maximized and no rectangle to be had: nothing to save
        assert_eq!(to_save(false, false, None), None);
    }
}
