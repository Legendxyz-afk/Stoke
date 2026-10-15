//! Physical and per-process memory, for the load check (spec §3.1) and the top bar (§11).

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

fn status() -> MEMORYSTATUSEX {
    let mut s = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    unsafe {
        let _ = GlobalMemoryStatusEx(&mut s);
    }
    s
}

pub fn available_physical_bytes() -> u64 {
    status().ullAvailPhys
}

pub fn total_physical_bytes() -> u64 {
    status().ullTotalPhys
}

/// (working set, private bytes) of one process.
pub fn process_memory(pid: u32) -> Option<(u64, u64)> {
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut c = PROCESS_MEMORY_COUNTERS_EX {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        let ok = GetProcessMemoryInfo(h, &mut c as *mut _ as *mut PROCESS_MEMORY_COUNTERS, c.cb);
        let _ = CloseHandle(h);
        ok.ok()?;
        Some((c.WorkingSetSize as u64, c.PrivateUsage as u64))
    }
}

/// Working set of `root` and every process below it (the WebView2 processes), except `exclude`
/// (the engine, reported on its own).
pub fn tree_working_set(root: u32, exclude: Option<u32>) -> u64 {
    let mut parent_of = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return 0 };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        if Process32FirstW(snap, &mut e).is_ok() {
            loop {
                parent_of.push((e.th32ProcessID, e.th32ParentProcessID));
                if Process32NextW(snap, &mut e).is_err() {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);
    }
    let mut tree = vec![root];
    let mut i = 0;
    while i < tree.len() {
        let p = tree[i];
        for &(child, parent) in &parent_of {
            if parent == p && child != p && !tree.contains(&child) && Some(child) != exclude {
                tree.push(child);
            }
        }
        i += 1;
    }
    tree.iter().filter_map(|&pid| process_memory(pid)).map(|(ws, _)| ws).sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn this_process_has_memory_and_the_machine_has_more() {
        let (ws, private) = super::process_memory(std::process::id()).unwrap();
        assert!(ws > 0 && private > 0);
        assert!(super::total_physical_bytes() > super::available_physical_bytes());
        assert!(super::tree_working_set(std::process::id(), None) >= ws);
    }
}
