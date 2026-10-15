//! A Windows job object that kills every process in it when its last handle closes, so the
//! engine never outlives the app, even when the app crashes (spec §3.1).

use std::os::windows::io::AsRawHandle;
use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

pub struct Job(HANDLE);

// The handle is only ever used through the Win32 calls below, which are thread-safe.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

fn io(e: windows::core::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
}

impl Job {
    pub fn kill_on_close() -> std::io::Result<Job> {
        unsafe {
            let h = CreateJobObjectW(None, PCWSTR::null()).map_err(io)?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(io)?;
            Ok(Job(h))
        }
    }

    pub fn assign(&self, child: &std::process::Child) -> std::io::Result<()> {
        unsafe { AssignProcessToJobObject(self.0, HANDLE(child.as_raw_handle() as _)).map_err(io) }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.0);
        }
    }
}
