//! Request records and the log ring (spec §3.4; the Requests and Logs tabs). In memory and
//! capped; log lines also go to a plain file so a crash leaves a trail.

use crate::store::now_ms;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use ts_rs::TS;

pub const MAX_REQUESTS: usize = 500;
pub const MAX_LOGS: usize = 5_000;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct LogLine {
    #[ts(type = "number")]
    pub seq: u64,
    #[ts(type = "number")]
    pub time_ms: i64,
    /// "app", "engine" or "updater"
    pub source: String,
    /// "info", "warn" or "error"
    pub level: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct RequestRecord {
    pub id: String,
    #[ts(type = "number")]
    pub started_ms: i64,
    pub source: String,
    pub endpoint: String,
    pub model: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number | null")]
    pub total_ms: Option<u64>,
    pub tokens_per_s: Option<f32>,
    /// "streaming", "ok", "stopped", "interrupted" or "error"
    pub status: String,
    pub dropped_turns: u32,
}

struct LogRing {
    seq: u64,
    lines: VecDeque<LogLine>,
}

/// Locks recover from poisoning: a panic in one holder (e.g. a caller closure in
/// update_request) must not turn every later log or request call into a panic.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Monitor {
    logs: Mutex<LogRing>,
    requests: Mutex<VecDeque<RequestRecord>>,
    /// Requests started since the app began. `requests` keeps only the newest MAX_REQUESTS, so its length stops at 500.
    started: AtomicU64,
    file: Option<Mutex<File>>,
}

impl Monitor {
    pub fn new(log_file: Option<PathBuf>) -> Arc<Monitor> {
        let mut open_error = None;
        let file = log_file.and_then(|p| {
            if let Some(d) = p.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            match OpenOptions::new().create(true).append(true).open(&p) {
                Ok(f) => Some(Mutex::new(f)),
                Err(e) => {
                    open_error = Some(format!("could not open log file {}: {e}", p.display()));
                    None
                }
            }
        });
        let m = Arc::new(Monitor {
            logs: Mutex::new(LogRing { seq: 0, lines: VecDeque::new() }),
            requests: Mutex::new(VecDeque::new()),
            started: AtomicU64::new(0),
            file,
        });
        if let Some(msg) = open_error {
            m.log("app", "warn", msg);
        }
        m
    }

    pub fn log(&self, source: &str, level: &str, text: impl Into<String>) {
        // Seq assignment, the ring push and the file write share one critical section, taken in
        // the order logs -> file (never the reverse), so the ring and the file are in seq order.
        let mut ring = lock(&self.logs);
        ring.seq += 1;
        let line = LogLine {
            seq: ring.seq,
            time_ms: now_ms(),
            source: source.into(),
            level: level.into(),
            text: text.into(),
        };
        if let Some(f) = &self.file {
            let mut f = lock(f);
            let _ = writeln!(f, "{} {:<7} {:<5} {}", line.time_ms, line.source, line.level, line.text);
        }
        ring.lines.push_back(line);
        while ring.lines.len() > MAX_LOGS {
            ring.lines.pop_front();
        }
    }

    pub fn logs_after(&self, seq: u64) -> Vec<LogLine> {
        lock(&self.logs).lines.iter().filter(|l| l.seq > seq).cloned().collect()
    }

    pub fn start_request(&self, source: &str, endpoint: &str, model: &str) -> String {
        let id = format!("req_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let record = RequestRecord {
            id: id.clone(),
            started_ms: now_ms(),
            source: source.into(),
            endpoint: endpoint.into(),
            model: model.into(),
            prompt_tokens: None,
            completion_tokens: None,
            ttft_ms: None,
            total_ms: None,
            tokens_per_s: None,
            status: "streaming".into(),
            dropped_turns: 0,
        };
        self.started.fetch_add(1, Ordering::Relaxed);
        let mut q = lock(&self.requests);
        q.push_back(record);
        while q.len() > MAX_REQUESTS {
            q.pop_front();
        }
        id
    }

    pub fn update_request(&self, id: &str, f: impl FnOnce(&mut RequestRecord)) {
        if let Some(r) = lock(&self.requests).iter_mut().rev().find(|r| r.id == id) {
            f(r);
        }
    }

    pub fn requests(&self) -> Vec<RequestRecord> {
        lock(&self.requests).iter().rev().cloned().collect()
    }

    /// How many requests have been started this session, however many of them `requests()` still holds.
    pub fn request_count(&self) -> u64 {
        self.started.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_are_numbered_and_can_be_read_after_a_sequence_number() {
        let m = Monitor::new(None);
        m.log("app", "info", "one");
        m.log("engine", "warn", "two");
        m.log("app", "info", "three");
        let after = m.logs_after(1);
        assert_eq!(after.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), vec!["two", "three"]);
        assert_eq!(after[0].source, "engine");
        assert_eq!(after[0].level, "warn");
    }

    #[test]
    fn the_log_ring_is_capped() {
        let m = Monitor::new(None);
        for i in 0..MAX_LOGS + 10 {
            m.log("app", "info", format!("line {i}"));
        }
        let all = m.logs_after(0);
        assert_eq!(all.len(), MAX_LOGS);
        assert_eq!(all[0].text, "line 10");
    }

    #[test]
    fn requests_are_listed_newest_first_and_can_be_updated() {
        let m = Monitor::new(None);
        let a = m.start_request("chat", "/v1/chat/completions", "nexus");
        let b = m.start_request("chat", "/v1/chat/completions", "qwen");
        m.update_request(&a, |r| {
            r.status = "ok".into();
            r.ttft_ms = Some(640);
        });
        let list = m.requests();
        assert_eq!(list[0].id, b);
        assert_eq!(list[1].status, "ok");
        assert_eq!(list[1].ttft_ms, Some(640));
        assert!(a.starts_with("req_") && a.len() == 12);
    }

    #[test]
    fn logs_also_go_to_the_file() {
        let path = std::env::temp_dir().join(format!("lorren-monitor-{}.log", uuid::Uuid::new_v4().simple()));
        let _ = std::fs::remove_file(&path);
        let m = Monitor::new(Some(path.clone()));
        m.log("app", "info", "written to disk");
        drop(m);
        let contents = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(contents.contains("written to disk"));
    }

    #[test]
    fn a_panic_inside_update_request_does_not_poison_the_monitor() {
        let m = Monitor::new(None);
        let id = m.start_request("chat", "/v1/chat/completions", "nexus");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.update_request(&id, |_| panic!("boom"));
        }));
        assert!(r.is_err());
        m.log("app", "info", "after panic");
        let id2 = m.start_request("chat", "/v1/chat/completions", "qwen");
        assert_eq!(m.requests()[0].id, id2);
        assert_eq!(m.logs_after(0).last().unwrap().text, "after panic");
    }

    #[test]
    fn requests_are_capped_at_max_requests_and_the_oldest_go() {
        let m = Monitor::new(None);
        let ids: Vec<String> = (0..MAX_REQUESTS + 1).map(|_| m.start_request("chat", "/e", "m")).collect();
        let list = m.requests();
        assert_eq!(list.len(), MAX_REQUESTS);
        assert_eq!(list[0].id, *ids.last().unwrap());
        assert!(!list.iter().any(|r| r.id == ids[0]));
    }

    /// The status bar's "N requests" is the number started this session. The list holds the newest MAX_REQUESTS only,
    /// so counting its length stops at 500 while the requests go on.
    #[test]
    fn the_request_count_goes_on_past_the_cap_of_the_list() {
        let m = Monitor::new(None);
        assert_eq!(m.request_count(), 0);
        for _ in 0..MAX_REQUESTS + 5 {
            m.start_request("chat", "/e", "m");
        }
        assert_eq!(m.requests().len(), MAX_REQUESTS);
        assert_eq!(m.request_count(), (MAX_REQUESTS + 5) as u64);
    }

    #[test]
    fn logs_after_a_dropped_range_returns_exactly_the_newer_lines_in_order() {
        let m = Monitor::new(None);
        let total = MAX_LOGS as u64 + 10;
        for i in 0..total {
            m.log("app", "info", format!("line {}", i + 1));
        }
        let seqs: Vec<u64> = m.logs_after(5003).iter().map(|l| l.seq).collect();
        let expected: Vec<u64> = (5004..=total).collect();
        assert_eq!(seqs, expected);
        assert_eq!(m.logs_after(0).first().unwrap().seq, 11);
    }

    #[test]
    fn concurrent_writers_get_strictly_increasing_seqs_in_the_ring() {
        let m = Monitor::new(None);
        let handles: Vec<_> = (0..4)
            .map(|t| {
                let m = m.clone();
                std::thread::spawn(move || {
                    for i in 0..2000 {
                        m.log("engine", "info", format!("t{t} {i}"));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let all = m.logs_after(0);
        assert_eq!(all.len(), MAX_LOGS);
        assert!(all.windows(2).all(|w| w[0].seq < w[1].seq));
        assert_eq!(all.last().unwrap().seq, 8000);
    }

    #[test]
    fn an_unopenable_log_file_is_reported_in_the_log_ring() {
        let base = std::env::temp_dir().join(format!("lorren-monitor-blocker-{}", uuid::Uuid::new_v4().simple()));
        std::fs::write(&base, "not a directory").unwrap();
        let path = base.join("sub").join("crash.log");
        let m = Monitor::new(Some(path.clone()));
        let _ = std::fs::remove_file(&base);
        let lines = m.logs_after(0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].level, "warn");
        assert_eq!(lines[0].source, "app");
        assert!(lines[0].text.contains(&format!("{}", path.display())));
    }
}
