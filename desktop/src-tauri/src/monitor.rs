//! Request records and the log ring (spec §3.4; the Requests and Logs tabs). In memory and
//! capped; log lines also go to a plain file so a crash leaves a trail.

use crate::store::now_ms;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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

pub struct Monitor {
    seq: AtomicU64,
    logs: Mutex<VecDeque<LogLine>>,
    requests: Mutex<VecDeque<RequestRecord>>,
    file: Option<Mutex<File>>,
}

impl Monitor {
    pub fn new(log_file: Option<PathBuf>) -> Arc<Monitor> {
        let file = log_file
            .and_then(|p| {
                if let Some(d) = p.parent() {
                    let _ = std::fs::create_dir_all(d);
                }
                OpenOptions::new().create(true).append(true).open(p).ok()
            })
            .map(Mutex::new);
        Arc::new(Monitor {
            seq: AtomicU64::new(0),
            logs: Mutex::new(VecDeque::new()),
            requests: Mutex::new(VecDeque::new()),
            file,
        })
    }

    pub fn log(&self, source: &str, level: &str, text: impl Into<String>) {
        let line = LogLine {
            seq: self.seq.fetch_add(1, Ordering::SeqCst) + 1,
            time_ms: now_ms(),
            source: source.into(),
            level: level.into(),
            text: text.into(),
        };
        if let Some(f) = &self.file {
            if let Ok(mut f) = f.lock() {
                let _ = writeln!(f, "{} {:<7} {:<5} {}", line.time_ms, line.source, line.level, line.text);
            }
        }
        let mut logs = self.logs.lock().unwrap();
        logs.push_back(line);
        while logs.len() > MAX_LOGS {
            logs.pop_front();
        }
    }

    pub fn logs_after(&self, seq: u64) -> Vec<LogLine> {
        self.logs.lock().unwrap().iter().filter(|l| l.seq > seq).cloned().collect()
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
        let mut q = self.requests.lock().unwrap();
        q.push_back(record);
        while q.len() > MAX_REQUESTS {
            q.pop_front();
        }
        id
    }

    pub fn update_request(&self, id: &str, f: impl FnOnce(&mut RequestRecord)) {
        if let Some(r) = self.requests.lock().unwrap().iter_mut().rev().find(|r| r.id == id) {
            f(r);
        }
    }

    pub fn requests(&self) -> Vec<RequestRecord> {
        self.requests.lock().unwrap().iter().rev().cloned().collect()
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
}
