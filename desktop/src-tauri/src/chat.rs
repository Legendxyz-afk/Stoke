//! The chat service (spec §3.2): turns a chat into engine requests, streams the reply into
//! the UI and the store, and handles Stop, crashes, the context limit and regenerate.

use crate::budget::{self, Turn};
use crate::engine::supervisor::{EngineError, ModelRef, Supervisor};
use crate::library::ModelInfo;
use crate::monitor::Monitor;
use crate::sse::{ChatChunk, SseEvent, SseParser, Usage};
use crate::store::{Message, MsgStats, MsgStatus, Role, Store};
use futures_util::StreamExt;
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use ts_rs::TS;

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct ReplyStats {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number")]
    pub total_ms: u64,
    pub tokens_per_s: Option<f32>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    Started { message_id: String, user_message_id: String, dropped_turns: u32, request_id: String },
    Token { text: String },
    Finished { stats: ReplyStats },
    Stopped { stats: ReplyStats },
    Interrupted { reason: String },
    Error { message: String },
}

pub struct ChatSettings {
    pub history_budget: u32,
    pub reply_cap: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    /// No bytes for this long means the engine is not responding (it pings every 15 s).
    pub liveness: Duration,
    pub save_every: Duration,
}

impl Default for ChatSettings {
    fn default() -> Self {
        ChatSettings {
            history_budget: budget::HISTORY_BUDGET,
            reply_cap: budget::REPLY_CAP,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            liveness: Duration::from_secs(45),
            save_every: Duration::from_millis(500),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("a reply is already being written in this chat")]
    Busy,
    #[error("no such model: {0}")]
    UnknownModel(String),
    #[error("{}", cannot_run_message(.0, .1))]
    NotRunnable(String, String),
    #[error("this message is about {0} tokens, more than a {1}-token context can take")]
    TooLong(u32, u32),
    #[error("there is no reply to regenerate")]
    NothingToRegenerate,
    #[error(transparent)]
    Store(#[from] rusqlite::Error),
}

/// "<name> cannot run: <reason>", for a chat that picks such a model and for a model switch (Task 11).
pub(crate) fn cannot_run_message(name: &str, reason: &str) -> String {
    format!("{name} cannot run: {reason}")
}

pub fn model_ref(m: &ModelInfo) -> ModelRef {
    ModelRef {
        id: m.id.clone(),
        path: PathBuf::from(&m.path),
        file_bytes: m.size_bytes,
        kv_bytes_per_token: m.kv_bytes_per_token,
    }
}

enum Outcome {
    Done,
    Stopped,
    Interrupted(String),
}

pub struct ChatService {
    pub store: Arc<Mutex<Store>>,
    pub supervisor: Arc<Supervisor>,
    pub library: Arc<RwLock<Vec<ModelInfo>>>,
    pub monitor: Arc<Monitor>,
    pub settings: ChatSettings,
    http: reqwest::Client,
    active: Mutex<HashMap<String, CancellationToken>>,
}

struct ActiveGuard<'a> {
    svc: &'a ChatService,
    chat_id: String,
}

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.svc.active.lock().unwrap().remove(&self.chat_id);
    }
}

fn turns(path: &[Message]) -> Vec<Turn> {
    path.iter()
        .filter(|m| m.status != MsgStatus::Error && !m.content.is_empty())
        .map(|m| Turn { role: m.role.as_str(), content: m.content.clone() })
        .collect()
}

impl ChatService {
    pub fn new(
        store: Arc<Mutex<Store>>,
        supervisor: Arc<Supervisor>,
        library: Arc<RwLock<Vec<ModelInfo>>>,
        monitor: Arc<Monitor>,
        settings: ChatSettings,
    ) -> Arc<ChatService> {
        Arc::new(ChatService {
            store,
            supervisor,
            library,
            monitor,
            settings,
            // The engine closes idle keep-alive connections after 60 s; reqwest's default would reuse a dead one.
            http: reqwest::Client::builder()
                .no_proxy()
                .pool_idle_timeout(Duration::from_secs(30))
                .build()
                .expect("http client"),
            active: Mutex::new(HashMap::new()),
        })
    }

    pub fn stop(&self, chat_id: &str) {
        if let Some(t) = self.active.lock().unwrap().get(chat_id) {
            t.cancel();
        }
    }

    /// Starts the engine on `model`, or switches it (spec §3.1, §5). `ensure_model` is not cancel-safe, so it runs in a
    /// task of its own and dropping the returned handle never drops it (R39). When the engine refuses the file, the
    /// model is marked CAN'T RUN with the engine's own line (R38); the supervisor has already started the previous
    /// model again. Uses `tokio::spawn`: call it inside a tokio runtime.
    pub fn start_model(&self, model: &ModelInfo) -> tokio::task::JoinHandle<Result<u16, EngineError>> {
        let supervisor = self.supervisor.clone();
        let library = self.library.clone();
        let m = model.clone();
        tokio::spawn(async move {
            let started = supervisor.ensure_model(&model_ref(&m)).await;
            // Only the engine's own refusal of the file marks the model: a start that ended with a line of the
            // engine's own in `LoadFailed`. An engine that died without a word (killed, or crashed silently), a memory
            // refusal, a load timeout and a spawn failure are not the file's fault and can pass later (R47).
            if let Err(EngineError::LoadFailed(Some(line))) = &started {
                if let Some(entry) = library.write().unwrap().iter_mut().find(|e| e.id == m.id) {
                    entry.runnable = false;
                    entry.reason = Some(line.clone());
                }
            }
            started
        })
    }

    fn begin(&self, chat_id: &str) -> Result<(ActiveGuard<'_>, CancellationToken), ChatError> {
        let mut a = self.active.lock().unwrap();
        if a.contains_key(chat_id) {
            return Err(ChatError::Busy);
        }
        let t = CancellationToken::new();
        a.insert(chat_id.to_string(), t.clone());
        Ok((ActiveGuard { svc: self, chat_id: chat_id.to_string() }, t))
    }

    fn model(&self, id: &str) -> Result<ModelInfo, ChatError> {
        let lib = self.library.read().unwrap();
        let m = lib.iter().find(|m| m.id == id).cloned().ok_or_else(|| ChatError::UnknownModel(id.into()))?;
        if !m.runnable {
            return Err(ChatError::NotRunnable(m.name.clone(), m.reason.clone().unwrap_or_default()));
        }
        Ok(m)
    }

    pub async fn send(
        self: &Arc<Self>,
        chat_id: &str,
        text: &str,
        sink: impl Fn(StreamEvent) + Send + Sync,
    ) -> Result<(), ChatError> {
        let (_guard, cancel) = self.begin(chat_id)?;
        let (chat, path) = {
            let s = self.store.lock().unwrap();
            (s.get_chat(chat_id)?, s.active_path(chat_id)?)
        };
        let model = self.model(&chat.model_id)?;
        let system = chat.system_prompt.clone().or_else(|| model.system_prompt.clone());
        if let Some(est) = budget::message_too_long(system.as_deref(), text, model.ctx_engine) {
            return Err(ChatError::TooLong(est, model.ctx_engine));
        }
        let parent = path.last().map(|m| m.id.clone());
        let user = self
            .store
            .lock()
            .unwrap()
            .append_message(chat_id, parent.as_deref(), Role::User, text, MsgStatus::Done, None)?;
        self.stream(chat_id, &model, system, turns(&path), text.to_string(), &user, &sink, &cancel).await
    }

    pub async fn regenerate(self: &Arc<Self>, chat_id: &str, sink: impl Fn(StreamEvent) + Send + Sync) -> Result<(), ChatError> {
        let (_guard, cancel) = self.begin(chat_id)?;
        let (chat, path) = {
            let s = self.store.lock().unwrap();
            (s.get_chat(chat_id)?, s.active_path(chat_id)?)
        };
        let pos = path.iter().rposition(|m| m.role == Role::User).ok_or(ChatError::NothingToRegenerate)?;
        let user = path[pos].clone();
        let model = self.model(&chat.model_id)?;
        let system = chat.system_prompt.clone().or_else(|| model.system_prompt.clone());
        self.stream(chat_id, &model, system, turns(&path[..pos]), user.content.clone(), &user, &sink, &cancel).await
    }

    fn request_body(&self, messages: &[(&str, String)], max_tokens: u32) -> serde_json::Value {
        let msgs: Vec<_> = messages.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect();
        serde_json::json!({
            "model": "local",
            "messages": msgs,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": max_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "repeat_penalty": self.settings.repeat_penalty,
        })
    }

    fn finish_early(&self, msg: &str, req: &str, sink: &(impl Fn(StreamEvent) + Send + Sync), status: MsgStatus, text: String) -> Result<(), ChatError> {
        let stats = MsgStats { error: (status == MsgStatus::Error).then(|| text.clone()), ..Default::default() };
        self.store.lock().unwrap().finish_message(msg, status, &stats)?;
        let label = if status == MsgStatus::Stopped { "stopped" } else { "error" };
        self.monitor.update_request(req, |r| r.status = label.into());
        if status == MsgStatus::Stopped {
            sink(StreamEvent::Stopped {
                stats: ReplyStats { prompt_tokens: None, completion_tokens: None, ttft_ms: None, total_ms: 0, tokens_per_s: None, finish_reason: None },
            });
        } else {
            self.monitor.log("app", "error", text.clone());
            sink(StreamEvent::Error { message: text });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream(
        &self,
        chat_id: &str,
        model: &ModelInfo,
        system: Option<String>,
        history: Vec<Turn>,
        user_text: String,
        user_msg: &Message,
        sink: &(impl Fn(StreamEvent) + Send + Sync),
        cancel: &CancellationToken,
    ) -> Result<(), ChatError> {
        let asst = self.store.lock().unwrap().append_message(
            chat_id,
            Some(&user_msg.id),
            Role::Assistant,
            "",
            MsgStatus::Streaming,
            Some(&model.id),
        )?;
        let req = self.monitor.start_request("chat", "/v1/chat/completions", &model.name);

        // 1. The engine: start or switch it, or wait for a start already under way. `ensure_model` is not
        // cancel-safe (R39), so it runs in the task `start_model` spawns and Stop only lets go of the handle:
        // the load finishes in the background and the engine ends up serving.
        let mut start = self.start_model(model);
        let port = tokio::select! {
            r = &mut start => match r {
                Ok(Ok(p)) => p,
                Ok(Err(e)) => return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, e.to_string()),
                Err(e) => return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, format!("the engine start failed: {e}")),
            },
            _ = cancel.cancelled() => return self.finish_early(&asst.id, &req, sink, MsgStatus::Stopped, String::new()),
        };

        // 2. The request, with one retry when the engine says the context is full.
        let (mut kept, mut dropped) = budget::select_history(&history, self.settings.history_budget);
        let mut forced_max: Option<u32> = None;
        let mut retried = false;
        let t0 = Instant::now();
        let resp = loop {
            let mut messages: Vec<(&str, String)> = Vec::new();
            if let Some(s) = &system {
                messages.push(("system", s.clone()));
            }
            messages.extend(kept.iter().map(|t| (t.role, t.content.clone())));
            messages.push(("user", user_text.clone()));
            let estimate: u32 = messages.iter().map(|(_, c)| budget::estimate_tokens(c)).sum();
            let max_tokens = match forced_max.or_else(|| budget::max_tokens_for(model.ctx_engine, estimate, self.settings.reply_cap)) {
                Some(m) => m,
                None => {
                    return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, "this chat no longer fits the context; start a new chat".into())
                }
            };
            let send = self.http.post(format!("http://127.0.0.1:{port}/v1/chat/completions")).json(&self.request_body(&messages, max_tokens)).send();
            let resp = tokio::select! {
                r = send => r,
                _ = cancel.cancelled() => return self.finish_early(&asst.id, &req, sink, MsgStatus::Stopped, String::new()),
            };
            let resp = match resp {
                Ok(r) => r,
                Err(e) => return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, format!("could not reach the engine: {e}")),
            };
            if resp.status().is_success() {
                break resp;
            }
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
            let is_ctx = parsed["error"]["type"] == "context_length_exceeded";
            let counts = parsed["error"]["message"].as_str().and_then(budget::parse_context_error);
            match (is_ctx, counts, retried) {
                (true, Some((n, _m, c)), false) => {
                    retried = true;
                    // The engine counted n real tokens where we estimated `estimate`; scale our
                    // estimates by that ratio so the retry is sized in the engine's units.
                    let scale = n as f32 / estimate.max(1) as f32;
                    let real = |est: u32| (est as f32 * scale).ceil() as u32;
                    let mut est_after = estimate;
                    if c.saturating_sub(n + 8) < budget::MIN_REPLY {
                        while !kept.is_empty() && real(est_after) + budget::MIN_REPLY + 8 > c {
                            est_after = est_after.saturating_sub(budget::estimate_tokens(&kept.remove(0).content));
                            dropped += 1;
                        }
                        while !kept.is_empty() && kept[0].role != "user" {
                            est_after = est_after.saturating_sub(budget::estimate_tokens(&kept.remove(0).content));
                            dropped += 1;
                        }
                    }
                    let n_after = real(est_after).min(n);
                    forced_max = Some(c.saturating_sub(n_after + 8).clamp(budget::MIN_REPLY, self.settings.reply_cap));
                    self.monitor.log("app", "warn", format!("context full ({n} of {c} tokens); trimmed and retried once"));
                }
                _ => {
                    return self.finish_early(&asst.id, &req, sink, MsgStatus::Error, format!("the engine refused the request ({status}): {body}"))
                }
            }
        };

        // 3. The stream.
        sink(StreamEvent::Started {
            message_id: asst.id.clone(),
            user_message_id: user_msg.id.clone(),
            dropped_turns: dropped,
            request_id: req.clone(),
        });
        self.monitor.update_request(&req, |r| r.dropped_turns = dropped);
        let mut body = resp.bytes_stream();
        let mut parser = SseParser::new();
        let mut content = String::new();
        let mut first: Option<Instant> = None;
        let mut usage: Option<Usage> = None;
        let mut finish: Option<String> = None;
        let mut last_save = Instant::now();
        let outcome = loop {
            let next = tokio::select! {
                n = tokio::time::timeout(self.settings.liveness, body.next()) => n,
                _ = cancel.cancelled() => break Outcome::Stopped,
            };
            match next {
                Err(_) => break Outcome::Interrupted("the engine stopped responding".into()),
                Ok(None) => {
                    break if finish.is_some() { Outcome::Done } else { Outcome::Interrupted("the engine closed the stream before finishing".into()) }
                }
                Ok(Some(Err(e))) => break Outcome::Interrupted(format!("the stream broke: {e}")),
                Ok(Some(Ok(bytes))) => {
                    let mut done = false;
                    for ev in parser.push(&bytes) {
                        match ev {
                            SseEvent::Comment => {}
                            SseEvent::Done => done = true,
                            SseEvent::Data(json) => {
                                let Ok(c) = serde_json::from_str::<ChatChunk>(&json) else { continue };
                                if c.usage.is_some() {
                                    usage = c.usage;
                                }
                                for ch in c.choices {
                                    if let Some(t) = ch.delta.content.filter(|t| !t.is_empty()) {
                                        first.get_or_insert_with(Instant::now);
                                        content.push_str(&t);
                                        sink(StreamEvent::Token { text: t });
                                    }
                                    if ch.finish_reason.is_some() {
                                        finish = ch.finish_reason;
                                    }
                                }
                            }
                        }
                    }
                    if last_save.elapsed() >= self.settings.save_every {
                        let _ = self.store.lock().unwrap().update_content(&asst.id, &content);
                        last_save = Instant::now();
                    }
                    if done {
                        break Outcome::Done;
                    }
                }
            }
        };
        drop(body); // closing the connection is what cancels generation in the engine

        // 4. Save and report.
        let total_ms = t0.elapsed().as_millis() as u64;
        let ttft_ms = first.map(|f| f.duration_since(t0).as_millis() as u64);
        let completion = usage.map(|u| u.completion_tokens);
        let tokens_per_s = match (first, completion) {
            (Some(f), Some(n)) if n > 1 => Some((n - 1) as f32 / f.elapsed().as_secs_f32().max(0.001)),
            _ => None,
        };
        let stats = ReplyStats { prompt_tokens: usage.map(|u| u.prompt_tokens), completion_tokens: completion, ttft_ms, total_ms, tokens_per_s, finish_reason: finish };
        let (status, event, label) = match outcome {
            Outcome::Done => (MsgStatus::Done, StreamEvent::Finished { stats: stats.clone() }, "ok"),
            Outcome::Stopped => (MsgStatus::Stopped, StreamEvent::Stopped { stats: stats.clone() }, "stopped"),
            Outcome::Interrupted(reason) => {
                self.monitor.log("app", "warn", format!("reply interrupted: {reason}"));
                (MsgStatus::Interrupted, StreamEvent::Interrupted { reason }, "interrupted")
            }
        };
        {
            let s = self.store.lock().unwrap();
            s.update_content(&asst.id, &content)?;
            s.finish_message(
                &asst.id,
                status,
                &MsgStats { prompt_tokens: stats.prompt_tokens, completion_tokens: completion, ttft_ms, total_ms: Some(total_ms), error: None },
            )?;
        }
        self.monitor.update_request(&req, |r| {
            r.prompt_tokens = stats.prompt_tokens;
            r.completion_tokens = completion;
            r.ttft_ms = ttft_ms;
            r.total_ms = Some(total_ms);
            r.tokens_per_s = tokens_per_s;
            r.status = label.into();
        });
        sink(event);
        Ok(())
    }
}
