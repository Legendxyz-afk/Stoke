//! Chats and their message trees in SQLite (spec §3.5). Editing or regenerating adds a
//! sibling; a chat shows one path, from the root to its `active_leaf_id`.

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use ts_rs::TS;

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
    fn parse(s: &str) -> Role {
        if s == "user" { Role::User } else { Role::Assistant }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, TS)]
#[ts(export)]
#[serde(rename_all = "snake_case")]
pub enum MsgStatus {
    Done,
    Streaming,
    Stopped,
    Interrupted,
    Error,
}

impl MsgStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MsgStatus::Done => "done",
            MsgStatus::Streaming => "streaming",
            MsgStatus::Stopped => "stopped",
            MsgStatus::Interrupted => "interrupted",
            MsgStatus::Error => "error",
        }
    }
    fn parse(s: &str) -> MsgStatus {
        match s {
            "done" => MsgStatus::Done,
            "streaming" => MsgStatus::Streaming,
            "stopped" => MsgStatus::Stopped,
            "interrupted" => MsgStatus::Interrupted,
            _ => MsgStatus::Error,
        }
    }
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Chat {
    pub id: String,
    pub title: String,
    pub model_id: String,
    pub system_prompt: Option<String>,
    pub active_leaf_id: Option<String>,
    #[ts(type = "number")]
    pub created_ms: i64,
    #[ts(type = "number")]
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Default, Serialize, TS)]
#[ts(export)]
pub struct MsgStats {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    #[ts(type = "number | null")]
    pub ttft_ms: Option<u64>,
    #[ts(type = "number | null")]
    pub total_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, TS)]
#[ts(export)]
pub struct Message {
    pub id: String,
    pub chat_id: String,
    pub parent_id: Option<String>,
    pub role: Role,
    pub content: String,
    pub status: MsgStatus,
    pub model_id: Option<String>,
    pub stats: MsgStats,
    #[ts(type = "number")]
    pub created_ms: i64,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS chats (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, model_id TEXT NOT NULL, system_prompt TEXT,
  active_leaf_id TEXT, created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
  parent_id TEXT REFERENCES messages(id),
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL, status TEXT NOT NULL, model_id TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, ttft_ms INTEGER, total_ms INTEGER, error TEXT,
  created_ms INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, created_ms);
CREATE INDEX IF NOT EXISTS messages_by_parent ON messages(parent_id);
";

const CHAT_COLS: &str = "id, title, model_id, system_prompt, active_leaf_id, created_ms, updated_ms";
const MSG_COLS: &str = "id, chat_id, parent_id, role, content, status, model_id, \
                        prompt_tokens, completion_tokens, ttft_ms, total_ms, error, created_ms";

fn chat_row(r: &Row) -> rusqlite::Result<Chat> {
    Ok(Chat {
        id: r.get(0)?,
        title: r.get(1)?,
        model_id: r.get(2)?,
        system_prompt: r.get(3)?,
        active_leaf_id: r.get(4)?,
        created_ms: r.get(5)?,
        updated_ms: r.get(6)?,
    })
}

fn msg_row(r: &Row) -> rusqlite::Result<Message> {
    let role: String = r.get(3)?;
    let status: String = r.get(5)?;
    Ok(Message {
        id: r.get(0)?,
        chat_id: r.get(1)?,
        parent_id: r.get(2)?,
        role: Role::parse(&role),
        content: r.get(4)?,
        status: MsgStatus::parse(&status),
        model_id: r.get(6)?,
        stats: MsgStats {
            prompt_tokens: r.get(7)?,
            completion_tokens: r.get(8)?,
            ttft_ms: r.get::<_, Option<i64>>(9)?.map(|v| v as u64),
            total_ms: r.get::<_, Option<i64>>(10)?.map(|v| v as u64),
            error: r.get(11)?,
        },
        created_ms: r.get(12)?,
    })
}

fn short_title(text: &str) -> String {
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
    if line.chars().count() <= 40 {
        line.to_string()
    } else {
        format!("{}…", line.chars().take(40).collect::<String>().trim_end())
    }
}

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path) -> rusqlite::Result<Store> {
        let conn = Connection::open(path)?;
        conn.pragma_update_and_check(None, "journal_mode", "WAL", |r| r.get::<_, String>(0))?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> rusqlite::Result<Store> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> rusqlite::Result<Store> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let check: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
        if check != "ok" {
            return Err(rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
                Some(format!("quick_check: {check}")),
            ));
        }
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", 1)?;
        // A reply that was streaming when the app died is interrupted, never "still streaming".
        conn.execute("UPDATE messages SET status = 'interrupted' WHERE status = 'streaming'", [])?;
        Ok(Store { conn })
    }

    pub fn create_chat(&self, model_id: &str, system_prompt: Option<&str>) -> rusqlite::Result<Chat> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        self.conn.execute(
            "INSERT INTO chats (id, title, model_id, system_prompt, active_leaf_id, created_ms, updated_ms) \
             VALUES (?1, 'New chat', ?2, ?3, NULL, ?4, ?4)",
            params![id, model_id, system_prompt, now],
        )?;
        self.get_chat(&id)
    }

    pub fn get_chat(&self, id: &str) -> rusqlite::Result<Chat> {
        self.conn.query_row(&format!("SELECT {CHAT_COLS} FROM chats WHERE id = ?1"), [id], chat_row)
    }

    pub fn list_chats(&self) -> rusqlite::Result<Vec<Chat>> {
        let mut st = self.conn.prepare(&format!("SELECT {CHAT_COLS} FROM chats ORDER BY updated_ms DESC, rowid DESC"))?;
        let rows = st.query_map([], chat_row)?;
        rows.collect()
    }

    pub fn set_chat_model(&self, id: &str, model_id: &str) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE chats SET model_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![id, model_id, now_ms()],
        )?;
        Ok(())
    }

    pub fn append_message(
        &self,
        chat_id: &str,
        parent_id: Option<&str>,
        role: Role,
        content: &str,
        status: MsgStatus,
        model_id: Option<&str>,
    ) -> rusqlite::Result<Message> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = now_ms();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO messages (id, chat_id, parent_id, role, content, status, model_id, created_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![id, chat_id, parent_id, role.as_str(), content, status.as_str(), model_id, now],
        )?;
        tx.execute(
            "UPDATE chats SET active_leaf_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![chat_id, id, now],
        )?;
        if role == Role::User {
            tx.execute(
                "UPDATE chats SET title = ?2 WHERE id = ?1 AND title = 'New chat'",
                params![chat_id, short_title(content)],
            )?;
        }
        tx.commit()?;
        self.get_message(&id)
    }

    pub fn get_message(&self, id: &str) -> rusqlite::Result<Message> {
        self.conn.query_row(&format!("SELECT {MSG_COLS} FROM messages WHERE id = ?1"), [id], msg_row)
    }

    pub fn update_content(&self, id: &str, content: &str) -> rusqlite::Result<()> {
        self.conn.execute("UPDATE messages SET content = ?2 WHERE id = ?1", params![id, content])?;
        Ok(())
    }

    pub fn finish_message(&self, id: &str, status: MsgStatus, s: &MsgStats) -> rusqlite::Result<()> {
        self.conn.execute(
            "UPDATE messages SET status = ?2, prompt_tokens = ?3, completion_tokens = ?4, \
             ttft_ms = ?5, total_ms = ?6, error = ?7 WHERE id = ?1",
            params![
                id,
                status.as_str(),
                s.prompt_tokens,
                s.completion_tokens,
                s.ttft_ms.map(|v| v as i64),
                s.total_ms.map(|v| v as i64),
                s.error
            ],
        )?;
        Ok(())
    }

    pub fn active_path(&self, chat_id: &str) -> rusqlite::Result<Vec<Message>> {
        let mut out = Vec::new();
        let mut cur = self.get_chat(chat_id)?.active_leaf_id;
        while let Some(id) = cur {
            let m = self.get_message(&id)?;
            cur = m.parent_id.clone();
            out.push(m);
        }
        out.reverse();
        Ok(out)
    }

    pub fn siblings(&self, message_id: &str) -> rusqlite::Result<Vec<Message>> {
        let m = self.get_message(message_id)?;
        let mut st = self.conn.prepare(&format!(
            "SELECT {MSG_COLS} FROM messages WHERE chat_id = ?1 AND parent_id IS ?2 AND role = ?3 \
             ORDER BY created_ms, rowid"
        ))?;
        let rows = st.query_map(params![m.chat_id, m.parent_id, m.role.as_str()], msg_row)?;
        rows.collect()
    }

    /// Shows `message_id`'s branch: follows its newest children down to a leaf.
    pub fn set_active_leaf(&self, chat_id: &str, message_id: &str) -> rusqlite::Result<()> {
        let mut leaf = message_id.to_string();
        loop {
            let next: Option<String> = self
                .conn
                .query_row(
                    "SELECT id FROM messages WHERE parent_id = ?1 ORDER BY created_ms DESC, rowid DESC LIMIT 1",
                    [&leaf],
                    |r| r.get(0),
                )
                .optional()?;
            match next {
                Some(n) => leaf = n,
                None => break,
            }
        }
        self.conn.execute(
            "UPDATE chats SET active_leaf_id = ?2, updated_ms = ?3 WHERE id = ?1",
            params![chat_id, leaf, now_ms()],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_with_reply(s: &Store) -> (Chat, Message, Message) {
        let chat = s.create_chat("file:q", None).unwrap();
        let user = s.append_message(&chat.id, None, Role::User, "BTC just broke 68,400. Long or wait?", MsgStatus::Done, None).unwrap();
        let reply = s.append_message(&chat.id, Some(&user.id), Role::Assistant, "Verdict: HOLD.", MsgStatus::Done, Some("file:q")).unwrap();
        (s.get_chat(&chat.id).unwrap(), user, reply)
    }

    #[test]
    fn a_chat_keeps_its_path_and_is_titled_from_the_first_message() {
        let s = Store::open_in_memory().unwrap();
        let (chat, user, reply) = chat_with_reply(&s);
        assert_eq!(chat.title, "BTC just broke 68,400. Long or wait?");
        assert_eq!(chat.active_leaf_id.as_deref(), Some(reply.id.as_str()));
        let path = s.active_path(&chat.id).unwrap();
        assert_eq!(path.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![user.id, reply.id]);
    }

    #[test]
    fn a_long_first_message_gives_a_short_title() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        s.append_message(&chat.id, None, Role::User, &"word ".repeat(40), MsgStatus::Done, None).unwrap();
        let title = s.get_chat(&chat.id).unwrap().title;
        assert!(title.chars().count() <= 41 && title.ends_with('…'), "{title}");
    }

    #[test]
    fn regenerating_adds_a_sibling_and_versions_can_be_switched() {
        let s = Store::open_in_memory().unwrap();
        let (chat, user, first) = chat_with_reply(&s);
        let second = s.append_message(&chat.id, Some(&user.id), Role::Assistant, "Verdict: SELL.", MsgStatus::Done, None).unwrap();
        let sibs = s.siblings(&second.id).unwrap();
        assert_eq!(sibs.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), vec![first.id.clone(), second.id.clone()]);
        assert_eq!(s.active_path(&chat.id).unwrap().last().unwrap().id, second.id);
        s.set_active_leaf(&chat.id, &first.id).unwrap();
        assert_eq!(s.active_path(&chat.id).unwrap().last().unwrap().id, first.id);
    }

    #[test]
    fn streaming_content_and_final_stats_are_saved() {
        let s = Store::open_in_memory().unwrap();
        let chat = s.create_chat("m", None).unwrap();
        let m = s.append_message(&chat.id, None, Role::Assistant, "", MsgStatus::Streaming, None).unwrap();
        s.update_content(&m.id, "Buy at ₹68,400 🚀").unwrap();
        let stats = MsgStats { prompt_tokens: Some(69), completion_tokens: Some(6), ttft_ms: Some(640), total_ms: Some(890), error: None };
        s.finish_message(&m.id, MsgStatus::Done, &stats).unwrap();
        let got = s.get_message(&m.id).unwrap();
        assert_eq!(got.content, "Buy at ₹68,400 🚀");
        assert_eq!(got.status, MsgStatus::Done);
        assert_eq!(got.stats.completion_tokens, Some(6));
        assert_eq!(got.stats.ttft_ms, Some(640));
    }

    /// Review Focus 4: the app died while a reply was streaming.
    #[test]
    fn a_reply_left_streaming_is_interrupted_after_reopening() {
        let dir = std::env::temp_dir().join(format!("lorren-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lorren.db");
        let _ = std::fs::remove_file(&path);
        let (chat_id, msg_id) = {
            let s = Store::open(&path).unwrap();
            let chat = s.create_chat("m", None).unwrap();
            let m = s.append_message(&chat.id, None, Role::Assistant, "half a rep", MsgStatus::Streaming, None).unwrap();
            (chat.id, m.id)
        };
        let s = Store::open(&path).unwrap();
        let m = s.get_message(&msg_id).unwrap();
        assert_eq!(m.status, MsgStatus::Interrupted);
        assert_eq!(m.content, "half a rep");
        assert_eq!(s.list_chats().unwrap()[0].id, chat_id);
    }

    #[test]
    fn chats_are_listed_newest_first() {
        let s = Store::open_in_memory().unwrap();
        let a = s.create_chat("m", None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = s.create_chat("m", None).unwrap();
        let ids: Vec<_> = s.list_chats().unwrap().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![b.id, a.id]);
    }
}
