#![cfg(feature = "fake-engine")]
//! The fake engine is the ground every supervisor and chat test stands on, so it is tested first.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Command, Stdio};

fn exe() -> &'static str {
    env!("CARGO_BIN_EXE_fake-engine")
}

#[test]
fn it_prints_the_engine_banner_and_answers_health() {
    let mut child = Command::new(exe())
        .args(["serve", "M:/New Drive/x.gguf", "--host", "127.0.0.1", "--port", "0", "--ctx", "4096"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    assert!(line.starts_with("stoke serve: listening on 127.0.0.1:"), "{line}");
    assert!(line.trim_end().ends_with("ctx=4096"), "{line}");
    let port: u16 = line.split("127.0.0.1:").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert!(resp.contains("{\"status\":\"ok\"}"));
    child.kill().unwrap();
}

#[test]
fn it_refuses_like_the_engine_does() {
    let out = Command::new(exe())
        .args(["serve", "x.gguf", "--port", "0"])
        .env("FAKE_ENGINE_MODE", "refuse")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("general.architecture is 'qwen2vl'"));
    let big = Command::new(exe()).args(["serve", "x.gguf", "--port", "0", "--ctx", "8192"]).output().unwrap();
    assert!(String::from_utf8_lossy(&big.stderr).contains("max_ctx 8192 exceeds the model's 4096"));
}

/// A running fake; Drop kills AND waits for it, so no test leaves a process behind.
struct Fake {
    child: std::process::Child,
    port: u16,
}

impl Fake {
    fn start() -> Fake {
        let mut child = Command::new(exe())
            .args(["serve", "M:/New Drive/Llama-3.2-3B.gguf", "--host", "127.0.0.1", "--port", "0", "--ctx", "4096"])
            .env("FAKE_ENGINE_TOKEN_MS", "1")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        let port: u16 = line.split("127.0.0.1:").nth(1).unwrap().split_whitespace().next().unwrap().parse().unwrap();
        Fake { child, port }
    }

    /// POSTs a chat request and returns the `data:` payloads of the response, in order.
    fn stream(&self, body: &str) -> Vec<String> {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).unwrap();
        resp.lines().filter_map(|l| l.strip_prefix("data: ")).map(String::from).collect()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

const REQ: &str = r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":50,"stream":true"#;

#[test]
fn usage_chunk_only_when_asked_for() {
    let f = Fake::start();
    let plain = f.stream(&format!("{REQ}}}"));
    assert_eq!(plain.last().unwrap(), "[DONE]");
    assert!(plain.iter().all(|c| !c.contains("\"usage\"")), "{plain:?}");

    let asked = f.stream(&format!("{REQ},\"stream_options\":{{\"include_usage\":true}}}}"));
    let n = asked.len();
    assert_eq!(asked[n - 1], "[DONE]");
    assert_eq!(asked.iter().filter(|c| c.contains("\"usage\"")).count(), 1, "{asked:?}");
    assert!(asked[n - 2].contains("\"usage\"") && asked[n - 2].contains("\"choices\":[]"), "{asked:?}");
    assert!(asked[n - 3].contains("\"finish_reason\":\"stop\""), "{asked:?}");
}

#[test]
fn chunks_carry_the_real_envelope() {
    let f = Fake::start();
    // Lorren's real requests carry "model":"local"; the real engine echoes the REQUEST's model in
    // every chunk (src/server.c stream_envelope) -- it never reports the file it loaded.
    let chunks = f.stream(&format!("{REQ},\"model\":\"local\",\"stream_options\":{{\"include_usage\":true}}}}"));
    let json: Vec<serde_json::Value> =
        chunks.iter().filter(|c| *c != "[DONE]").map(|c| serde_json::from_str(c).unwrap()).collect();
    assert!(json.len() >= 3);
    assert_eq!(json[0]["choices"][0]["delta"]["role"], "assistant");
    for (i, c) in json.iter().enumerate() {
        assert_eq!(c["id"], json[0]["id"]);
        assert!(c["id"].as_str().unwrap().starts_with("chatcmpl-"));
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["model"], "local", "the fake must echo the request's model, not name its file: {c}");
        assert_eq!(c["created"], json[0]["created"]);
        assert!(c["created"].as_u64().unwrap() > 1_600_000_000);
        if i > 0 {
            assert!(c["choices"].as_array().unwrap().is_empty() || c["choices"][0]["delta"].get("role").is_none());
        }
        if let Some(ch) = c["choices"].get(0) {
            assert!(ch.get("logprobs").is_some() && ch["logprobs"].is_null(), "{c}");
        }
    }
    // A second request gets a different id.
    let again = f.stream(&format!("{REQ}}}"));
    let second: serde_json::Value = serde_json::from_str(&again[0]).unwrap();
    assert_ne!(second["id"], json[0]["id"]);
    // No usable "model" in the request (absent, or empty -- src/server.c:889-895 reads both as
    // unset): every chunk says "stoke", the real engine's fallback. A fake that fell back to the
    // model file's name would let a client test pass that the real engine fails.
    for body in [
        format!("{REQ},\"stream_options\":{{\"include_usage\":true}}}}"),
        format!("{REQ},\"model\":\"\",\"stream_options\":{{\"include_usage\":true}}}}"),
    ] {
        let chunks = f.stream(&body);
        let json: Vec<serde_json::Value> =
            chunks.iter().filter(|c| *c != "[DONE]").map(|c| serde_json::from_str(c).unwrap()).collect();
        assert!(json.len() >= 3, "{body} -> {chunks:?}");
        for c in &json {
            assert_eq!(c["model"], "stoke", "{body} -> {c}");
        }
    }
}

#[test]
fn an_unknown_mode_exits_2() {
    for mode in ["bogus", "Hang", "crash_after_ms:abc"] {
        let out = Command::new(exe()).args(["serve", "x.gguf", "--port", "0"]).env("FAKE_ENGINE_MODE", mode).output().unwrap();
        assert_eq!(out.status.code(), Some(2), "{mode}");
        assert!(String::from_utf8_lossy(&out.stderr).contains(mode), "{mode}");
    }
}
