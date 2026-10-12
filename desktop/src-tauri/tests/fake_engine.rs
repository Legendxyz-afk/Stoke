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
