//! A test double of `stoke serve` (feature "fake-engine"; never shipped). See Task 7 of the
//! milestone-1 plan for the switches. Formats follow the real server (src/server.c):
//! `data: {json}\n\n` chunks, `: ping\n\n` comments, `data: [DONE]\n\n`, and errors shaped
//! {"error":{"message":..,"type":..,"param":null,"code":null}}.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn event(line: &str) {
    if let Some(p) = env("FAKE_ENGINE_EVENTS") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("write-fixtures") {
        write_fixtures(args.get(2).expect("write-fixtures <dir>"));
        return;
    }
    if args.get(1).map(String::as_str) != Some("serve") {
        eprintln!("fatal: usage: stoke serve <path.gguf> [--host A] [--port P] [--ctx N] [--seqs N] [--kv q8]");
        std::process::exit(2);
    }
    event(&format!("argv {}", serde_json::to_string(&args).unwrap()));
    let model = args.get(2).cloned().unwrap_or_default();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let host = flag("--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let ctx: u32 = flag("--ctx").and_then(|p| p.parse().ok()).unwrap_or(4096);
    if ctx > 4096 {
        eprintln!("stoke serve: stoke_open: max_ctx {ctx} exceeds the model's 4096");
        std::process::exit(1);
    }
    let mode = env("FAKE_ENGINE_MODE").unwrap_or_else(|| "ok".into());
    if mode == "refuse" {
        // The real engine's sentence (src/model.c:755-757, behind the `stoke serve: ` of src/server.c:1752).
        eprintln!(
            "stoke serve: model: '{model}' -- general.architecture is 'qwen2vl', expected 'qwen3' or 'llama' (this loader implements those two)"
        );
        std::process::exit(1);
    }
    if let Some(ms) = env("FAKE_ENGINE_LOAD_MS").and_then(|v| v.parse().ok()) {
        std::thread::sleep(Duration::from_millis(ms));
    }
    let listener = TcpListener::bind((host.as_str(), port)).expect("bind");
    let actual = listener.local_addr().unwrap().port();
    println!("stoke serve: listening on {host}:{actual}  ctx={ctx}");
    std::io::stdout().flush().unwrap();
    eprintln!("fake engine: serving {model}");
    if let Some(ms) = mode.strip_prefix("crash_after_ms:").and_then(|v| v.parse::<u64>().ok()) {
        let armed = match env("FAKE_ENGINE_CRASH_ONCE") {
            Some(marker) if std::path::Path::new(&marker).exists() => false,
            Some(marker) => {
                let _ = std::fs::write(marker, "crashed");
                true
            }
            None => true,
        };
        if armed {
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                eprintln!("fake engine: crashing on purpose");
                std::process::exit(3);
            });
        }
    }
    let hang = mode == "hang";
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        if hang {
            std::mem::forget(stream); // accept, then never answer
            continue;
        }
        let model = model.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &model, ctx);
        });
    }
}

fn handle(mut s: TcpStream, model: &str, ctx: u32) -> std::io::Result<()> {
    let mut r = BufReader::new(s.try_clone()?);
    let mut first = String::new();
    r.read_line(&mut first)?;
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        r.read_line(&mut h)?;
        let h = h.trim_end().to_ascii_lowercase();
        if h.is_empty() {
            break;
        }
        if let Some(v) = h.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => respond(&mut s, 200, "{\"status\":\"ok\"}\n"),
        ("GET", "/v1/models") => {
            let id = std::path::Path::new(model).file_stem().map(|x| x.to_string_lossy().to_string()).unwrap_or_default();
            let body = serde_json::json!({"object":"list","data":[{"id":id,"object":"model","created":0,"owned_by":"stoke"}]});
            respond(&mut s, 200, &format!("{body}\n"))
        }
        ("POST", "/v1/chat/completions") => chat(&mut s, &body, ctx),
        _ => respond(&mut s, 404, &error_body("not found", "invalid_request_error")),
    }
}

fn error_body(message: &str, kind: &str) -> String {
    format!("{}\n", serde_json::json!({"error":{"message":message,"type":kind,"param":null,"code":null}}))
}

fn respond(s: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    write!(
        s,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    s.flush()
}

fn chat(s: &mut TcpStream, body: &[u8], ctx: u32) -> std::io::Result<()> {
    event(&format!("request {}", String::from_utf8_lossy(body)));
    let v: serde_json::Value = serde_json::from_slice(body).unwrap_or_default();
    let chars: usize = v["messages"]
        .as_array()
        .map(|m| m.iter().map(|x| x["content"].as_str().unwrap_or("").chars().count()).sum())
        .unwrap_or(0);
    let prompt_tokens = (chars / 4).max(1) as u64;
    let max_tokens = v["max_tokens"].as_u64().unwrap_or(128);
    if prompt_tokens + max_tokens > ctx as u64 {
        event("context_length_exceeded");
        let msg = format!(
            "prompt is {prompt_tokens} tokens and max_tokens is {max_tokens}, which exceeds the context length of {ctx}"
        );
        return respond(s, 400, &error_body(&msg, "context_length_exceeded"));
    }
    let reply = env("FAKE_ENGINE_REPLY").unwrap_or_else(|| "Hello from the fake engine.".into());
    let token_ms: u64 = env("FAKE_ENGINE_TOKEN_MS").and_then(|x| x.parse().ok()).unwrap_or(20);
    write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    s.write_all(b": ping\n\n")?;
    let words: Vec<&str> = reply.split_inclusive(' ').collect();
    let mut n = 0u64;
    for w in &words {
        if n >= max_tokens {
            break;
        }
        let chunk = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk",
            "choices":[{"index":0,"delta":{"content":w},"finish_reason":null}]});
        if s.write_all(format!("data: {chunk}\n\n").as_bytes()).and_then(|_| s.flush()).is_err() {
            event("cancelled");
            return Ok(());
        }
        n += 1;
        std::thread::sleep(Duration::from_millis(token_ms));
    }
    let finish = if (n as usize) < words.len() { "length" } else { "stop" };
    let done = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk",
        "choices":[{"index":0,"delta":{},"finish_reason":finish}]});
    let usage = serde_json::json!({"id":"chatcmpl-fake","object":"chat.completion.chunk","choices":[],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":n,"total_tokens":prompt_tokens + n}});
    if write!(s, "data: {done}\n\ndata: {usage}\n\ndata: [DONE]\n\n").and_then(|_| s.flush()).is_err() {
        event("cancelled");
        return Ok(());
    }
    event("done");
    Ok(())
}

fn write_fixtures(dir: &str) {
    use lorren_lib::gguf::fixture::{write, Spec};
    let d = std::path::Path::new(dir);
    std::fs::create_dir_all(d).unwrap();
    write(&d.join("nexus.gguf"), &Spec::llama("Nexus_Model")).unwrap();
    std::fs::write(
        d.join("Modelfile"),
        "FROM C:/Users/Kabir/ai_models/nexus.gguf\nSYSTEM \"You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts.\"\n",
    )
    .unwrap();
    write(&d.join("Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &Spec::qwen3("Qwen3-4B")).unwrap();
    write(&d.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), &Spec::llama("Llama-3.2-3B")).unwrap();
    write(&d.join("qwen2.5vl-3b.gguf"), &Spec { arch: "qwen2vl", ..Spec::qwen3("qwen2.5vl") }).unwrap();
    println!("fixtures written to {}", d.display());
}
