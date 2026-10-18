//! A test double of `stoke serve` (feature "fake-engine"; never shipped). See Task 7 of the
//! milestone-1 plan for the switches. Formats follow the real server (src/server.c):
//! `data: {json}\n\n` chunks, `: ping\n\n` comments, `data: [DONE]\n\n`, and errors shaped
//! {"error":{"message":..,"type":..,"param":null,"code":null}}.
//! `FAKE_ENGINE_REFUSE_MODEL=<text>` (Task 10): refuse exactly as `FAKE_ENGINE_MODE=refuse` does, but only a model whose path contains `<text>`.
//! `FAKE_ENGINE_MODE=silent_exit` (R47): an engine killed, or crashing, during its load: it waits out `FAKE_ENGINE_LOAD_MS`, then exits 1
//! with no banner and not a line on either stream.
//! `FAKE_ENGINE_LOAD_MS` and `FAKE_ENGINE_TOKEN_MS` take a whole number of milliseconds; anything else is refused at startup (exit 2).
//!
//! It is the real engine's shape, not the real engine. KNOWN DIFFERENCES, which a green test
//! against the fake does not cover:
//!
//! - the fake closes each connection after a response (`Connection: close`); the real engine
//!   streams chunked over keep-alive and closes idle keep-alive connections after 60 s;
//! - the fake answers `/health` on its own thread at once; the real engine answers only between
//!   scheduler steps;
//! - the fake validates no request fields; the real engine answers 400 for out-of-range values;
//! - the fake decodes JSON string escapes; the real engine templates the raw string (see the
//!   Global Constraint on message content);
//! - the fake does not honour `"stream":false`: every chat request is answered as a stream,
//!   because the app always streams (the real engine then answers one JSON body);
//! - the fake sends one `: ping` at stream start; the real engine pings every 15 s of silence
//!   (`SSE_PING_MS_DEFAULT`, src/server.c);
//! - the fake answers 404 "not found" for an unknown route, and 404 (never 405) for the wrong
//!   method on a known one; the real engine says "unknown endpoint" for the first and answers 405
//!   for the second (it also serves `/v1/completions`, which the fake does not);
//! - the fake checks `--ctx` before the architecture and hardcodes `max_ctx` as 4096; the real
//!   engine opens the model first and uses min(model ctx, 4096).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

/// A millisecond switch: unset (or empty) is `None`; a value that is not a whole number is an error that names the
/// variable and the value.
fn env_ms(name: &str) -> Result<Option<u64>, String> {
    match env(name) {
        None => Ok(None),
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|_| format!("fake engine: {name} '{v}' is not a whole number of milliseconds")),
    }
}

fn event(line: &str) {
    if let Some(p) = env("FAKE_ENGINE_EVENTS") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// What `FAKE_ENGINE_MODE` asks for. Anything else is refused at startup: a mistyped mode that
/// silently behaved as `ok` would turn a hang or crash test into a green run of the wrong thing.
enum Mode {
    /// `ok`, the default: serve normally.
    Serve,
    Refuse,
    Hang,
    CrashAfterMs(u64),
    /// Exits 1 during the load, before the banner, having printed nothing.
    SilentExit,
}

fn parse_mode(raw: &str) -> Result<Mode, String> {
    match raw {
        "ok" => Ok(Mode::Serve),
        "refuse" => Ok(Mode::Refuse),
        "hang" => Ok(Mode::Hang),
        "silent_exit" => Ok(Mode::SilentExit),
        _ => match raw.strip_prefix("crash_after_ms:").map(str::parse::<u64>) {
            Some(Ok(ms)) => Ok(Mode::CrashAfterMs(ms)),
            _ => Err(format!(
                "fake engine: unknown FAKE_ENGINE_MODE '{raw}' (expected ok | refuse | hang | silent_exit | crash_after_ms:<N>)"
            )),
        },
    }
}

/// The N in `chatcmpl-N`: one counter for the process, from 1, taken when a request is accepted
/// (src/server.c `next_req_id`, `start_generation`). A request refused with a 400 takes none.
static NEXT_REQ_ID: AtomicU64 = AtomicU64::new(1);

/// `s` as a quoted, escaped JSON string (the real server's `buf_json_str`).
fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
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
    // A bad mode is a usage error too: refused before anything is recorded, bound or printed.
    let mode = match parse_mode(&env("FAKE_ENGINE_MODE").unwrap_or_else(|| "ok".into())) {
        Ok(mode) => mode,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };
    // The two millisecond switches are validated here as well: a value that is not a whole number is a usage error, not
    // a switch that quietly means the default (`FAKE_ENGINE_TOKEN_MS=10ms` would run at 20 ms a token).
    let (load_ms, token_ms) = match (env_ms("FAKE_ENGINE_LOAD_MS"), env_ms("FAKE_ENGINE_TOKEN_MS")) {
        (Ok(load), Ok(token)) => (load, token.unwrap_or(20)),
        (Err(msg), _) | (_, Err(msg)) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };
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
    // FAKE_ENGINE_REFUSE_MODEL=<text>: the refusal below, for the model whose path contains <text> only. A model
    // that does not contain it gets whatever FAKE_ENGINE_MODE says, as usual. (`env` reads an empty value as unset.)
    let refuses_this_model = env("FAKE_ENGINE_REFUSE_MODEL").is_some_and(|text| model.contains(&text));
    if matches!(mode, Mode::Refuse) || refuses_this_model {
        // The real engine's sentence (src/model.c:755-757, behind the `stoke serve: ` of src/server.c:1752).
        eprintln!(
            "stoke serve: model: '{model}' -- general.architecture is 'qwen2vl', expected 'qwen3' or 'llama' (this loader implements those two)"
        );
        std::process::exit(1);
    }
    if let Some(ms) = load_ms {
        std::thread::sleep(Duration::from_millis(ms));
    }
    if matches!(mode, Mode::SilentExit) {
        // Killed, or crashed, while loading (R47): no banner, and no line for the supervisor to take as the reason.
        std::process::exit(1);
    }
    let listener = TcpListener::bind((host.as_str(), port)).expect("bind");
    let actual = listener.local_addr().unwrap().port();
    // The real engine writes nothing to stderr once it is serving, so the fake's own "serving" line goes to STDOUT, after
    // the banner. On stderr the supervisor's stderr reader logged it as an `error` whenever it won the race with the
    // stdout reader that sets "the banner has been seen". Both lines go in one write, and a reader that has already gone
    // away is no error: the tests that wait for the banner read one line and let the pipe go.
    {
        let mut out = std::io::stdout().lock();
        let _ = write!(out, "stoke serve: listening on {host}:{actual}  ctx={ctx}\nfake engine: serving {model}\n");
        let _ = out.flush();
    }
    if let Mode::CrashAfterMs(ms) = mode {
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
    let hang = matches!(mode, Mode::Hang);
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        if hang {
            std::mem::forget(stream); // accept, then never answer
            continue;
        }
        let model = model.clone();
        std::thread::spawn(move || {
            let _ = handle(stream, &model, ctx, token_ms);
        });
    }
}

fn handle(mut s: TcpStream, model: &str, ctx: u32, token_ms: u64) -> std::io::Result<()> {
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
        ("POST", "/v1/chat/completions") => chat(&mut s, &body, ctx, token_ms),
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

fn chat(s: &mut TcpStream, body: &[u8], ctx: u32, token_ms: u64) -> std::io::Result<()> {
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
    // Accepted: from here the request has its id and its one timestamp (src/server.c
    // start_generation), and every chunk of the answer shares them.
    let id = NEXT_REQ_ID.fetch_add(1, Ordering::Relaxed);
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // The usage chunk is sent only when the request asks for it (src/server.c parse_body, :882-887).
    let want_usage = v["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    // The model every chunk reports is the REQUEST's own "model" -- "stoke" when it sent none, or an
    // empty one (src/server.c parse_body, :889-895) -- never the name of the file that was loaded.
    let model = v["model"].as_str().filter(|m| !m.is_empty()).unwrap_or("stoke");
    // The envelope every chunk of one answer shares, in the real server's order (src/server.c
    // stream_envelope, :470-482); each frame then adds its own "choices" (and "usage").
    let envelope = format!(
        r#"{{"id":"chatcmpl-{id}","object":"chat.completion.chunk","created":{created},"model":{},"#,
        json_str(model)
    );
    let frame = |rest: &str| format!("data: {envelope}{rest}}}\n\n");
    let reply = env("FAKE_ENGINE_REPLY").unwrap_or_else(|| "Hello from the fake engine.".into());
    write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    s.write_all(b": ping\n\n")?;
    let words: Vec<&str> = reply.split_inclusive(' ').collect();
    let mut n = 0u64;
    for w in &words {
        if n >= max_tokens {
            break;
        }
        // The role rides on the FIRST content delta and on no other (src/server.c stream_delta).
        let role = if n == 0 { r#""role":"assistant","# } else { "" };
        let chunk = frame(&format!(
            r#""choices":[{{"index":0,"delta":{{{role}"content":{}}},"logprobs":null,"finish_reason":null}}]"#,
            json_str(w)
        ));
        if s.write_all(chunk.as_bytes()).and_then(|_| s.flush()).is_err() {
            event("cancelled");
            return Ok(());
        }
        n += 1;
        std::thread::sleep(Duration::from_millis(token_ms));
    }
    let finish = if (n as usize) < words.len() { "length" } else { "stop" };
    // The finish chunk (an empty delta carrying the reason), then -- only when asked for -- the usage
    // chunk with an EMPTY "choices", then the sentinel (src/server.c stream_final, :649-693).
    let mut tail = frame(&format!(
        r#""choices":[{{"index":0,"delta":{{}},"logprobs":null,"finish_reason":"{finish}"}}]"#
    ));
    if want_usage {
        tail += &frame(&format!(
            r#""choices":[],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{n},"total_tokens":{}}}"#,
            prompt_tokens + n
        ));
    }
    tail += "data: [DONE]\n\n";
    if s.write_all(tail.as_bytes()).and_then(|_| s.flush()).is_err() {
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
