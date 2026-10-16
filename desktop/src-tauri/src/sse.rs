//! Splits an SSE byte stream into events (spec §3.2). Bytes are buffered until a whole line
//! has arrived, so a UTF-8 character split across network chunks is never decoded in halves
//! (a 0x0A byte never occurs inside a multi-byte UTF-8 sequence).

use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub enum SseEvent {
    Data(String),
    Comment,
    Done,
}

#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=nl).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            if line[0] == b':' {
                out.push(SseEvent::Comment);
            } else if let Some(rest) = line.strip_prefix(b"data:") {
                let text = String::from_utf8_lossy(rest).trim_start().to_string();
                out.push(if text == "[DONE]" { SseEvent::Done } else { SseEvent::Data(text) });
            }
        }
        out
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review Focus 2: a character split across network chunks must survive.
    #[test]
    fn multibyte_characters_survive_every_split_point() {
        let line = "data: {\"content\":\"Buy at ₹68,400 🚀\"}\n\n".as_bytes();
        for cut in 1..line.len() {
            let mut p = SseParser::new();
            let mut events = p.push(&line[..cut]);
            events.extend(p.push(&line[cut..]));
            assert_eq!(events, vec![SseEvent::Data("{\"content\":\"Buy at ₹68,400 🚀\"}".into())], "cut at byte {cut}");
        }
    }

    #[test]
    fn pings_done_crlf_and_several_events_in_one_chunk() {
        let mut p = SseParser::new();
        let ev = p.push(b": ping\n\ndata: {\"a\":1}\r\n\r\ndata: [DONE]\n\n");
        assert_eq!(ev, vec![SseEvent::Comment, SseEvent::Data("{\"a\":1}".into()), SseEvent::Done]);
    }

    #[test]
    fn a_partial_line_waits_for_the_rest() {
        let mut p = SseParser::new();
        assert!(p.push(b"data: {\"a\"").is_empty());
        assert_eq!(p.push(b":2}\n"), vec![SseEvent::Data("{\"a\":2}".into())]);
    }

    #[test]
    fn engine_chunks_parse() {
        let c: ChatChunk = serde_json::from_str(
            r#"{"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(c.choices[0].delta.content.as_deref(), Some("Hi"));
        let u: ChatChunk = serde_json::from_str(
            r#"{"choices":[],"usage":{"prompt_tokens":69,"completion_tokens":6,"total_tokens":75}}"#,
        )
        .unwrap();
        assert_eq!(u.usage.unwrap().completion_tokens, 6);
    }
}
