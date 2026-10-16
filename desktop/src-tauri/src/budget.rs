//! Which part of a chat to send, and how long a reply may be (spec §3.2).
//! The app has no tokenizer: it estimates conservatively, and the engine's 400 reply corrects it.

/// Earlier turns sent with each message, in estimated tokens (decided 2026-09-25).
pub const HISTORY_BUDGET: u32 = 1_500;
pub const REPLY_CAP: u32 = 1_024;
pub const MIN_REPLY: u32 = 64;
/// Template tokens around each message (role headers, end markers).
const PER_MESSAGE: u32 = 6;
const MARGIN: u32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// "system", "user" or "assistant"
    pub role: &'static str,
    pub content: String,
}

/// About 3.5 characters per token, rounded up, plus the template's per-message cost.
pub fn estimate_tokens(text: &str) -> u32 {
    let chars = text.chars().count() as u32;
    (chars * 2 + 6) / 7 + PER_MESSAGE
}

/// Keeps the most recent earlier turns that fit in `budget`, whole messages only, and never
/// starts on an assistant turn. Returns the kept turns and how many were dropped.
pub fn select_history(history: &[Turn], budget: u32) -> (Vec<Turn>, u32) {
    let mut used = 0u32;
    let mut start = history.len();
    for (i, t) in history.iter().enumerate().rev() {
        let cost = estimate_tokens(&t.content);
        if used + cost > budget {
            break;
        }
        used += cost;
        start = i;
    }
    while start < history.len() && history[start].role != "user" {
        start += 1;
    }
    (history[start..].to_vec(), start as u32)
}

/// The reply length to ask for: at most `cap`, and what the context leaves after the prompt.
/// `None` when the prompt leaves less than MIN_REPLY tokens.
pub fn max_tokens_for(ctx: u32, prompt_tokens: u32, cap: u32) -> Option<u32> {
    let room = ctx.checked_sub(prompt_tokens + MARGIN)?;
    if room < MIN_REPLY { None } else { Some(room.min(cap)) }
}

/// Spec §5: a message that cannot fit even with no history is refused before sending.
pub fn message_too_long(system: Option<&str>, message: &str, ctx: u32) -> Option<u32> {
    let est = system.map(estimate_tokens).unwrap_or(0) + estimate_tokens(message);
    max_tokens_for(ctx, est, REPLY_CAP).is_none().then_some(est)
}

/// Reads the engine's 400 `context_length_exceeded` message:
/// "prompt is N tokens and max_tokens is M, which exceeds the context length of C".
pub fn parse_context_error(msg: &str) -> Option<(u32, u32, u32)> {
    let after = |key: &str| -> Option<u32> {
        let i = msg.find(key)? + key.len();
        msg[i..].trim_start().split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    Some((after("prompt is")?, after("max_tokens is")?, after("context length of")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &'static str, n: usize) -> Turn {
        Turn { role, content: "x".repeat(n) }
    }

    #[test]
    fn the_most_recent_turns_that_fit_are_kept_whole() {
        let h = vec![turn("user", 700), turn("assistant", 700), turn("user", 700), turn("assistant", 700)];
        let per = estimate_tokens(&h[0].content);
        let (kept, dropped) = select_history(&h, per * 2);
        assert_eq!(kept.len(), 2);
        assert_eq!(dropped, 2);
        assert_eq!(kept[0].role, "user");
    }

    #[test]
    fn history_never_starts_on_an_assistant_turn() {
        let h = vec![turn("user", 700), turn("assistant", 70), turn("user", 70), turn("assistant", 70)];
        let (kept, _) = select_history(&h, estimate_tokens(&"x".repeat(70)) * 3);
        assert_eq!(kept.first().unwrap().role, "user");
    }

    #[test]
    fn max_tokens_is_capped_and_leaves_room_for_the_prompt() {
        assert_eq!(max_tokens_for(4096, 500, REPLY_CAP), Some(1024));
        assert_eq!(max_tokens_for(4096, 3500, REPLY_CAP), Some(4096 - 3500 - 8));
        assert_eq!(max_tokens_for(4096, 4060, REPLY_CAP), None);
    }

    /// Review Focus 5: a pasted document longer than the whole context.
    #[test]
    fn a_message_longer_than_the_context_is_caught_before_sending() {
        let doc = "word ".repeat(4_000);
        let est = message_too_long(None, &doc, 4096).expect("refused");
        assert!(est > 4096);
        assert_eq!(message_too_long(Some("be brief"), "hello", 4096), None);
    }

    #[test]
    fn the_engines_context_message_is_read() {
        let msg = "prompt is 4210 tokens and max_tokens is 1024, which exceeds the context length of 4096";
        assert_eq!(parse_context_error(msg), Some((4210, 1024, 4096)));
        assert_eq!(parse_context_error("something else"), None);
    }
}
