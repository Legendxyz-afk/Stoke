import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { api } from "../ipc/api";
import type { Chat } from "../ipc/bindings/Chat";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import type { StreamEvent } from "../ipc/bindings/StreamEvent";
import "./chat.css";
import { ChatList } from "./ChatList";
import { Composer } from "./Composer";
import { MessageRow } from "./MessageView";
import { ReplyPanel } from "./ReplyPanel";
import { applyEvent, begin, idle, statsOf, type Live } from "./stream";

/** At most ~30 redraws a second while tokens stream, so the UI does not compete with the engine (spec §3.8). */
const FLUSH_MS = 33;

/** The reader counts as being at the end of the transcript within this many pixels of it (R56). */
const AT_BOTTOM_PX = 24;

/** The index of the message after which the history cut falls: the `dropped`-th message that counts as a
 *  turn. The host builds the history it sends from the same list (chat.rs `turns()`): an error reply and an
 *  empty message are not turns, so they are not counted here either. -1 when nothing was dropped. */
function lastDroppedIndex(messages: ChatView["messages"], dropped: number): number {
  let turns = 0;
  for (let i = 0; i < messages.length && dropped > 0; i++) {
    const m = messages[i].message;
    if (m.status !== "error" && m.content !== "") turns += 1;
    if (turns === dropped) return i;
  }
  return -1;
}

/** The model a chat made now would get (demo D1): the one picked in the panel if it can still run, else the default if it can
 *  run, else the first that can. The host's `pick_model(None)` takes the same road (R50), so the picker never promises a
 *  model the host would not give. */
function modelForNewChat(models: ModelInfo[], picked: string | null): string | null {
  const runnable = models.filter((m) => m.runnable);
  return (runnable.find((m) => m.id === picked) ?? runnable.find((m) => m.is_default) ?? runnable[0])?.id ?? null;
}

export function ChatTab({ overview, active: shown = true }: {
  overview: Overview | null;
  /** The Chat tab is showing. App keeps this component mounted while another tab shows (R36a), so each time this
   *  turns true it asks the host again for what can change in the meantime (R60). */
  active?: boolean;
}) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [view, setView] = useState<ChatView | null>(null);
  const [live, setLive] = useState<Live>(idle);
  const [notice, setNotice] = useState<string | null>(null);
  /** The composer's text lives here, not in Composer, so a message the host refuses can be put back (R36b). */
  const [draft, setDraft] = useState("");
  /** With no chat open, the model picked in the panel for the chat that Send will make (demo D1). Null until one is picked. */
  const [picked, setPicked] = useState<string | null>(null);
  /** A message typed with no chat open, waiting for the chat made for it to be open: opening a chat resets the live reply, so
   *  the send may only start after that has happened. The effect that opens a chat consumes it. */
  const sendOnOpen = useRef<{ chatId: string; text: string } | null>(null);
  /** The id of the chat that is open (the state above, readable from callbacks that outlive a render). */
  const active = useRef<string | null>(null);
  const pending = useRef("");
  const timer = useRef<number | null>(null);
  const transcript = useRef<HTMLDivElement>(null);
  /** Whether the reader was at the end of the transcript at the last scroll event. It has to be known from before
   *  new text makes the transcript longer: measured afterwards, a reader who is at the end looks like one who is not (R56). */
  const atBottom = useRef(true);
  /** Reload calls made, and the newest one whose answer is on screen: an older answer that arrives late is dropped (R59). */
  const requested = useRef(0);
  const applied = useRef(0);
  /** The chat list has been loaded once. Until then each showing of the tab tries again (R60). */
  const chatsLoaded = useRef(false);
  /** Which host call the notice is about, so that the same call working later takes the notice away again. */
  const noticeFrom = useRef<string | null>(null);

  /** A host call failed (R57): log it once, the way App does for overview(), and show it in the notice, unless the
   *  failure is about a chat that is no longer the open one. */
  const fail = useCallback((what: string, e: unknown, chatId?: string) => {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`${what} failed:`, message);
    if (chatId === undefined || active.current === chatId) {
      noticeFrom.current = what;
      setNotice(message);
    }
  }, []);

  /** Take the notice away: whatever it says, or only if it is about `what` (that call has worked now). */
  const clearNotice = useCallback((what?: string) => {
    if (what !== undefined && noticeFrom.current !== what) return;
    noticeFrom.current = null;
    setNotice(null);
  }, []);

  /** Never rejects: a failure is reported by `fail`. Null when there is no list. */
  const refreshChats = useCallback(async (): Promise<Chat[] | null> => {
    try {
      const cs = await api.chats();
      chatsLoaded.current = true;
      setChats(cs);
      clearNotice("chats");
      return cs;
    } catch (e) {
      fail("chats", e);
      return null;
    }
  }, [fail, clearNotice]);

  /** Never rejects: a failure is reported by `fail`. An answer is dropped when another chat is open by now, or when
   *  a later call's answer is already on screen: the one that landed last must not be an older view (R59). */
  const reload = useCallback(async (chatId: string) => {
    const mine = ++requested.current;
    try {
      const v = await api.chat(chatId);
      if (active.current !== chatId || mine < applied.current) return;
      applied.current = mine;
      setView(v);
      clearNotice("chat");
    } catch (e) {
      fail("chat", e, chatId);
    }
  }, [fail, clearNotice]);

  useEffect(() => {
    if (!shown) return;
    // Each time the tab is shown (and when it first mounts): the model list again, since the host can mark a model as
    // one that cannot run; the chat list only if it never loaded, since what changes it is done from here (R60).
    api.models().then(
      (ms) => {
        setModels(ms);
        clearNotice("models");
      },
      (e) => fail("models", e),
    );
    if (!chatsLoaded.current) {
      void refreshChats().then((cs) => {
        if (cs && cs.length > 0) setActiveId((id) => id ?? cs[0].id);
      });
    }
  }, [shown, refreshChats, fail, clearNotice]);

  useEffect(() => {
    const previous = active.current;
    active.current = activeId;
    atBottom.current = true; // a chat is opened following its end
    setLive(idle);
    // A notice belongs to the chat it came from. The first open, from no chat, follows none: the failure of a call that
    // is not about a chat (the model list, say) must not be wiped by the chat list's arrival.
    if (previous !== null) clearNotice();
    if (activeId) void reload(activeId);
    else setView(null);
    // A message typed before this chat existed (demo D1) goes now, through the same path as any other: after the reset
    // above, so that nothing of the reply that starts is wiped by it, and with the refusal handling of every send.
    const first = sendOnOpen.current;
    if (first && first.chatId === activeId) {
      sendOnOpen.current = null;
      void run("send", first.chatId, (cb) => api.send(first.chatId, first.text, cb), first.text);
    }
  }, [activeId, reload, clearNotice]);

  // R56: while a reply streams, the transcript follows the new text, but only if the reader is at its end. A reader
  // who scrolled up to read is not pulled down. Hidden (display: none) the transcript has no layout, so it is left
  // alone; when it is shown again it follows to the end, if that is where the reader was.
  // R61: it follows a change of the chat view as well, under the same rule. A chat is opened with `atBottom` true, so it
  // opens at its end; and the reload that ends a reply (the stored reply replaces the live text, and may be longer)
  // re-follows a reader who stayed at the end.
  useLayoutEffect(() => {
    const el = transcript.current;
    if (shown && el && atBottom.current) el.scrollTop = el.scrollHeight;
  }, [live.text, shown, view]);

  const flush = () => {
    timer.current = null;
    const text = pending.current;
    pending.current = "";
    if (text) setLive((s) => applyEvent(s, { kind: "token", text }));
  };

  const onEvent = (chatId: string) => (e: StreamEvent) => {
    if (active.current !== chatId) return; // the reply still lands in the store; reopening the chat shows it
    if (e.kind === "token") {
      pending.current += e.text;
      if (timer.current == null) timer.current = window.setTimeout(flush, FLUSH_MS);
      return;
    }
    if (timer.current != null) window.clearTimeout(timer.current);
    flush();
    setLive((s) => applyEvent(s, e));
    void reload(chatId);
    if (e.kind !== "started") void refreshChats();
  };

  /** `restore` is the text of a send. When the host refuses it (`call` rejects: busy, too long, a model that cannot
   *  run) nothing was stored, so the text goes back into the composer, unless something new was typed meanwhile. */
  const run = async (
    what: "send" | "regenerate",
    chatId: string,
    call: (cb: (e: StreamEvent) => void) => Promise<void>,
    restore?: string,
  ) => {
    clearNotice();
    setLive(begin());
    try {
      await call(onEvent(chatId));
      // The host always ends a reply with an event; never leave the composer stuck on STOP if one is lost.
      if (active.current === chatId) setLive((s) => (s.busy ? { ...s, busy: false } : s));
    } catch (e) {
      if (restore !== undefined) setDraft((d) => (d === "" ? restore : d));
      if (active.current === chatId) setLive(idle);
      fail(what, e, chatId);
    }
  };

  const send = (text: string) => {
    const id = activeId;
    if (!id) {
      void startChat(text);
      return;
    }
    setDraft("");
    void run("send", id, (cb) => api.send(id, text, cb), text);
  };
  /** Demo D1: with no chat open, a message makes the chat. The chat is made on the model the panel shows, the list is
   *  refreshed, the chat is opened, and the text is sent in it by the effect that opens it (`sendOnOpen`). The composer
   *  empties and counts as busy from the moment of Send, as for any send, so a second Enter makes no second chat; if the chat
   *  cannot be made the text goes back (R36b) and the notice says why. */
  const startChat = async (text: string) => {
    clearNotice();
    setDraft("");
    setLive(begin());
    try {
      const c = await api.newChat(modelForNewChat(models, picked));
      await refreshChats();
      sendOnOpen.current = { chatId: c.id, text };
      setActiveId(c.id);
    } catch (e) {
      setDraft((d) => (d === "" ? text : d));
      setLive(idle);
      fail("newChat", e);
    }
  };
  const regenerate = () => {
    const id = activeId;
    if (id) void run("regenerate", id, (cb) => api.regenerate(id, cb));
  };
  const stop = () => {
    const id = activeId;
    if (id) api.stop(id).catch((e) => fail("stop", e, id));
  };
  const newChat = async () => {
    try {
      const c = await api.newChat(null);
      await refreshChats();
      setActiveId(c.id);
    } catch (e) {
      fail("newChat", e);
    }
  };
  const setModel = async (modelId: string) => {
    const id = activeId;
    if (!id) {
      setPicked(modelId); // no chat yet: this is the model Send will make one on, and no host call
      return;
    }
    try {
      await api.setChatModel(id, modelId);
      await reload(id);
      await refreshChats();
    } catch (e) {
      fail("setChatModel", e, id);
    }
  };
  const selectVersion = async (messageId: string) => {
    const id = activeId;
    if (!id) return;
    try {
      await api.selectVersion(id, messageId);
      await reload(id);
    } catch (e) {
      fail("selectVersion", e, id);
    }
  };

  const messages = view?.messages ?? [];
  const chat = view?.chat ?? null;
  const lastAssistant = [...messages].reverse().find((m) => m.message.role === "assistant");
  const busy = live.busy || messages.some((m) => m.message.status === "streaming");
  const dropped = live.dropped;
  const cutAfter = lastDroppedIndex(messages, dropped);
  const picker = chat?.model_id ?? modelForNewChat(models, picked) ?? "";
  // Demo D2: the reply being written has no token yet. Its row shows `<phase> · <N> s` in place of an empty text; while the
  // engine is not serving this chat's model (starting, restarting, or on another one) the phase is `loading <model>`.
  const waiting = live.busy && live.text === "" && live.messageId !== null && live.sentAt !== null ? { id: live.messageId, since: live.sentAt } : null;
  const engine = overview?.engine;
  const loading =
    chat && (engine?.state === "starting" || engine?.state === "restarting" || engine?.model_id !== chat.model_id)
      ? (models.find((m) => m.id === chat.model_id)?.name ?? "the model")
      : null;

  return (
    <>
      <ChatList chats={chats} models={models} activeId={activeId} onSelect={setActiveId} onNew={newChat} />
      <section className="col center">
        <div className="head">
          <span className="title ell">{chat?.title || "Chat"}</span>
          <span className="cap">{models.find((m) => m.id === chat?.model_id)?.name ?? ""}</span>
        </div>
        <div
          ref={transcript}
          className="scroll messages"
          aria-label="messages"
          onScroll={(e) => {
            const el = e.currentTarget;
            atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < AT_BOTTOM_PX;
          }}
        >
          {!chat && <div className="dim mono" style={{ fontSize: 11 }}>Type a message to start a chat.</div>}
          {messages.map((mv, i) => (
            <Fragment key={mv.message.id}>
              <MessageRow
                mv={mv}
                liveText={mv.message.id === live.messageId && (live.busy || mv.message.status === "streaming") ? live.text : null}
                pending={waiting && waiting.id === mv.message.id ? { since: waiting.since, loading } : null}
                onVersion={selectVersion}
              />
              {dropped > 0 && i === cutAfter && (
                <div className="history-cut cap">
                  {dropped} earlier turn{dropped === 1 ? "" : "s"} not sent: outside the ~1,500-token history budget
                </div>
              )}
            </Fragment>
          ))}
        </div>
        {(notice ?? live.error) && <div className="notice" role="alert">{notice ?? live.error}</div>}
        <Composer draft={draft} onDraft={setDraft} busy={busy} canRegenerate={!!lastAssistant}
                  onSend={send} onStop={stop} onRegenerate={regenerate} />
      </section>
      <ReplyPanel chat={chat} modelId={picker} models={models} overview={overview} live={live} busy={busy}
                  last={lastAssistant ? statsOf(lastAssistant.message) : null} onModel={setModel} />
    </>
  );
}
