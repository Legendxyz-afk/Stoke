import { Fragment, useCallback, useEffect, useRef, useState } from "react";
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

export function ChatTab({ overview }: { overview: Overview | null }) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [view, setView] = useState<ChatView | null>(null);
  const [live, setLive] = useState<Live>(idle);
  const [notice, setNotice] = useState<string | null>(null);
  /** The composer's text lives here, not in Composer, so a message the host refuses can be put back (R36b). */
  const [draft, setDraft] = useState("");
  const active = useRef<string | null>(null);
  const pending = useRef("");
  const timer = useRef<number | null>(null);

  const refreshChats = useCallback(async () => {
    const cs = await api.chats();
    setChats(cs);
    return cs;
  }, []);

  const reload = useCallback(async (chatId: string) => {
    const v = await api.chat(chatId);
    if (active.current === chatId) setView(v);
  }, []);

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
    refreshChats()
      .then((cs) => {
        if (cs.length > 0) setActiveId((id) => id ?? cs[0].id);
      })
      .catch((e) => setNotice(String(e)));
  }, [refreshChats]);

  useEffect(() => {
    active.current = activeId;
    setLive(idle);
    setNotice(null);
    if (activeId) reload(activeId).catch((e) => setNotice(String(e)));
    else setView(null);
  }, [activeId, reload]);

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
    reload(chatId).catch(() => {});
    if (e.kind !== "started") refreshChats().catch(() => {});
  };

  /** `restore` is the text of a send. When the host refuses it (`call` rejects: busy, too long, a model that cannot
   *  run) nothing was stored, so the text goes back into the composer, unless something new was typed meanwhile. */
  const run = async (chatId: string, call: (cb: (e: StreamEvent) => void) => Promise<void>, restore?: string) => {
    setNotice(null);
    setLive(begin());
    try {
      await call(onEvent(chatId));
      // The host always ends a reply with an event; never leave the composer stuck on STOP if one is lost.
      if (active.current === chatId) setLive((s) => (s.busy ? { ...s, busy: false } : s));
    } catch (e) {
      if (restore !== undefined) setDraft((d) => (d === "" ? restore : d));
      if (active.current === chatId) {
        setLive(idle);
        setNotice(String(e));
      }
    }
  };

  const send = (text: string) => {
    const id = activeId;
    if (!id) return;
    setDraft("");
    void run(id, (cb) => api.send(id, text, cb), text);
  };
  const regenerate = () => {
    const id = activeId;
    if (id) void run(id, (cb) => api.regenerate(id, cb));
  };
  const stop = () => {
    if (activeId) api.stop(activeId).catch(() => {});
  };
  const newChat = async () => {
    try {
      const c = await api.newChat(null);
      await refreshChats();
      setActiveId(c.id);
    } catch (e) {
      setNotice(String(e));
    }
  };
  const setModel = async (modelId: string) => {
    const id = activeId;
    if (!id) return;
    try {
      await api.setChatModel(id, modelId);
      await reload(id);
      await refreshChats();
    } catch (e) {
      setNotice(String(e));
    }
  };
  const selectVersion = async (messageId: string) => {
    const id = activeId;
    if (!id) return;
    await api.selectVersion(id, messageId);
    await reload(id);
  };

  const messages = view?.messages ?? [];
  const chat = view?.chat ?? null;
  const lastAssistant = [...messages].reverse().find((m) => m.message.role === "assistant");
  const busy = live.busy || messages.some((m) => m.message.status === "streaming");
  const dropped = live.dropped;
  const cutAfter = lastDroppedIndex(messages, dropped);

  return (
    <>
      <ChatList chats={chats} models={models} activeId={activeId} onSelect={setActiveId} onNew={newChat} />
      <section className="col center">
        <div className="head">
          <span className="title ell">{chat?.title || "Chat"}</span>
          <span className="cap">{models.find((m) => m.id === chat?.model_id)?.name ?? ""}</span>
        </div>
        <div className="scroll messages" aria-label="messages">
          {!chat && <div className="dim mono" style={{ fontSize: 11 }}>Start a chat with + NEW.</div>}
          {messages.map((mv, i) => (
            <Fragment key={mv.message.id}>
              <MessageRow
                mv={mv}
                liveText={mv.message.id === live.messageId && (live.busy || mv.message.status === "streaming") ? live.text : null}
                onVersion={selectVersion}
              />
              {dropped > 0 && i === cutAfter && (
                <div className="history-cut cap">
                  {dropped} earlier turn{dropped === 1 ? "" : "s"} not sent: outside the ~1,500-token history budget
                </div>
              )}
            </Fragment>
          ))}
          {live.busy && live.messageId == null && (
            <div className="waiting">waiting for the engine ({overview?.engine.state ?? "starting"})…</div>
          )}
        </div>
        {(notice ?? live.error) && <div className="notice" role="alert">{notice ?? live.error}</div>}
        <Composer draft={draft} onDraft={setDraft} busy={busy} disabled={!chat} canRegenerate={!!lastAssistant}
                  onSend={send} onStop={stop} onRegenerate={regenerate} />
      </section>
      <ReplyPanel chat={chat} models={models} overview={overview} live={live} busy={busy}
                  last={lastAssistant ? statsOf(lastAssistant.message) : null} onModel={setModel} />
    </>
  );
}
