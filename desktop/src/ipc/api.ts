import { Channel, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { Chat } from "./bindings/Chat";
import type { ChatView } from "./bindings/ChatView";
import type { EngineStatus } from "./bindings/EngineStatus";
import type { LogLine } from "./bindings/LogLine";
import type { ModelInfo } from "./bindings/ModelInfo";
import type { Overview } from "./bindings/Overview";
import type { RequestRecord } from "./bindings/RequestRecord";
import type { StreamEvent } from "./bindings/StreamEvent";
import type { UpdateState } from "./bindings/UpdateState";

function channel(onEvent: (e: StreamEvent) => void): Channel<StreamEvent> {
  const ch = new Channel<StreamEvent>();
  ch.onmessage = onEvent;
  return ch;
}

export const api = {
  overview: () => invoke<Overview>("overview"),
  models: () => invoke<ModelInfo[]>("models"),
  chats: () => invoke<Chat[]>("chats"),
  chat: (chatId: string) => invoke<ChatView>("chat", { chatId }),
  newChat: (modelId: string | null) => invoke<Chat>("new_chat", { modelId }),
  send: (chatId: string, text: string, onEvent: (e: StreamEvent) => void) =>
    invoke<void>("send", { chatId, text, onEvent: channel(onEvent) }),
  regenerate: (chatId: string, onEvent: (e: StreamEvent) => void) =>
    invoke<void>("regenerate", { chatId, onEvent: channel(onEvent) }),
  stop: (chatId: string) => invoke<void>("stop", { chatId }),
  setChatModel: (chatId: string, modelId: string) => invoke<void>("set_chat_model", { chatId, modelId }),
  selectVersion: (chatId: string, messageId: string) => invoke<void>("select_version", { chatId, messageId }),
  requests: () => invoke<RequestRecord[]>("requests"),
  logs: (after: number) => invoke<LogLine[]>("logs", { after }),
  updateState: () => invoke<UpdateState>("update_state"),
  checkUpdate: () => invoke<void>("check_update"),
  installUpdate: () => invoke<void>("install_update"),
  onEngineStatus: (cb: (s: EngineStatus) => void): Promise<UnlistenFn> =>
    listen<EngineStatus>("engine-status", (e) => cb(e.payload)),
  onUpdateState: (cb: (s: UpdateState) => void): Promise<UnlistenFn> =>
    listen<UpdateState>("update-state", (e) => cb(e.payload)),
};

export type Api = typeof api;
