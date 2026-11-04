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
  /** The user has started a draft in a chat on `modelId` (or in the box that makes one), and the host decides what that means: it
   *  loads the model if the engine is idle and does not have it, without waiting for the load, and does nothing if the engine has
   *  it. While a reply is being written in any chat the engine is left alone and the host keeps the model, the latest one asked
   *  for, the engine's own included, and settles it when the last reply has ended (R76, R77). */
  prepareModel: (modelId: string) => invoke<void>("prepare_model", { modelId }),
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
