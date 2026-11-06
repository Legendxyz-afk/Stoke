import { vi } from "vitest";
import type { Api } from "../ipc/api";
import type { CatalogEntry } from "../ipc/bindings/CatalogEntry";
import type { CatalogView } from "../ipc/bindings/CatalogView";
import type { DiskRoom } from "../ipc/bindings/DiskRoom";
import type { DownloadView } from "../ipc/bindings/DownloadView";
import type { Chat } from "../ipc/bindings/Chat";
import type { ChatView } from "../ipc/bindings/ChatView";
import type { Message } from "../ipc/bindings/Message";
import type { ModelInfo } from "../ipc/bindings/ModelInfo";
import type { Overview } from "../ipc/bindings/Overview";
import type { UpdateState } from "../ipc/bindings/UpdateState";

export const overview: Overview = {
  engine: {
    state: "serving",
    model_id: "file:q",
    model_path: "M:\\New Drive\\Desktop\\Programming\\stoke\\models\\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    port: 50123,
    pid: 42,
    ctx: 4096,
    seqs: 2,
    threads: null,
    kv: "q8",
    load_ms: 2160,
    restarts: 0,
    last_error: null,
  },
  memory: { app_bytes: 150 * 1048576, engine_bytes: 2.4 * 1073741824, used_bytes: 10 * 1073741824, total_bytes: 15.6 * 1073741824 },
  app_version: "0.0.1",
  started_ms: Date.now() - 65 * 60000,
  request_count: 3,
};

const GIB = 1073741824;

export const qwenEntry: CatalogEntry = {
  id: "qwen3-4b", name: "Qwen3-4B-Instruct-2507", version: 1,
  description: "Qwen3 4B Instruct, July 2025: Lorren's default, quick and capable on the processor.",
  architecture: "qwen3", quantization: "Q4_K_M", file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", size: 2_497_280_736,
  sha256: "2fde00ce", url: "https://huggingface.co/x/resolve/ae44f08/q.gguf",
  licence: { name: "Apache-2.0", url: "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/LICENSE" },
  source: { repo: "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF", commit: "ae44f08f00d", path: "q.gguf" },
  system_prompt: null, min_app_version: "0.1.0",
};
export const llamaEntry: CatalogEntry = {
  ...qwenEntry, id: "llama-3.2-3b", name: "Llama-3.2-3B-Instruct", description: "Meta's Llama 3.2 3B Instruct.",
  architecture: "llama", file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", size: 2_019_377_696,
  licence: { name: "Llama 3.2 Community License", url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt" },
  source: { repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", commit: "5ab33fa1234", path: "l.gguf" },
};
export const catalogView: CatalogView = {
  configured: true, entries: [qwenEntry, llamaEntry], needs_app: [], fetched_ms: Date.now(), stale: false, error: null,
};
export const diskRoom: DiskRoom = { dir: "M:/New Drive/ai_models/lorren", free_bytes: 412 * GIB };

export function dl(entryId: string, p: Partial<DownloadView>): DownloadView {
  const e = [qwenEntry, llamaEntry].find((x) => x.id === entryId) ?? qwenEntry;
  return {
    entry_id: entryId, name: e.name, file: e.file, version: e.version, state: "downloading", done: 0, total: e.size,
    bytes_per_sec: 0, retry_in_ms: null, error: null, ...p,
  };
}

/** The module the tests put in place of `../ipc/api`. `satisfies Api` makes a mock that drifts from the real API (a call
 *  renamed, one missing, one whose arguments or result changed) a type error, not a test that runs against nothing. */
export function mockApi() {
  return {
    overview: vi.fn(async () => overview),
    models: vi.fn(async () => []),
    chats: vi.fn(async () => []),
    chat: vi.fn(),
    newChat: vi.fn(),
    send: vi.fn(async () => {}),
    regenerate: vi.fn(async () => {}),
    stop: vi.fn(async () => {}),
    setChatModel: vi.fn(async () => {}),
    prepareModel: vi.fn(async () => {}),
    selectVersion: vi.fn(async () => {}),
    requests: vi.fn(async () => []),
    logs: vi.fn(async () => []),
    updateState: vi.fn(async (): Promise<UpdateState> => ({ state: "not_configured" })),
    checkUpdate: vi.fn(async () => {}),
    installUpdate: vi.fn(async () => {}),
    catalog: vi.fn(async (): Promise<CatalogView> => catalogView),
    refreshCatalog: vi.fn(async (): Promise<CatalogView> => catalogView),
    downloads: vi.fn(async (): Promise<DownloadView[]> => []),
    diskRoom: vi.fn(async (): Promise<DiskRoom> => diskRoom),
    startDownload: vi.fn(async (id: string): Promise<DownloadView> => dl(id, { state: "downloading" })),
    pauseDownload: vi.fn(async () => {}),
    resumeDownload: vi.fn(async () => {}),
    cancelDownload: vi.fn(async () => {}),
    openLicence: vi.fn(async () => {}),
    onEngineStatus: vi.fn(async () => () => {}),
    onDownloadState: vi.fn(async () => () => {}),
    onLibraryChanged: vi.fn(async () => () => {}),
    onUpdateState: vi.fn(async () => () => {}),
  } satisfies Api;
}

/** Take the "<call> failed:" lines that ChatTab logs when a host call is rejected (R57) out of stderr and into the
 *  returned spy. Any other console.error, React's act() warnings among them, still reaches stderr, and is in the
 *  spy's calls too, so a test that compares them with the expected lines also fails on a stray warning. */
export function captureHostErrors() {
  const real = console.error.bind(console);
  return vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
    if (typeof args[0] === "string" && /^\w+ failed:$/.test(args[0])) return;
    real(...args);
  });
}

export function model(p: Partial<ModelInfo>): ModelInfo {
  return {
    id: "file:x", name: "x", path: "M:\\New Drive\\ai_models\\x.gguf", source: "folder",
    size_bytes: 2_019_377_696, arch: "llama", quant: "Q4_K_M", size_label: "3B",
    n_layer: 28, n_head: 24, n_head_kv: 8, head_dim: 128, vocab: 128256,
    ctx_file: 131072, ctx_engine: 4096, kv_bytes_per_token: 60928, run_bytes: 2_786_935_328, template: "llama3",
    runnable: true, reason: null, system_prompt: null, system_source: null, is_default: false,
    ...p,
  };
}

export const models: ModelInfo[] = [
  model({ id: "file:q", name: "Qwen3-4B-Instruct-2507-Q4_K_M", arch: "qwen3", template: "chatml", n_layer: 36,
          kv_bytes_per_token: 78336, size_bytes: 2_497_280_736, run_bytes: 3_407_444_704, is_default: true }),
  model({ id: "file:n", name: "nexus", system_source: "Modelfile",
          system_prompt: "You are Nexus AI, a professional quantitative trader. You give concise, decisive trading verdicts." }),
  model({ id: "ollama:qwen2.5vl:3b", name: "qwen2.5vl:3b", source: "ollama", arch: "qwen2vl", runnable: false,
          reason: "architecture qwen2vl is not supported by the engine (llama and qwen3 only)" }),
];

const noStats = { prompt_tokens: null, completion_tokens: null, ttft_ms: null, total_ms: null, error: null };

export function msg(p: Partial<Message> & Pick<Message, "id" | "role" | "content">): Message {
  return { chat_id: "c1", parent_id: null, status: "done", model_id: null, stats: noStats, created_ms: 1, ...p };
}

export const chat: Chat = {
  id: "c1", title: "Long or wait?", model_id: "file:n", system_prompt: null,
  active_leaf_id: "a2", created_ms: 1, updated_ms: 2,
};

/** A chat whose reply was regenerated once: version 2 of 2 is showing. */
export const view: ChatView = {
  chat,
  messages: [
    { message: msg({ id: "u1", role: "user", content: "Long or wait?" }), version: 1, versions: 1, sibling_ids: ["u1"] },
    {
      message: msg({ id: "a2", role: "assistant", parent_id: "u1", model_id: "file:n", content: "**Verdict: HOLD.**",
                     stats: { prompt_tokens: 41, completion_tokens: 5, ttft_ms: 480, total_ms: 730, error: null } }),
      version: 2, versions: 2, sibling_ids: ["a1", "a2"],
    },
  ],
};

export function withTurn(base: ChatView, userText: string, reply: Message): ChatView {
  return {
    ...base,
    messages: [
      ...base.messages,
      { message: msg({ id: "u3", role: "user", parent_id: "a2", content: userText }), version: 1, versions: 1, sibling_ids: ["u3"] },
      { message: reply, version: 1, versions: 1, sibling_ids: [reply.id] },
    ],
  };
}
