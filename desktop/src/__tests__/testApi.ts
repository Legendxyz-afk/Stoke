import { vi } from "vitest";
import type { Overview } from "../ipc/bindings/Overview";

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
    selectVersion: vi.fn(async () => {}),
    requests: vi.fn(async () => []),
    logs: vi.fn(async () => []),
    onEngineStatus: vi.fn(async () => () => {}),
  };
}
