import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { App } from "../App";
import { ChatTab } from "../chat/ChatTab";
import { api } from "../ipc/api";
import { resetDiscoverSession } from "../tabs/ModelsTab";
import { catalogView, model, models, overview } from "./testApi";

beforeEach(() => {
  vi.clearAllMocks();
  resetDiscoverSession();
  vi.mocked(api.overview).mockResolvedValue(overview);
  vi.mocked(api.chats).mockResolvedValue([]);
  vi.mocked(api.refreshCatalog).mockResolvedValue(catalogView);
});

describe("the chat with no model", () => {
  it("with_no_runnable_model_the_chat_says_so_and_its_button_opens_discover", async () => {
    vi.mocked(api.models).mockResolvedValue([model({ id: "o", name: "o", runnable: false, reason: "no" })]);
    const onDiscover = vi.fn();
    render(<ChatTab overview={overview} onDiscover={onDiscover} />);
    expect(await screen.findByText(/No model yet\. Lorren needs one to chat: get one from Discover/)).toBeTruthy();
    const composer = screen.getByLabelText("message") as HTMLTextAreaElement;
    expect(composer.disabled).toBe(true);
    expect(composer.placeholder).toBe("Get a model from Discover to start chatting");
    expect((screen.getByText("SEND") as HTMLButtonElement).disabled).toBe(true);
    const select = screen.getByLabelText("model") as HTMLSelectElement;
    expect(select.disabled).toBe(true);
    expect(Array.from(select.options).map((o) => o.text)).toEqual(["no model"]);
    fireEvent.click(screen.getByText("OPEN DISCOVER"));
    expect(onDiscover).toHaveBeenCalledTimes(1);
  });

  it("with a runnable model it is the ordinary chat", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<ChatTab overview={overview} onDiscover={() => {}} />);
    expect(await screen.findByText("Type a message to start a chat.")).toBeTruthy();
    expect((screen.getByLabelText("message") as HTMLTextAreaElement).disabled).toBe(false);
    expect(screen.queryByText("OPEN DISCOVER")).toBeNull();
  });

  it("the chat's button opens the Models tab on Discover", async () => {
    vi.mocked(api.models).mockResolvedValue([]);
    render(<App />);
    fireEvent.click(await screen.findByText("OPEN DISCOVER"));
    expect(await screen.findByText(/IN THE CATALOG/)).toBeTruthy();
    expect(api.refreshCatalog).toHaveBeenCalledTimes(1);
  });
});

describe("the Models tab", () => {
  it("the_models_tab_opens_on_discover_when_the_library_is_empty", async () => {
    vi.mocked(api.models).mockResolvedValue([]);
    render(<App />);
    fireEvent.click(await screen.findByRole("tab", { name: "Models" }));
    expect((await screen.findAllByText("Qwen3-4B-Instruct-2507")).length).toBeGreaterThan(0);
    expect(api.refreshCatalog).toHaveBeenCalledTimes(1);
  });

  it("opens on the library, with Discover closed, when there is a model", async () => {
    vi.mocked(api.models).mockResolvedValue(models);
    render(<App />);
    fireEvent.click(await screen.findByRole("tab", { name: "Models" }));
    await screen.findByText("nexus");
    expect(screen.getByText("OPEN")).toBeTruthy();
    await waitFor(() => expect(api.catalog).toHaveBeenCalled());
    expect(api.refreshCatalog).not.toHaveBeenCalled();
  });
});
