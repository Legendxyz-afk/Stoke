import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../ipc/api", async () => ({ api: (await import("./testApi")).mockApi() }));

import { api } from "../ipc/api";
import type { DownloadView } from "../ipc/bindings/DownloadView";
import { ModelsTab, resetDiscoverSession } from "../tabs/ModelsTab";
import { catalogView, diskRoom, dl, llamaEntry, model, overview, qwenEntry } from "./testApi";

const GIB = 1073741824;
const BS = String.fromCharCode(92);

/** The Models tab with the view held the way App holds it. */
function Host({ start = "library" }: { start?: "library" | "discover" }) {
  const [view, setView] = useState<"library" | "discover">(start);
  return <ModelsTab overview={overview} view={view} onView={setView} />;
}

/** The newest callback the tab gave to an event subscription. */
function fire(which: "onDownloadState", d: DownloadView): void;
function fire(which: "onLibraryChanged"): void;
function fire(which: "onDownloadState" | "onLibraryChanged", d?: DownloadView) {
  const calls = vi.mocked(api[which]).mock.calls as unknown as Array<[(x?: DownloadView) => void]>;
  act(() => calls[calls.length - 1][0](d));
}

// The last match: a running download also has a row in the library table above Discover.
const rowOrNull = (name: string) => screen.queryAllByRole("row").filter((r) => within(r).queryByText(name)).pop() ?? null;
const row = (name: string) => {
  const r = rowOrNull(name);
  if (!r) throw new Error(`no row named ${name}`);
  return r;
};
/** The row named `name`, once it is there (the same name is also the right panel's title, so findByText would find two). */
const findRow = (name: string) => waitFor(() => row(name));

beforeEach(() => {
  vi.clearAllMocks();
  resetDiscoverSession();
  vi.mocked(api.models).mockResolvedValue([]);
  vi.mocked(api.downloads).mockResolvedValue([]);
  vi.mocked(api.catalog).mockResolvedValue(catalogView);
  vi.mocked(api.refreshCatalog).mockResolvedValue(catalogView);
  vi.mocked(api.diskRoom).mockResolvedValue(diskRoom);
});

describe("Discover", () => {
  it("opening_discover_reads_the_catalog_once_and_the_library_tab_never_does", async () => {
    vi.mocked(api.models).mockResolvedValue([model({ id: "file:a", name: "a" })]);
    const { unmount } = render(<Host />);
    await findRow("a");
    // The library view: the Discover section is closed to its head, and nothing is read from the network.
    expect(screen.getByText("Models you can download. Lorren reads the list from the internet only when you open it.")).toBeTruthy();
    expect(api.refreshCatalog).not.toHaveBeenCalled();
    expect(rowOrNull("Qwen3-4B-Instruct-2507")).toBeNull();

    fireEvent.click(screen.getByText("OPEN"));
    await findRow("Qwen3-4B-Instruct-2507");
    expect(api.refreshCatalog).toHaveBeenCalledTimes(1);

    // A later opening, in this tab or after it was left, shows the kept view.
    unmount();
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    expect(api.refreshCatalog).toHaveBeenCalledTimes(1);
    expect(api.catalog).toHaveBeenCalled();
  });

  it("each_entry_shows_size_licence_and_disk_room_before_GET", async () => {
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    const r = row("Qwen3-4B-Instruct-2507");
    expect(within(r).getByText("2.3 GB")).toBeTruthy();
    expect(within(r).getByText("Apache-2.0")).toBeTruthy();
    expect(within(r).getByText("GET")).toBeTruthy();
    expect(row("Llama-3.2-3B-Instruct")).toBeTruthy();
    fireEvent.click(row("Qwen3-4B-Instruct-2507"));
    expect(await screen.findByText("needs 2.3 GB · 412.0 GB free")).toBeTruthy();
    expect(screen.getByRole("button", { name: "GET" })).toHaveProperty("disabled", false);
  });

  it("with too little room the panel says so and GET is disabled", async () => {
    vi.mocked(api.diskRoom).mockResolvedValue({ ...diskRoom, free_bytes: 2.5 * GIB });
    render(<Host start="discover" />);
    expect(await screen.findByText("needs 2.3 GB · 2.5 GB free: free some space first")).toBeTruthy();
    expect(screen.getByRole("button", { name: "GET" })).toHaveProperty("disabled", true);
  });

  it("GET_starts_and_the_row_follows_download_state_events", async () => {
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    fireEvent.click(screen.getByRole("button", { name: "GET" }));
    await waitFor(() => expect(api.startDownload).toHaveBeenCalledWith("qwen3-4b"));
    expect((await screen.findAllByText("DOWNLOADING")).length).toBe(2); // the library row and the panel

    fire("onDownloadState", dl("qwen3-4b", { done: 1.2 * GIB, bytes_per_sec: 18.4 * 1048576 }));
    expect(within(row("Qwen3-4B-Instruct-2507")).getByText("51%")).toBeTruthy();
    expect(screen.getByText("1.2 GB / 2.3 GB · 51%")).toBeTruthy();
    expect(screen.getByText(/18\.4 MB\/s · \d+ s left/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "PAUSE" }));
    await waitFor(() => expect(api.pauseDownload).toHaveBeenCalledWith("qwen3-4b"));
    fire("onDownloadState", dl("qwen3-4b", { state: "paused", done: 1.2 * GIB }));
    expect(within(row("Qwen3-4B-Instruct-2507")).getByText("PAUSED")).toBeTruthy();
    expect(screen.getByText("paused at 51%")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "RESUME" }));
    await waitFor(() => expect(api.resumeDownload).toHaveBeenCalledWith("qwen3-4b"));
    fire("onDownloadState", dl("qwen3-4b", { done: 1.3 * GIB, bytes_per_sec: 1048576 }));
    expect(screen.getByRole("button", { name: "PAUSE" })).toBeTruthy();

    fire("onDownloadState", dl("qwen3-4b", { state: "verifying", done: 2_497_280_736 }));
    expect(within(row("Qwen3-4B-Instruct-2507")).getByText("VERIFYING")).toBeTruthy();
    expect(screen.getByText("checking the file against its sha256")).toBeTruthy();

    // CANCEL: the host emits no event, so the UI drops the entry's view itself once the call resolves.
    fire("onDownloadState", dl("qwen3-4b", { done: 1.3 * GIB }));
    fireEvent.click(screen.getByRole("button", { name: "CANCEL" }));
    await waitFor(() => expect(api.cancelDownload).toHaveBeenCalledWith("qwen3-4b"));
    await waitFor(() => expect(within(row("Qwen3-4B-Instruct-2507")).getByText("GET")).toBeTruthy());
  });

  it("retry_in_ms_shows_the_retry_line", async () => {
    vi.mocked(api.downloads).mockResolvedValue([dl("qwen3-4b", { done: 1.2 * GIB, retry_in_ms: 4200 })]);
    render(<Host start="discover" />);
    expect(await screen.findByText("connection lost · retrying in 5 s")).toBeTruthy();
    fire("onDownloadState", dl("qwen3-4b", { done: 1.2 * GIB, retry_in_ms: 1000 }));
    expect(screen.getByText("connection lost · retrying in 1 s")).toBeTruthy();
    fire("onDownloadState", dl("qwen3-4b", { done: 1.25 * GIB, bytes_per_sec: 1048576 }));
    expect(screen.queryByText(/connection lost/)).toBeNull();
  });

  it("a_failed_download_shows_its_error_and_offers_GET_again", async () => {
    vi.mocked(api.downloads).mockResolvedValue([dl("qwen3-4b", { state: "failed", error: "the file's sha256 is not the catalog's" })]);
    render(<Host start="discover" />);
    expect(await screen.findByText("the file's sha256 is not the catalog's")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "GET" }));
    await waitFor(() => expect(api.startDownload).toHaveBeenCalledWith("qwen3-4b"));
  });

  it("a refusal of GET is shown in the panel in the flame colour", async () => {
    vi.mocked(api.startDownload).mockRejectedValue("needs 2.3 GB with 1 GB to spare; 2.5 GB free");
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    fireEvent.click(screen.getByRole("button", { name: "GET" }));
    const line = await screen.findByText("needs 2.3 GB with 1 GB to spare; 2.5 GB free");
    expect(line.className).toContain("flame-text");
  });

  it("an_entry_on_disk_or_in_the_library_has_no_GET", async () => {
    const onDisk = model({ id: "file:q", name: "Qwen3-4B-Instruct-2507-Q4_K_M",
                           path: `M:${BS}New Drive${BS}ai_models${BS}lorren${BS}${qwenEntry.file.toLowerCase()}` });
    const inLib = model({ id: "file:l", name: "Llama-3.2-3B-Instruct-Q4_K_M",
                          path: `M:/New Drive/ai_models/lorren/${llamaEntry.file}`, source: "folder" });
    vi.mocked(api.models).mockResolvedValue([onDisk, inLib]);
    vi.mocked(api.downloads).mockResolvedValue([dl("llama-3.2-3b", { state: "done", done: llamaEntry.size })]);
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    // the file name is already a library file, with no download row of ours
    expect(within(row("Qwen3-4B-Instruct-2507")).getByText("ON DISK")).toBeTruthy();
    expect(within(row("Llama-3.2-3B-Instruct")).getByText("IN LIBRARY")).toBeTruthy();
    fireEvent.click(row("Qwen3-4B-Instruct-2507"));
    expect(screen.getByText("Qwen3-4B-Instruct-2507-Q4_K_M.gguf is already in your library. Lorren never replaces a file.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "GET" })).toBeNull();
    fireEvent.click(row("Llama-3.2-3B-Instruct"));
    expect(screen.getByText("Saved as Llama-3.2-3B-Instruct-Q4_K_M.gguf. Chat can use it now.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "GET" })).toBeNull();
    // the library table names a downloaded model's source "discover"
    const libRow = screen.getByText("Llama-3.2-3B-Instruct-Q4_K_M").closest("[role=row]") as HTMLElement;
    expect(within(libRow).getByText("discover")).toBeTruthy();
    const folderRow = screen.getByText("Qwen3-4B-Instruct-2507-Q4_K_M").closest("[role=row]") as HTMLElement;
    expect(within(folderRow).getByText("discover")).toBeTruthy();
  });

  it("an older downloaded version says UPDATE AVAILABLE and offers GET", async () => {
    vi.mocked(api.catalog).mockResolvedValue({ ...catalogView, entries: [{ ...qwenEntry, version: 2 }, llamaEntry] });
    vi.mocked(api.refreshCatalog).mockResolvedValue({ ...catalogView, entries: [{ ...qwenEntry, version: 2 }, llamaEntry] });
    vi.mocked(api.models).mockResolvedValue([model({ id: "file:q", name: "Qwen3-4B-Instruct-2507-Q4_K_M", path: `M:/New Drive/ai_models/lorren/${qwenEntry.file}` })]);
    vi.mocked(api.downloads).mockResolvedValue([dl("qwen3-4b", { state: "done", version: 1, done: qwenEntry.size })]);
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    expect(within(row("Qwen3-4B-Instruct-2507")).getByText("UPDATE")).toBeTruthy();
    fireEvent.click(row("Qwen3-4B-Instruct-2507"));
    expect(screen.getByText("UPDATE AVAILABLE")).toBeTruthy();
    expect(screen.getByText("Version 2 is out; you have version 1. It is saved beside the old file.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "GET" }));
    await waitFor(() => expect(api.startDownload).toHaveBeenCalledWith("qwen3-4b"));
  });

  it("an_entry_for_a_newer_app_says_so_and_cannot_be_started", async () => {
    const view = { ...catalogView, entries: [qwenEntry, { ...llamaEntry, min_app_version: "9.0.0" }], needs_app: ["llama-3.2-3b"] };
    vi.mocked(api.refreshCatalog).mockResolvedValue(view);
    render(<Host start="discover" />);
    await findRow("Llama-3.2-3B-Instruct");
    expect(within(row("Llama-3.2-3B-Instruct")).getByText("NEWER APP")).toBeTruthy();
    fireEvent.click(row("Llama-3.2-3B-Instruct"));
    expect(screen.getByText("NEEDS A NEWER LORREN")).toBeTruthy();
    expect(screen.getByText("This model needs Lorren 9.0.0 or newer.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "GET" })).toBeNull();
  });

  it("a_stale_catalog_says_when_it_was_read_and_why", async () => {
    // Read today a few minutes ago: the time of day alone.
    const at = Date.now() - 3 * 60000;
    const stale = { ...catalogView, stale: true, fetched_ms: at, error: "could not reach the catalog" };
    vi.mocked(api.refreshCatalog).mockResolvedValueOnce(stale).mockResolvedValue(catalogView);
    const { unmount } = render(<Host start="discover" />);
    const clockOf = new Date(at).toLocaleTimeString("en-GB", { hour12: false });
    expect(await screen.findByText(`the list as read ${clockOf} · reading it again failed: could not reach the catalog`)).toBeTruthy();
    unmount();
    // Read 3 days ago: the day too (`6 Nov 09:41`), so it cannot read as today.
    resetDiscoverSession();
    const old = new Date(Date.now() - 3 * 86400000);
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const hh = String(old.getHours()).padStart(2, "0"), mm = String(old.getMinutes()).padStart(2, "0");
    vi.mocked(api.refreshCatalog).mockResolvedValueOnce({ ...stale, fetched_ms: old.getTime() });
    render(<Host start="discover" />);
    expect(await screen.findByText(`the list as read ${old.getDate()} ${months[old.getMonth()]} ${hh}:${mm} · reading it again failed: could not reach the catalog`)).toBeTruthy();
    fireEvent.click(screen.getByText("READ AGAIN"));
    await waitFor(() => expect(screen.queryByText(/reading it again failed/)).toBeNull());
    expect(api.refreshCatalog).toHaveBeenCalledTimes(2);
  });

  it("says when the build has no catalog, and when the first read failed", async () => {
    vi.mocked(api.refreshCatalog).mockResolvedValueOnce({ ...catalogView, configured: false, entries: [] });
    const { unmount } = render(<Host start="discover" />);
    expect(await screen.findByText("This build has no model catalog.")).toBeTruthy();
    unmount();
    resetDiscoverSession();
    vi.mocked(api.refreshCatalog).mockResolvedValueOnce({ ...catalogView, entries: [], stale: true, error: "offline" });
    render(<Host start="discover" />);
    expect(await screen.findByText("The catalog could not be read: offline")).toBeTruthy();
    expect(screen.getByText("READ AGAIN")).toBeTruthy();
  });

  it("the_licence_link_calls_openLicence_and_navigates_nowhere", async () => {
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    const link = await screen.findByText("Apache-2.0", { selector: "button" });
    // Nothing in the panel carries an href: a middle-click or "open in new window" has nothing to open.
    expect(link.hasAttribute("href")).toBe(false);
    expect(document.querySelector("a[href]")).toBeNull();
    fireEvent.click(link);
    expect(api.openLicence).toHaveBeenCalledWith("qwen3-4b");
  });

  it("the search bar filters the entries by name", async () => {
    render(<Host start="discover" />);
    await findRow("Qwen3-4B-Instruct-2507");
    fireEvent.change(screen.getByLabelText("search the catalog"), { target: { value: "llama" } });
    fireEvent.click(screen.getByText("SEARCH"));
    expect(rowOrNull("Qwen3-4B-Instruct-2507")).toBeNull();
    expect(row("Llama-3.2-3B-Instruct")).toBeTruthy();
  });

  it("library_changed_reloads_the_library_table", async () => {
    render(<Host start="library" />);
    await waitFor(() => expect(api.diskRoom).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("Qwen3-4B-Instruct-2507-Q4_K_M")).toBeNull();
    vi.mocked(api.models).mockResolvedValue([model({ id: "file:q", name: "Qwen3-4B-Instruct-2507-Q4_K_M", path: `${diskRoom.dir}/${qwenEntry.file}` })]);
    fire("onLibraryChanged");
    expect(await screen.findByText("Qwen3-4B-Instruct-2507-Q4_K_M")).toBeTruthy();
    expect(api.diskRoom).toHaveBeenCalledTimes(2);
  });

  it("the_library_table_shows_DOWNLOADING_for_a_running_download", async () => {
    vi.mocked(api.models).mockResolvedValue([model({ id: "file:a", name: "a" })]);
    vi.mocked(api.downloads).mockResolvedValue([dl("qwen3-4b", { done: 1 * GIB })]);
    render(<Host start="library" />);
    const r = await findRow("Qwen3-4B-Instruct-2507");
    for (const w of ["discover", "qwen3", "Q4_K_M", "2.3 GB", "—", "DOWNLOADING"]) expect(within(r).getByText(w)).toBeTruthy();
    fire("onDownloadState", dl("qwen3-4b", { state: "paused", done: 1 * GIB }));
    expect(within(r).getByText("PAUSED")).toBeTruthy();
    fire("onDownloadState", dl("qwen3-4b", { state: "done", done: qwenEntry.size }));
    expect(rowOrNull("Qwen3-4B-Instruct-2507")).toBeNull();
  });

  it("unsubscribes from both events when the tab goes away", async () => {
    const off = vi.fn();
    vi.mocked(api.onDownloadState).mockResolvedValue(off);
    vi.mocked(api.onLibraryChanged).mockResolvedValue(off);
    const { unmount } = render(<Host />);
    await waitFor(() => expect(api.onLibraryChanged).toHaveBeenCalled());
    unmount();
    await waitFor(() => expect(off).toHaveBeenCalledTimes(2));
  });
});
