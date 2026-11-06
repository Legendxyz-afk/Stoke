// A local file server for the e2e: the model catalog and its files (port 8766), or a copy of the M1 kit's update feed (8765).
//   node e2e/catalog-server.mjs <root> --port 8766 [--pace <bytes per second>] [--drop <file>:<bytes>] [--log <file>] [--exit-with-stdin]
// Serves any file under <root> (the catalog is <root>/catalog/*, the models <root>/files/*) for GET and HEAD, with Range
// (`bytes=N-` and `bytes=N-M`: 206 with Content-Range, 416 past the end). Every body is sent in 50 ms steps of pace/20 bytes,
// so a pace of 1,000,000 is 1 MB/s; no --pace sends at full speed. The FIRST request for <file> (a path ending in /<file>) is cut
// after <bytes> bytes of its body: the bytes are sent, then the connection is closed short of the Content-Length, which is what a
// dropped connection looks like to the client; later requests for it are served whole. Every request is appended to the log
// (default <root>/requests.log) as one JSON line { t, method, path, range }, when the request arrives. --exit-with-stdin ends
// the server when its stdin closes, so a run that dies cannot leave it behind.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const USAGE = "usage: node e2e/catalog-server.mjs <root> --port <port> [--pace <bytes per second>] [--drop <file>:<bytes>] [--log <file>] [--exit-with-stdin]";

function parse(argv) {
  const out = { root: null, port: null, pace: 0, drop: null, log: null, exitWithStdin: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) throw new Error(USAGE);
      return argv[++i];
    };
    if (a === "--port") out.port = Number(value());
    else if (a === "--pace") out.pace = Number(value());
    else if (a === "--drop") {
      const v = value();
      const cut = v.lastIndexOf(":");
      const bytes = Number(v.slice(cut + 1));
      if (cut < 1 || !Number.isInteger(bytes) || bytes < 0) throw new Error(`--drop wants <file>:<bytes>, not "${v}"`);
      out.drop = { file: v.slice(0, cut), bytes };
    } else if (a === "--log") out.log = value();
    else if (a === "--exit-with-stdin") out.exitWithStdin = true;
    else if (!a.startsWith("--") && out.root === null) out.root = a;
    else throw new Error(USAGE);
  }
  if (out.root === null || !Number.isInteger(out.port) || out.port < 1 || !(out.pace >= 0)) throw new Error(USAGE);
  return out;
}

let opts;
try {
  opts = parse(process.argv.slice(2));
} catch (e) {
  console.error(e.message);
  process.exit(2);
}
const root = path.resolve(opts.root);
const logPath = path.resolve(opts.log ?? path.join(root, "requests.log"));
fs.mkdirSync(path.dirname(logPath), { recursive: true });
if (!fs.existsSync(logPath)) fs.writeFileSync(logPath, "");
let dropped = false;

/** The file a URL path names under the root, or null (outside the root, not a file, or the log). */
function resolveFile(urlPath) {
  let rel;
  try {
    rel = decodeURIComponent(urlPath);
  } catch {
    return null;
  }
  const full = path.resolve(root, "." + path.posix.normalize("/" + rel));
  if (full !== root && !full.startsWith(root + path.sep)) return null;
  if (full === logPath) return null;
  try {
    return fs.statSync(full).isFile() ? full : null;
  } catch {
    return null;
  }
}

/** `bytes=N-` or `bytes=N-M` against a file of `total` bytes: { start, end } (inclusive), "unsatisfiable", or null (no range, or one that is not understood: the whole file). */
function parseRange(header, total) {
  const m = /^bytes=(\d+)-(\d*)$/.exec(header ?? "");
  if (!m) return null;
  const start = Number(m[1]);
  const end = m[2] === "" ? total - 1 : Math.min(Number(m[2]), total - 1);
  if (start >= total || start > end) return "unsatisfiable";
  return { start, end };
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  const range = req.headers.range ?? null;
  fs.appendFileSync(logPath, JSON.stringify({ t: Date.now(), method: req.method, path: url.pathname, range }) + "\n");
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405, { Allow: "GET, HEAD", Connection: "close" }).end();
    return;
  }
  const file = resolveFile(url.pathname);
  if (!file) {
    res.writeHead(404, { Connection: "close" }).end("not found");
    return;
  }
  const total = fs.statSync(file).size;
  const asked = parseRange(range, total);
  const common = { "Accept-Ranges": "bytes", "Content-Type": "application/octet-stream", Connection: "close" };
  if (asked === "unsatisfiable") {
    res.writeHead(416, { ...common, "Content-Range": `bytes */${total}`, "Content-Length": 0 }).end();
    return;
  }
  const start = asked ? asked.start : 0;
  const end = asked ? asked.end : total - 1;
  const length = end - start + 1;
  if (asked) res.writeHead(206, { ...common, "Content-Range": `bytes ${start}-${end}/${total}`, "Content-Length": length });
  else res.writeHead(200, { ...common, "Content-Length": length });
  if (req.method === "HEAD") {
    res.end();
    return;
  }

  // The first request for the file named by --drop is cut short after that many bytes of its body.
  let cutAt = length;
  if (opts.drop && !dropped && url.pathname.endsWith("/" + opts.drop.file)) {
    dropped = true;
    cutAt = Math.min(opts.drop.bytes, length);
  }
  const step = opts.pace > 0 ? Math.max(1, Math.round(opts.pace / 20)) : 1 << 20;
  const fd = fs.openSync(file, "r");
  let sent = 0;
  let timer = null;
  let done = false;
  const stop = () => {
    if (done) return;
    done = true;
    if (timer) clearTimeout(timer);
    fs.closeSync(fd);
  };
  res.on("close", stop); // the client went away (a pause, a closed app): nothing more is read for it
  const next = () => {
    if (done) return;
    if (sent >= cutAt) {
      stop();
      if (cutAt < length) {
        // The dropped connection: an orderly FIN short of the Content-Length (so every byte sent has been delivered), then a hard close if the client is slow to notice.
        const socket = res.socket;
        socket?.end();
        setTimeout(() => socket?.destroy(), 1000).unref();
      } else {
        res.end();
      }
      return;
    }
    const n = Math.min(step, cutAt - sent);
    const buf = Buffer.alloc(n);
    fs.readSync(fd, buf, 0, n, start + sent);
    sent += n;
    res.write(buf, () => {
      if (done) return;
      if (opts.pace > 0) timer = setTimeout(next, 50);
      else setImmediate(next);
    });
  };
  next();
});

server.on("clientError", (_e, socket) => socket.destroy());
server.listen(opts.port, "127.0.0.1", () => console.log(`serving ${root} on http://127.0.0.1:${opts.port}/ (log ${logPath})`));
if (opts.exitWithStdin) {
  process.stdin.resume();
  process.stdin.on("close", () => process.exit(0));
  process.stdin.on("end", () => process.exit(0));
}
