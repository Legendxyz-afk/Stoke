"""The updater's test kit (milestone 1 only; the real keys and R2 arrive in milestone 4).

  python scripts/update-feed.py keys            two TEST key pairs in .keys/ (git-ignored); the first
                                                one's public key goes into tauri.conf.json
  python scripts/update-feed.py build <version> a signed NSIS installer of that version, built with
                                                the test overlay (http feed on 127.0.0.1)
  python scripts/update-feed.py feeds           e2e/.work/feed/{good,tampered,wrongkey,nosig} from 0.0.2
  python scripts/update-feed.py serve <feed>    serves one feed on http://127.0.0.1:8765/ until Ctrl+C

Run from desktop/. Uses the project's own Tauri CLI through node, never a global one.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path.cwd()
KEYS = HERE / ".keys"
WORK = HERE / "e2e" / ".work"
CONF = HERE / "src-tauri" / "tauri.conf.json"
OVERLAY = HERE / "src-tauri" / "tauri.updater-test.conf.json"
PASSWORD = "lorren-test"
PORT = 8765
TAURI = [shutil.which("node") or "node", str(HERE / "node_modules" / "@tauri-apps" / "cli" / "tauri.js")]


def run(args, env=None):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run(args, check=True, env=env)


def keys():
    KEYS.mkdir(exist_ok=True)
    for name in ("test", "wrong"):
        if not (KEYS / f"{name}.key").exists():
            run(TAURI + ["signer", "generate", "--ci", "-w", str(KEYS / f"{name}.key"), "-p", PASSWORD])
    pub = (KEYS / "test.key.pub").read_text().strip()
    conf = json.loads(CONF.read_text(encoding="utf-8"))
    conf.setdefault("plugins", {}).setdefault("updater", {})["pubkey"] = pub
    CONF.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    print("test public key written to", CONF)


def build(version):
    env = dict(os.environ)
    # `tauri build` (CLI 2.12) reads the key from TAURI_SIGNING_PRIVATE_KEY, as its text or as a path to the file;
    # TAURI_SIGNING_PRIVATE_KEY_PATH is read only by `tauri signer sign`, and a build given only that one fails
    # with "A public key has been found, but no private key".
    env["TAURI_SIGNING_PRIVATE_KEY"] = str(KEYS / "test.key")
    env["TAURI_SIGNING_PRIVATE_KEY_PASSWORD"] = PASSWORD
    run(TAURI + ["build", "--config", str(OVERLAY), "--config", json.dumps({"version": version})], env=env)
    nsis = HERE / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    exe = nsis / f"Lorren_{version}_x64-setup.exe"
    sig = pathlib.Path(str(exe) + ".sig")
    assert exe.exists() and sig.exists(), f"missing {exe} or its .sig"
    out = WORK / "builds" / version
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exe, out / exe.name)
    shutil.copy2(sig, out / sig.name)
    print("built", out / exe.name)


def write_feed(name, installer, signature, note):
    d = WORK / "feed" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "Lorren_0.0.2_x64-setup.exe").write_bytes(installer)
    latest = {
        "version": "0.0.2",
        "notes": f"Lorren 0.0.2 (test feed: {note})",
        "pub_date": "2026-10-12T00:00:00Z",
        "platforms": {
            "windows-x86_64": {
                "signature": signature,
                "url": f"http://127.0.0.1:{PORT}/Lorren_0.0.2_x64-setup.exe",
            }
        },
    }
    (d / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    print("feed", name, "->", d)


def feeds():
    src = WORK / "builds" / "0.0.2" / "Lorren_0.0.2_x64-setup.exe"
    installer = src.read_bytes()
    good_sig = pathlib.Path(str(src) + ".sig").read_text().strip()
    write_feed("good", installer, good_sig, "good")

    flipped = bytearray(installer)
    flipped[len(flipped) // 2] ^= 0x01
    assert bytes(flipped) != installer
    write_feed("tampered", bytes(flipped), good_sig, "one byte flipped, original signature")

    tmp = WORK / "wrongkey-sign" / src.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp)
    run(TAURI + ["signer", "sign", "-f", str(KEYS / "wrong.key"), "-p", PASSWORD, str(tmp)])
    wrong_sig = pathlib.Path(str(tmp) + ".sig").read_text().strip()
    assert wrong_sig != good_sig
    write_feed("wrongkey", installer, wrong_sig, "signed with a key the app does not trust")

    write_feed("nosig", installer, "", "no signature")


def serve(name):
    d = WORK / "feed" / name
    assert (d / "latest.json").exists(), f"no feed {name}; run `feeds` first"
    print(f"serving {d} on http://127.0.0.1:{PORT}/ (Ctrl+C stops)", flush=True)
    run([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1", "--directory", str(d)])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "keys":
        keys()
    elif cmd == "build" and len(sys.argv) == 3:
        build(sys.argv[2])
    elif cmd == "feeds":
        feeds()
    elif cmd == "serve" and len(sys.argv) == 3:
        serve(sys.argv[2])
    else:
        sys.exit(__doc__)
