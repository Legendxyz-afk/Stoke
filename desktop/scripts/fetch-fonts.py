"""Downloads the design's fonts (OFL) into src/fonts: Archivo 400/500/600 and JetBrains Mono
400/500 as the static TrueType files Google Fonts serves, plus each family's OFL text.

The target is anchored on this file (desktop/src/fonts), so it does not matter where the script is started
from: a relative path would create src/fonts in the current directory, and from the repo root that is the
frozen engine tree. Every download is checked before anything is written: a font must start with the
TrueType magic and a licence must carry the OFL's name, so an HTML error page is never saved as either."""
import pathlib
import re
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "fonts"
UA = {"User-Agent": "curl/8.0"}  # a non-browser agent is served TrueType URLs
TTF_MAGIC = bytes([0, 1, 0, 0])
OFL_NAME = b"SIL OPEN FONT LICENSE"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def check_ttf(data, what):
    assert data[:4] == TTF_MAGIC, (what, "is not a TrueType font", data[:16])
    return data


def check_ofl(data, what):
    assert OFL_NAME in data, (what, "is not the OFL text", data[:60])
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for family, stem, weights in (("Archivo", "Archivo", (400, 500, 600)), ("JetBrains+Mono", "JetBrainsMono", (400, 500))):
        css = get(f"https://fonts.googleapis.com/css2?family={family}:wght@{';'.join(map(str, weights))}").decode()
        found = re.findall(r"font-weight:\s*(\d+);.*?url\((https://[^)]+\.ttf)\)", css, re.S)
        assert len(found) == len(weights), (family, found)
        for w, url in found:
            (OUT / f"{stem}-{w}.ttf").write_bytes(check_ttf(get(url), url))
            print("font", stem, w)
    for name, path in (("Archivo", "ofl/archivo/OFL.txt"), ("JetBrainsMono", "ofl/jetbrainsmono/OFL.txt")):
        (OUT / f"OFL-{name}.txt").write_bytes(check_ofl(get(f"https://raw.githubusercontent.com/google/fonts/main/{path}"), name))
        print("licence", name)


if __name__ == "__main__":
    main()
