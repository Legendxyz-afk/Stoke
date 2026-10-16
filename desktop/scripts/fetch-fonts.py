"""Downloads the design's fonts (OFL) into src/fonts: Archivo 400/500/600 and JetBrains Mono
400/500 as the static TrueType files Google Fonts serves, plus each family's OFL text."""
import pathlib
import re
import urllib.request

OUT = pathlib.Path("src/fonts")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "curl/8.0"}  # a non-browser agent is served TrueType URLs


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


for family, stem, weights in (("Archivo", "Archivo", (400, 500, 600)), ("JetBrains+Mono", "JetBrainsMono", (400, 500))):
    css = get(f"https://fonts.googleapis.com/css2?family={family}:wght@{';'.join(map(str, weights))}").decode()
    found = re.findall(r"font-weight:\s*(\d+);.*?url\((https://[^)]+\.ttf)\)", css, re.S)
    assert len(found) == len(weights), (family, found)
    for w, url in found:
        (OUT / f"{stem}-{w}.ttf").write_bytes(get(url))
        print("font", stem, w)
for name, path in (("Archivo", "ofl/archivo/OFL.txt"), ("JetBrainsMono", "ofl/jetbrainsmono/OFL.txt")):
    (OUT / f"OFL-{name}.txt").write_bytes(get(f"https://raw.githubusercontent.com/google/fonts/main/{path}"))
    print("licence", name)
