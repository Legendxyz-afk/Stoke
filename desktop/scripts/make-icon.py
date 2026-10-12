"""Writes the 1024x1024 source icon for `tauri icon`: an amber square 'l' on the console's ground."""
from PIL import Image, ImageDraw

SIZE = 1024
GROUND = (9, 11, 14)      # oklch(0.15 0.006 255)
ACCENT = (214, 167, 113)  # oklch(0.76 0.09 70)
img = Image.new("RGB", (SIZE, SIZE), GROUND)
d = ImageDraw.Draw(img)
d.rectangle([96, 96, SIZE - 96, SIZE - 96], outline=ACCENT, width=40)
d.rectangle([420, 250, 540, 700], fill=ACCENT)   # the stem of the 'l'
d.rectangle([420, 640, 640, 760], fill=ACCENT)   # its foot
img.save("scripts/icon-1024.png")
print("wrote scripts/icon-1024.png")
