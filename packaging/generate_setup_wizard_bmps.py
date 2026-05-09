#!/usr/bin/env python3
"""Generate Inno Setup wizard BMPs in IKEA-style blue + yellow (brand-inspired palette)."""
from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Requires Pillow: pip install Pillow") from e

# IKEA-inspired (approximate): blue #0058AB, yellow #FFCC00
BLUE = (0x00, 0x58, 0xAB)
YELLOW = (0xFF, 0xCC, 0x00)


def _lerp(a: float, b: float, t: float) -> int:
    return int(round(a + (b - a) * t))


def _blend_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (_lerp(c0[0], c1[0], t), _lerp(c0[1], c1[1], t), _lerp(c0[2], c1[2], t))


def main() -> None:
    root = Path(__file__).resolve().parent

    # Large side panel — vertical gradient (blue top → yellow bottom), Inno modern default 164×314
    w, h = 164, 314
    side = Image.new("RGB", (w, h))
    pix = side.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        rgb = _blend_rgb(BLUE, YELLOW, t)
        for x in range(w):
            pix[x, y] = rgb
    side.save(root / "setup_wizard.bmp", format="BMP")

    # Small header — blue field + yellow band at bottom (IKEA stripe feel), 55×55
    sw, sh = 55, 55
    small = Image.new("RGB", (sw, sh), BLUE)
    band_h = 14
    spix = small.load()
    for y in range(sh - band_h, sh):
        for x in range(sw):
            spix[x, y] = YELLOW
    small.save(root / "setup_wizard_small.bmp", format="BMP")

    print(f"Wrote {root / 'setup_wizard.bmp'} and {root / 'setup_wizard_small.bmp'}")


if __name__ == "__main__":
    main()
