#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
ICONSET_DIR = ASSETS_DIR / "Talksribe.iconset"
ICNS_PATH = ASSETS_DIR / "Talksribe.icns"
MASTER_PNG_PATH = ASSETS_DIR / "Talksribe.png"

ICONSET_SIZES = {
    "icon_16x16.png": 16,
    "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32,
    "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128,
    "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256,
    "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512,
    "icon_512x512@2x.png": 1024,
}


def _rounded_gradient(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            nx = x / max(size - 1, 1)
            ny = y / max(size - 1, 1)
            r = int(250 - 60 * ny - 12 * nx)
            g = int(164 - 30 * ny + 8 * nx)
            b = int(88 + 36 * nx)
            pixels[x, y] = (r, g, b, 255)

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=int(size * 0.23),
        fill=255,
    )
    image.putalpha(mask)
    return image


def _draw_wave(draw: ImageDraw.ImageDraw, size: int) -> None:
    bar_w = size * 0.055
    gap = size * 0.028
    base_x = size * 0.24
    center_y = size * 0.53
    heights = [0.17, 0.27, 0.38, 0.28, 0.17]
    for idx, scale in enumerate(heights):
        x0 = base_x + idx * (bar_w + gap)
        h = size * scale
        y0 = center_y - h / 2
        y1 = center_y + h / 2
        draw.rounded_rectangle(
            (x0, y0, x0 + bar_w, y1),
            radius=bar_w / 2,
            fill=(255, 247, 238, 255),
        )


def _draw_caption(draw: ImageDraw.ImageDraw, size: int) -> None:
    line_w = size * 0.17
    line_h = size * 0.028
    x = size * 0.61
    top = size * 0.39
    gap = size * 0.058
    widths = [line_w, line_w * 0.84, line_w * 0.62]
    for idx, width in enumerate(widths):
        y0 = top + idx * gap
        draw.rounded_rectangle(
            (x, y0, x + width, y0 + line_h),
            radius=line_h / 2,
            fill=(55, 33, 20, 230),
        )


def render_master_icon(size: int = 1024) -> Image.Image:
    icon = _rounded_gradient(size)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse(
        (
            size * 0.18,
            size * 0.18,
            size * 0.58,
            size * 0.82,
        ),
        fill=(28, 17, 10, 90),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=size * 0.02))
    icon = Image.alpha_composite(icon, shadow)

    draw = ImageDraw.Draw(icon)
    draw.ellipse(
        (
            size * 0.16,
            size * 0.16,
            size * 0.56,
            size * 0.80,
        ),
        fill=(255, 244, 236, 255),
    )
    draw.ellipse(
        (
            size * 0.22,
            size * 0.22,
            size * 0.50,
            size * 0.74,
        ),
        fill=(248, 124, 66, 255),
    )

    _draw_wave(draw, size)
    _draw_caption(draw, size)

    ring_box = (
        size * 0.16,
        size * 0.16,
        size * 0.56,
        size * 0.80,
    )
    draw.arc(ring_box, start=235, end=305, fill=(255, 252, 247, 180), width=int(size * 0.02))
    draw.arc(ring_box, start=55, end=125, fill=(255, 252, 247, 180), width=int(size * 0.02))

    highlight = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    highlight_draw.ellipse(
        (
            size * 0.10,
            size * 0.05,
            size * 0.90,
            size * 0.48,
        ),
        fill=(255, 255, 255, 38),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=size * 0.05))
    icon = Image.alpha_composite(icon, highlight)
    return icon


def write_icon_assets() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    if ICONSET_DIR.exists():
        for path in ICONSET_DIR.iterdir():
            path.unlink()
    else:
        ICONSET_DIR.mkdir()

    master = render_master_icon(1024)
    master.save(MASTER_PNG_PATH)

    for filename, size in ICONSET_SIZES.items():
        resized = master.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(ICONSET_DIR / filename)

    master.save(ICNS_PATH, format="ICNS", sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)])


if __name__ == "__main__":
    write_icon_assets()
