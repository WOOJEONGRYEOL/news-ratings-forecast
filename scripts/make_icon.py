"""바탕화면 런처 아이콘 생성 — TV 수상기 화면에 시청률 %.

  .venv/bin/python scripts/make_icon.py            # PNG + icns 생성
  .venv/bin/python scripts/make_icon.py --preview  # 미리보기 PNG 만 (512)

macOS 아이콘은 32px 에서도 알아봐야 하므로, 형태(수상기 실루엣)와
화면 숫자의 대비를 최우선으로 둔다. 잔무늬는 큰 사이즈에서만 보이게 약하게.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "assets" / "icon_1024.png"
OUT_ICNS = ROOT / "assets" / "AppIcon.icns"

S = 1024                                   # 마스터 크기
BG_TOP, BG_BOT = (22, 26, 46), (9, 11, 22)         # 배경 그라디언트 (짙은 남색)
BODY_TOP, BODY_BOT = (216, 220, 232), (150, 157, 178)   # 수상기 본체 (은색)
SCREEN = (13, 16, 32)                                    # 화면 바탕
AMBER = (255, 196, 75)                                   # 화면 숫자 (앰버 CRT)
BLUE = (96, 165, 250)                                    # 상승선

FONT_BLACK = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def _vgrad(size: tuple[int, int], top: tuple, bot: tuple) -> Image.Image:
    """세로 그라디언트."""
    w, h = size
    g = Image.new("RGB", (1, h))
    px = g.load()
    for y in range(h):
        t = y / max(h - 1, 1)
        px[0, y] = tuple(round(a + (b - a) * t) for a, b in zip(top, bot))
    return g.resize((w, h), Image.BILINEAR)


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def _fit_font(path: str, text: str, target_w: int, target_h: int) -> ImageFont.FreeTypeFont:
    """폭·높이 안에 들어가는 최대 크기 폰트."""
    lo, hi = 10, 600
    best = ImageFont.truetype(path, lo)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(path, mid)
        box = f.getbbox(text)
        if box[2] - box[0] <= target_w and box[3] - box[1] <= target_h:
            best, lo = f, mid + 1
        else:
            hi = mid - 1
    return best


def build(value: str = "3.2") -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # --- 배경: macOS 스타일 둥근 사각형 ---
    bg = _vgrad((S, S), BG_TOP, BG_BOT).convert("RGBA")
    bg.putalpha(_rounded_mask((S, S), 230))
    img.alpha_composite(bg)

    d = ImageDraw.Draw(img, "RGBA")

    # --- 안테나: 본체보다 먼저 그려 뒤로 보내기 ---
    ax, ay = S // 2, 300
    for dx in (-190, 190):
        d.line([(ax, ay), (ax + dx, 108)], fill=(196, 203, 222, 255), width=17)
        d.ellipse([ax + dx - 21, 108 - 21, ax + dx + 21, 108 + 21],
                  fill=(226, 231, 244, 255))

    # --- 수상기 본체 ---
    bx0, by0, bx1, by1 = 128, 286, 896, 806
    body = _vgrad((bx1 - bx0, by1 - by0), BODY_TOP, BODY_BOT).convert("RGBA")
    body.putalpha(_rounded_mask((bx1 - bx0, by1 - by0), 76))
    img.alpha_composite(body, (bx0, by0))
    # 윗면 하이라이트 (입체감)
    d.rounded_rectangle([bx0 + 8, by0 + 6, bx1 - 8, by0 + 150], radius=62,
                        fill=(255, 255, 255, 34))

    # --- 다리 ---
    for cx in (300, 724):
        d.polygon([(cx - 52, by1 - 6), (cx + 52, by1 - 6),
                   (cx + 34, by1 + 74), (cx - 34, by1 + 74)],
                  fill=(150, 157, 178, 255))

    # --- 화면 ---
    sx0, sy0, sx1, sy1 = 186, 344, 760, 748
    d.rounded_rectangle([sx0 - 9, sy0 - 9, sx1 + 9, sy1 + 9], radius=56,
                        fill=(96, 101, 122, 255))          # 베젤 안쪽 그림자
    d.rounded_rectangle([sx0, sy0, sx1, sy1], radius=48, fill=SCREEN + (255,))

    # 숫자는 위, 상승선은 아래 — 겹치면 둘 다 안 읽힌다
    num_f = _fit_font(FONT_BLACK, value, 336, 172)
    pct_f = ImageFont.truetype(FONT_BLACK, int(num_f.size * 0.62))
    nb, pb = num_f.getbbox(value), pct_f.getbbox("%")
    gap = 20
    total_w = (nb[2] - nb[0]) + gap + (pb[2] - pb[0])
    cx = (sx0 + sx1) // 2
    top_y = sy0 + 76
    nx = cx - total_w // 2
    d.text((nx - nb[0], top_y - nb[1]), value, font=num_f, fill=AMBER + (255,))
    d.text((nx + (nb[2] - nb[0]) + gap - pb[0],
            top_y + (nb[3] - nb[1]) - (pb[3] - pb[1]) - pb[1]),
           "%", font=pct_f, fill=AMBER + (255,))

    # 상승선(예측 느낌)은 화면 아래쪽 띠에만 둔다
    ly0, ly1 = sy1 - 118, sy1 - 40
    pts = [(sx0 + 60, ly1), (sx0 + 190, ly1 - 28), (sx0 + 318, ly1 - 14),
           (sx0 + 446, ly0 + 18), (sx1 - 60, ly0)]
    d.line(pts, fill=BLUE + (235,), width=16, joint="curve")
    for q in pts:
        d.ellipse([q[0] - 12, q[1] - 12, q[0] + 12, q[1] + 12], fill=BLUE + (255,))

    # 화면 광택 (왼쪽 위에서 비스듬히)
    gloss = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).polygon(
        [(sx0, sy0), (sx0 + 300, sy0), (sx0, sy0 + 250)], fill=(255, 255, 255, 26))
    gloss = gloss.filter(ImageFilter.GaussianBlur(10))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([sx0, sy0, sx1, sy1], radius=48, fill=255)
    img.alpha_composite(Image.composite(gloss, Image.new("RGBA", (S, S), (0, 0, 0, 0)), mask))

    # --- 오른쪽 조작부 (수상기 티 내기) ---
    kx = 828
    for i, ky in enumerate((430, 560)):
        d.ellipse([kx - 34, ky - 34, kx + 34, ky + 34], fill=(120, 127, 148, 255))
        d.ellipse([kx - 21, ky - 21, kx + 21, ky + 21], fill=(206, 212, 228, 255))
    d.rounded_rectangle([kx - 30, 640, kx + 30, 712], radius=16, fill=(120, 127, 148, 255))

    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="3.2", help="화면에 띄울 시청률 숫자")
    ap.add_argument("--preview", action="store_true", help="512 미리보기만 저장")
    args = ap.parse_args()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img = build(args.value)
    img.save(OUT_PNG)
    print(f"PNG: {OUT_PNG}")

    if args.preview:
        p = OUT_PNG.with_name("icon_preview.png")
        img.resize((512, 512), Image.LANCZOS).save(p)
        print(f"미리보기: {p}")
        return

    iconset = OUT_PNG.parent / "AppIcon.iconset"
    iconset.mkdir(exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512, 1024):
        img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        if size <= 512:
            img.resize((size * 2, size * 2), Image.LANCZOS).save(
                iconset / f"icon_{size}x{size}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT_ICNS)], check=True)
    print(f"ICNS: {OUT_ICNS}")


if __name__ == "__main__":
    main()
