"""바탕화면 런처 아이콘 생성 — TV 수상기 화면에 상승 화살표와 %.

  .venv/bin/python scripts/make_icon.py                 # PNG + icns 생성
  .venv/bin/python scripts/make_icon.py --style over    # 배치 바꾸기
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

# 배경 프리셋: (그라디언트 위, 아래, 안테나 색)
# 어두운 바탕화면에서 아이콘 경계가 묻히지 않도록 밝은 배경을 기본으로 둔다.
# 배경이 밝으면 은색 안테나가 사라지므로 안테나 색도 함께 바꾼다.
BG_PRESETS = {
    # 차분한 중간톤 — 바탕화면과도, 은색 수상기와도 대비가 산다
    "slate":    ((124, 138, 160), (84, 97, 119), (238, 241, 248)),
    "graphite": ((104, 108, 120), (66, 70, 82), (234, 237, 244)),
    "denim":    ((74, 98, 138), (46, 63, 94), (228, 236, 248)),
    "sage":     ((150, 168, 156), (112, 131, 120), (242, 246, 242)),
    # 밝은 계열
    "ivory":    ((246, 242, 233), (224, 217, 203), (96, 88, 74)),
    "stone":    ((206, 202, 196), (174, 168, 160), (72, 68, 62)),
    "white":    ((250, 250, 253), (219, 224, 235), (88, 95, 116)),
    # 그 외
    "amber":    ((255, 186, 74), (237, 124, 26), (92, 56, 12)),
    "navy":     ((22, 26, 46), (9, 11, 22), (196, 203, 222)),   # 기존(어두움)
}
BG_TOP, BG_BOT, ANTENNA = BG_PRESETS["slate"]
BODY_TOP, BODY_BOT = (216, 220, 232), (150, 157, 178)   # 수상기 본체 (은색)
SCREEN = (13, 16, 32)                                    # 화면 바탕
AMBER = (255, 196, 75)                                   # 화면 숫자 (앰버 CRT)
BLUE = (96, 165, 250)                                    # 상승선
MARK = BLUE                                              # 합성 기호 단색 (--color 로 변경)

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



def _arrow(d: ImageDraw.ImageDraw, pts, color, width: int, head: int) -> None:
    """상승 꺾은선 + 끝에 화살촉. 예보 느낌을 주면서 '오른다'가 바로 읽힌다."""
    d.line(pts, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    tip = (x1 + math.cos(ang) * head * 0.42, y1 + math.sin(ang) * head * 0.42)
    left = (x1 + math.cos(ang + 2.5) * head, y1 + math.sin(ang + 2.5) * head)
    right = (x1 + math.cos(ang - 2.5) * head, y1 + math.sin(ang - 2.5) * head)
    d.polygon([tip, left, right], fill=color)


def _pct_arrow(d: ImageDraw.ImageDraw, cx: int, cy: int, size: int,
               ring_color, arrow_color) -> None:
    """% 기호의 사선을 우상승 화살표로 바꾼 합성 기호.

    화살표와 % 를 따로 두면 요소가 둘이라 작은 크기에서 서로를 방해한다.
    % 의 사선이 원래 우상향(/)이므로, 그 자리에 화살촉만 얹으면
    '퍼센트'와 '오른다'가 기호 하나로 읽힌다.
    """
    import math

    # 고리를 사선에 바짝 붙여야 % 로 읽힌다. 떨어뜨리면 'O / O' 로 보인다.
    a = size * 0.46                      # 사선 반길이
    t = max(int(size * 0.125), 3)        # 사선 두께
    r = size * 0.215                     # 고리 바깥 반지름
    rt = max(int(size * 0.105), 2)       # 고리 두께 (사선과 비슷하게)
    head = size * 0.26                   # 화살촉 크기
    off = 0.50                           # 중심에서 고리까지 (작을수록 사선에 밀착)

    x0, y0 = cx - a, cy + a              # 왼쪽 아래에서
    x1, y1 = cx + a, cy - a              # 오른쪽 위로
    ang = math.atan2(y1 - y0, x1 - x0)
    # 화살촉이 얹힐 만큼 사선을 조금 줄인다
    ex, ey = x1 - math.cos(ang) * head * 0.52, y1 - math.sin(ang) * head * 0.52
    d.line([(x0, y0), (ex, ey)], fill=arrow_color, width=t)
    d.polygon([(x1, y1),
               (x1 + math.cos(ang + 2.45) * head, y1 + math.sin(ang + 2.45) * head),
               (x1 + math.cos(ang - 2.45) * head, y1 + math.sin(ang - 2.45) * head)],
              fill=arrow_color)

    # 고리 둘 — 왼쪽 위, 오른쪽 아래
    for sx, sy in ((-1, -1), (1, 1)):
        gx, gy = cx + sx * a * off, cy + sy * a * off
        d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=ring_color, width=rt)



def _draw_screen(d: ImageDraw.ImageDraw, sx0: int, sy0: int, sx1: int, sy1: int,
                 style: str) -> None:
    """화면 내용: 상승 화살표와 % 만 쓴다 (숫자 없음).

    32px 에서도 알아봐야 하므로 요소를 최소로 두고 크게 그린다.
    """
    w, h = sx1 - sx0, sy1 - sy0
    cx, cy = (sx0 + sx1) // 2, (sy0 + sy1) // 2

    if style == "glyph":
        # % 의 사선 자체가 화살표 — 기호 하나로 '퍼센트'와 '상승'을 같이 말한다
        c = MARK + (255,)
        _pct_arrow(d, cx, cy, int(min(w, h) * 0.86), c, c)

    elif style == "side":
        # 왼쪽 화살표 · 오른쪽 % — 요소가 겹치지 않아 작은 크기에서도 또렷하다
        ax0, ax1 = sx0 + int(w * 0.13), sx0 + int(w * 0.52)
        pts = [(ax0, cy + int(h * 0.22)),
               (ax0 + (ax1 - ax0) * 0.48, cy + int(h * 0.01)),
               (ax1, cy - int(h * 0.24))]
        _arrow(d, pts, BLUE + (255,), 32, 56)
        f = _fit_font(FONT_BLACK, "%", int(w * 0.32), int(h * 0.56))
        bb = f.getbbox("%")
        d.text((sx0 + int(w * 0.68) - bb[0], cy - (bb[3] - bb[1]) // 2 - bb[1]),
               "%", font=f, fill=AMBER + (255,))

    else:
        # % 를 크게 두고 화살표가 그 위를 가로질러 오른다
        f = _fit_font(FONT_BLACK, "%", int(w * 0.52), int(h * 0.74))
        bb = f.getbbox("%")
        d.text((cx - (bb[2] - bb[0]) // 2 - bb[0], cy - (bb[3] - bb[1]) // 2 - bb[1]),
               "%", font=f, fill=AMBER + (255,))
        pts = [(sx0 + int(w * 0.08), sy1 - int(h * 0.18)),
               (sx0 + int(w * 0.38), sy1 - int(h * 0.34)),
               (sx0 + int(w * 0.62), sy1 - int(h * 0.30)),
               (sx1 - int(w * 0.08), sy0 + int(h * 0.16))]
        _arrow(d, pts, BLUE + (255,), 30, 52)


def build(style: str = "glyph") -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # --- 배경: macOS 스타일 둥근 사각형 ---
    bg = _vgrad((S, S), BG_TOP, BG_BOT).convert("RGBA")
    bg.putalpha(_rounded_mask((S, S), 200))   # 바탕화면 런처 공통 모서리 반경
    img.alpha_composite(bg)

    d = ImageDraw.Draw(img, "RGBA")

    # --- 안테나: 본체보다 먼저 그려 뒤로 보내기 ---
    ax, ay = S // 2, 300
    for dx in (-190, 190):
        d.line([(ax, ay), (ax + dx, 108)], fill=ANTENNA + (255,), width=17)
        d.ellipse([ax + dx - 21, 108 - 21, ax + dx + 21, 108 + 21],
                  fill=ANTENNA + (255,))

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

    _draw_screen(d, sx0, sy0, sx1, sy1, style)

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
    ap.add_argument("--style", default="glyph", choices=["glyph", "side", "over"],
                    help="glyph=%% 사선이 화살표(기본), side=나란히, over=겹침")
    ap.add_argument("--color", default="blue",
                    choices=["amber", "blue", "white", "green"],
                    help="합성 기호 색 (style=glyph 일 때)")
    ap.add_argument("--bg", default="slate", choices=list(BG_PRESETS),
                    help="배경색 (어두운 바탕화면과 구분되도록 기본은 amber)")
    ap.add_argument("--preview", action="store_true", help="512 미리보기만 저장")
    args = ap.parse_args()

    global MARK, BG_TOP, BG_BOT, ANTENNA
    MARK = {"amber": AMBER, "blue": BLUE,
            "white": (238, 242, 252), "green": (74, 222, 128)}[args.color]
    BG_TOP, BG_BOT, ANTENNA = BG_PRESETS[args.bg]

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img = build(args.style)
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
