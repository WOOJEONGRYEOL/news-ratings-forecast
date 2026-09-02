"""시청률이 지속적으로 튄 구간을 찾아 events.csv 후보를 만든다.

자동 탐지는 '이 날들이 특이했다'까지만 말해 준다. 무슨 사건이었는지는 사람이 채워야
모델이 미래 시나리오로 쓸 수 있다. label 을 직접 수정한 뒤 data/events.csv 로 저장하라.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ratings import data as rdata                        # noqa: E402
from ratings.config import DATA_DIR, HOUSEHOLD_COLS      # noqa: E402

MIN_LEN = 2         # 이 일수 이상 이어진 구간만 이벤트로 본다
GAP = 5             # 이 일수 이내로 떨어진 구간은 하나로 합친다
LIFT = 1.45         # 기준선 대비 이 배율 이상이면 급등일
BASE_WEEKS = 13     # 같은 요일 기준선 창(주)


def _lift(df: pd.DataFrame, col: str) -> pd.Series:
    """**같은 요일끼리** 중심 중앙값을 잡고 그 대비 배율을 낸다.

    기준선 선택에 세 번 실패했다.
      1. 후행 28일 중앙값 — 수 주 이어지는 국면에서 기준선이 같이 올라가
         국면 중간의 사소한 요철만 골라낸다 (2025-07 오탐).
      2. 후행 30~120일 중앙값 — 시청률이 3년간 절반으로 떨어지는 추세 탓에
         과거가 체계적으로 높아, 국면이 아니라 수준 변화를 잡는다 (128~166일 구간).
      3. 전 기간 요일평균 + 중심 중앙값 — 2026-01-10 채널A 편성 변경으로
         TV조선 주말이 오른 것을 이벤트로 오인해 토요일이 무더기로 잡힌다.

    같은 요일끼리 국소 중앙값을 쓰면 추세·요일효과·편성변경이 모두 흡수되고,
    국면이 창의 절반 미만이면 중앙값이 거의 움직이지 않아 오염되지도 않는다.
    """
    s = df.set_index("Date")[col]
    out = pd.Series(index=s.index, dtype=float)
    for _, g in s.groupby(s.index.dayofweek):
        base = g.rolling(BASE_WEEKS, center=True,
                         min_periods=max(3, BASE_WEEKS // 3)).median()
        out.loc[g.index] = (g / base).to_numpy()
    return out


def main() -> None:
    df = rdata.load_ratings()

    lifts = pd.DataFrame({c: _lift(df, c) for c in HOUSEHOLD_COLS})
    hot = (lifts > LIFT).any(axis=1)          # 한 채널만 튀어도 뉴스 국면일 수 있다

    dates = list(lifts.index[hot.fillna(False)])
    if not dates:
        print("탐지된 구간이 없습니다.")
        return

    groups, cur = [], [dates[0]]
    for d in dates[1:]:
        if (d - cur[-1]).days <= GAP:
            cur.append(d)
        else:
            groups.append(cur)
            cur = [d]
    groups.append(cur)

    idx = df.set_index("Date")
    rows = []
    for g in groups:
        start, end = g[0], g[-1]
        if (end - start).days + 1 < MIN_LEN:
            continue
        ratio = {c: float(lifts.loc[start:end, c].mean()) for c in HOUSEHOLD_COLS}
        lead = max(ratio, key=ratio.get)
        rows.append({
            "start": start.date(), "end": end.date(),
            "label": f"미확인 이슈 {start.date()}",     # ← 사람이 채울 자리
            "type": "",                                # ← 사건 유형도 사람이 채운다
            "weight": 1.0,
            "일수": (end - start).days + 1,
            "주도채널": lead,
            "주도채널배율": round(ratio[lead], 2),
            "4사동반": "○" if sum(v > 1.15 for v in ratio.values()) >= 3 else "",
        })

    out = pd.DataFrame(rows).sort_values("주도채널배율", ascending=False)
    print(out.to_string(index=False))
    dest = DATA_DIR / "events_candidates.csv"
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"\n후보 저장 → {dest}")
    print("label 과 type 을 채우고 필요한 행만 남긴 뒤 data/events.csv 로 저장하세요.")
    print("type 은 features.EVENT_TYPES 중 하나여야 모델이 별도 변수로 학습합니다.")


if __name__ == "__main__":
    main()
