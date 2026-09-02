"""파이프라인 스모크 테스트용 합성 시청률 데이터 생성기.

⚠️ 실제 데이터가 아니다. 원본 `종편_4사_메인_시청률.csv` 를 data/ 에 넣으면 그것이 쓰인다.
대화에서 확인된 실제 데이터의 통계적 성질만 재현한다:
  - 2023-07-01 ~ 2026-08-27, 중간에 결측일 7일
  - 주말 시청률이 평일 대비 40~50% 급락
  - 2024-12 초 대형 정치 이슈로 평시의 2~3배 폭등
  - 2049 지표는 JTBC가 타사의 2.5배 수준
  - KBO 야구 중계가 19시대 뉴스를 잠식
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ratings import kbo                                    # noqa: E402
from ratings.config import CHANNELS, DATA_DIR              # noqa: E402

RNG = np.random.default_rng(20260829)

# (평일 기준선, 주말 배율, 2049 비중, KBO 민감도)
PROFILE = {
    "channelA":  dict(base=1.95, weekend=0.55, youth=0.135, kbo=0.16),
    "jtbc":      dict(base=2.85, weekend=0.52, youth=0.230, kbo=0.13),
    "mbn":       dict(base=1.85, weekend=0.58, youth=0.095, kbo=0.15),
    "tvchosun":  dict(base=2.55, weekend=0.60, youth=0.060, kbo=0.05),
}
MISSING = ["2024-04-10", "2024-12-25", "2025-01-01", "2025-03-01",
           "2025-06-06", "2025-10-06", "2026-05-05"]


def build() -> pd.DataFrame:
    dates = pd.date_range("2023-07-01", "2026-08-27", freq="D")
    n = len(dates)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    t = np.arange(n) / n

    exo = kbo.load_daily(pd.Series(dates)).set_index("Date")
    kbo_press = exo["kbo_overlap_19"].to_numpy() * (1 + 0.4 * exo["kbo_seoul_games"].to_numpy())
    kbo_press21 = exo["kbo_overlap_21"].to_numpy()

    # 2024-12 비상 국면: 급등 후 지수 감쇠
    event = np.zeros(n)
    spike_start = np.searchsorted(dates, pd.Timestamp("2024-12-03"))
    ramp = np.exp(-np.arange(n - spike_start) / 22.0)
    event[spike_start:] = 1.9 * ramp
    event += 0.35 * np.exp(-((np.arange(n) - np.searchsorted(dates, pd.Timestamp("2025-04-04"))) ** 2) / 120)

    season = 0.12 * np.cos(2 * np.pi * (doy - 20) / 365.25)      # 겨울 ↑ 여름 ↓
    trend = -0.28 * t                                            # 전반적 시청률 하락
    holiday = np.isin(dates.strftime("%m-%d"), ["01-01", "12-25", "08-15", "10-03"]) * -0.25

    out = {"날짜": dates.strftime("%y%m%d")}
    shared = RNG.normal(0, 0.09, n)                              # 4사 공통 뉴스 수요 충격
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.55 * ar[i - 1] + shared[i]

    for ch in CHANNELS:
        p = PROFILE[ch.key]
        weekend_mult = np.where(dow >= 5, p["weekend"], 1.0)
        slots = np.array([ch.at(d).slot for d in dates])
        press = np.where(slots == 19, kbo_press, kbo_press21)

        level = (p["base"] * (1 + season + trend) * weekend_mult
                 + holiday
                 - p["kbo"] * press
                 + event * p["base"] / 2.6
                 + ar * p["base"] / 2.6
                 + RNG.normal(0, 0.11, n))
        level = np.clip(level, 0.25, None)
        out[ch.household_col] = np.round(level, 3)

        youth = (level * p["youth"]
                 * (1 + 0.25 * np.where(dow >= 5, 1, 0))
                 + RNG.normal(0, 0.02, n))
        out[ch.col_2049] = np.round(np.clip(youth, 0.02, None), 3)

    df = pd.DataFrame(out)
    df = df.loc[~pd.DatetimeIndex(dates).isin(pd.to_datetime(MISSING))]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "SAMPLE_종편_4사_메인_시청률.csv"
    df = build()
    df.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"생성: {dest}  ({len(df)}행)")
    print(df.head(3).to_string(index=False))
