"""미래 시청률 예측 및 What-If 시나리오 적용.

지평별 직접 예측이므로 D+1 예측값을 D+2 입력으로 되먹이지 않는다.
D+h 행의 자기회귀 피처는 전부 shift(h) 이상이라 마지막 관측일까지의 실측값만 참조한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    CHANNELS, DEFAULT_HORIZON, TARGET_COLS, channel_of, is_2049, overlap_matrix,
)
from .features import build_features
from .model import TargetModel


@dataclass
class Scenario:
    """What-If 조작 노브. 기본값은 '있는 그대로의 편성/달력'."""

    kbo: str = "auto"                 # auto | none | weekday_1830 | weekend_1700
    temp_offset: float = 0.0          # 평년 대비 기온 (°C)
    news_event: str = ""              # 속보 국면 유형 ("" = 평시). features.EVENT_TYPES 참고
    cancelled_games: int | None = None  # 당일 확정된 우천취소 경기 수 (None=미확정)
    start_time: float | None = None     # 당일 실제 경기 시작 시각 (None=편성표대로)
    extra: dict = field(default_factory=dict)   # 개별 피처 직접 덮어쓰기

    @property
    def is_default(self) -> bool:
        return (self.kbo == "auto" and self.temp_offset == 0.0
                and not self.news_event and self.cancelled_games is None
                and self.start_time is None and not self.extra)


EVENT_BASE_WEEKS = 8      # 국면 직전 같은 요일 몇 주를 '평시'로 볼 것인가


def event_multipliers(df: pd.DataFrame, events: pd.DataFrame | None,
                      detail: bool = False):
    """과거 국면에서 **실제로 관측된** 채널별 배율.

    기준선은 **국면 직전 8주의 같은 요일 중앙값**이다. 전 기간 요일평균을 쓰면
    안 된다 — 2025-01 구속영장 국면은 이미 탄핵 정국으로 높은 수준 위에서 일어나
    전 기간 평균 대비로는 2.78배가 되지만, 직전 수준 대비로는 그보다 훨씬 작다.
    이 배율은 **모델 예측값에 곱하는 값**이고 모델 예측은 최근 수준을 반영하므로,
    '직전 대비 얼마나 뛰었나'가 맞는 기준이다.

    모델이 학습하지 못하는 값이다 — 유형당 사례가 2~16일뿐이라 트리가 split 조차
    못 한다. 그래서 모델 밖에서 곱하는 방식으로 시나리오를 구현한다.

    detail=True 면 구간별 배율까지 함께 돌려준다(범위 표시용).
    """
    if events is None or events.empty or "type" not in events.columns:
        return ({}, {}) if detail else {}

    hist = df.set_index("Date")
    out: dict = {}
    per_window: dict = {}

    for etype, grp in events.groupby("type"):
        etype = str(etype).strip()
        if not etype:
            continue
        ratios: dict[str, list[float]] = {}
        for _, row in grp.iterrows():
            start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
            pre_lo = start - pd.Timedelta(weeks=EVENT_BASE_WEEKS)
            for col in TARGET_COLS:
                seg = hist.loc[start:end, col]
                pre = hist.loc[pre_lo:start - pd.Timedelta(days=1), col]
                if seg.empty or pre.empty:
                    continue
                # 같은 요일끼리 비교해야 주말 급락이 섞이지 않는다
                pre_by_dow = pre.groupby(pre.index.dayofweek).median()
                base = seg.index.dayofweek.map(pre_by_dow)
                vals = (seg.to_numpy() / np.asarray(base, dtype=float))
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    ratios.setdefault(col, []).append(float(np.mean(vals)))
                    per_window.setdefault((etype, col), []).append(
                        (str(row["start"]), float(np.mean(vals))))
        if ratios:
            out[etype] = {c: float(np.mean(v)) for c, v in ratios.items()}

    return (out, per_window) if detail else out


def extend_dates(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """미래 지평만큼 빈 행을 덧붙인 프레임 (타깃은 NaN)."""
    last = df["Date"].max()
    future = pd.DataFrame({
        "Date": pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D")
    })
    for col in TARGET_COLS:
        future[col] = np.nan
    if "is_imputed" in df.columns:
        future["is_imputed"] = True
    return pd.concat([df, future], ignore_index=True)


def apply_scenario(frame: pd.DataFrame, mask: pd.Series, sc: Scenario) -> pd.DataFrame:
    """미래 행(mask)에만 시나리오를 반영한 피처 프레임을 돌려준다."""
    if sc.is_default:
        return frame

    out = frame.copy()
    idx = out.index[mask]

    if sc.kbo == "none":
        for c, v in {"kbo_games": 0.0, "kbo_any_game": 0.0, "kbo_seoul_games": 0.0,
                     "kbo_first_start": -1.0, "kbo_main_start": -1.0,
                     "kbo_games_19": 0.0, "kbo_games_21": 0.0,
                     "kbo_overlap_19": 0.0, "kbo_overlap_21": 0.0,
                     "kbo_pressure_19": 0.0, "kbo_pressure_21": 0.0}.items():
            if c in out.columns:
                out.loc[idx, c] = v
        if "kbo_cancelled" in out.columns:
            out.loc[idx, "kbo_cancelled"] = 5.0        # 전 경기 우천취소
        # 우천취소 시나리오면 비도 함께 온 것으로 둔다
        for col, val in (("wx_precip_evening", 5.0), ("precip_evening_log", np.log1p(5.0)),
                         ("is_rainy_evening", 1.0), ("kbo_rain_risk", 0.0)):
            if col in out.columns:
                out.loc[idx, col] = val
    elif sc.kbo in {"weekday_1830", "weekend_1700"}:
        start = 18.5 if sc.kbo == "weekday_1830" else 17.0
        ov19 = max(0.0, min(start + 3.25, 20.0) - max(start, 19.0))
        ov21 = max(0.0, min(start + 3.25, 22.0) - max(start, 21.0))
        for c, v in {"kbo_games": 5.0, "kbo_any_game": 1.0, "kbo_seoul_games": 1.0,
                     "kbo_cancelled": 0.0, "kbo_first_start": start, "kbo_main_start": start,
                     "kbo_games_19": 5.0 if ov19 > 0 else 0.0,
                     "kbo_games_21": 5.0 if ov21 > 0 else 0.0,
                     "kbo_overlap_19": ov19, "kbo_overlap_21": ov21,
                     "kbo_pressure_19": ov19 * 1.5, "kbo_pressure_21": ov21 * 1.5}.items():
            if c in out.columns:
                out.loc[idx, c] = v

    if sc.temp_offset:
        # temp_effective 가 실제 모델 입력이다. temp_proxy 만 바꾸면 실측/예보가
        # 있는 날엔 아무 일도 일어나지 않는다.
        for col in ("temp_proxy", "wx_temp", "temp_effective"):
            if col in out.columns:
                out.loc[idx, col] = out.loc[idx, col] + sc.temp_offset
        if "cold_and_dark" in out.columns and "temp_effective" in out.columns:
            out.loc[idx, "cold_and_dark"] = (
                (20.0 - out.loc[idx, "temp_effective"])
                * (20.0 - out.loc[idx, "sunset_hour"])
            )

    if sc.cancelled_games is not None and "kbo_cancelled" in out.columns:
        # 당일 취소가 확정됐다면 그만큼 편성에서 빼고 취소 수를 채운다
        n = float(sc.cancelled_games)
        out.loc[idx, "kbo_cancelled"] = n
        for c in ("kbo_games", "kbo_games_19", "kbo_games_21"):
            if c in out.columns:
                out.loc[idx, c] = (out.loc[idx, c] - n).clip(lower=0)
        if "kbo_any_game" in out.columns:
            out.loc[idx, "kbo_any_game"] = (out.loc[idx, "kbo_games"] > 0).astype(float)

    if sc.start_time is not None:
        # 실시간으로 확인한 실제 시작 시각. 순연·더블헤더로 편성표와 달라질 수 있다.
        from . import kbo as _kbo
        st = float(sc.start_time)
        ov19 = _kbo._overlap(st, _kbo.NEWS_WINDOW_19) if st > 0 else 0.0
        ov21 = _kbo._overlap(st, _kbo.NEWS_WINDOW_21) if st > 0 else 0.0
        for c, v in (("kbo_main_start", st), ("kbo_first_start", st),
                     ("kbo_overlap_19", ov19), ("kbo_overlap_21", ov21)):
            if c in out.columns:
                out.loc[idx, c] = v
        if "kbo_seoul_games" in out.columns:
            seoul = out.loc[idx, "kbo_seoul_games"]
            for c, ov in (("kbo_pressure_19", ov19), ("kbo_pressure_21", ov21)):
                if c in out.columns:
                    out.loc[idx, c] = ov * (1.0 + 0.5 * seoul)
        # 채널별 겹침도 새 시작 시각으로 다시 계산
        for ch in CHANNELS:
            col = f"{ch.key}_kbo_overlap"
            if col not in out.columns or f"{ch.key}_air_hour" not in out.columns:
                continue
            a0 = out.loc[idx, f"{ch.key}_air_hour"]
            a1 = out.loc[idx, f"{ch.key}_air_end"] if f"{ch.key}_air_end" in out.columns \
                else a0 + 1.0
            lap = (np.minimum(st + _kbo.GAME_HOURS, a1) - np.maximum(st, a0)).clip(lower=0.0)
            out.loc[idx, col] = np.where(st > 0, lap / (a1 - a0).replace(0, np.nan), 0.0)

    if sc.news_event:
        # 강도 슬라이더는 두지 않는다. 학습 데이터에서 이 변수는 0 아니면 1 뿐이라
        # 트리 모델에겐 중간값이 의미가 없다 — 0.2 든 1.0 이든 결과가 같다.
        if "is_news_event" in out.columns:
            out.loc[idx, "is_news_event"] = 1.0
        if "news_event_day" in out.columns:
            out.loc[idx, "news_event_day"] = 1.0
        col = f"event_{sc.news_event}"
        if col in out.columns:
            out.loc[idx, col] = 1.0

    for col, val in sc.extra.items():
        if col in out.columns:
            out.loc[idx, col] = val

    return out


def future_frames(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON,
                  events=None, kbo_source=None,
                  scenario: Scenario | None = None,
                  known_through=None) -> dict[int, pd.Series]:
    """지평별로 '예측 대상 미래 1개 행'을 담은 피처 시리즈.

    `known_through` 는 **외생 변수를 어디까지 아는가**를 정한다. 기본값은 마지막
    관측일 — 즉 우천취소를 아직 모르는 사전 예측이다. 오늘 낮에 취소가 확정됐다면
    `known_through=오늘` 로 올려 당일 수정 예측을 낸다. 시청률 시차 피처는 언제나
    마지막 관측일까지만 쓰므로 타깃 누수는 생기지 않는다.
    """
    sc = scenario or Scenario()
    extended = extend_dates(df, horizon)
    last_obs = df["Date"].max()
    as_of = pd.Timestamp(known_through) if known_through is not None else last_obs

    rows: dict[int, pd.Series] = {}
    for h in range(1, horizon + 1):
        # 마지막 관측일 이후는 우천취소 여부를 알 수 없다 (학습 때와 같은 조건으로 맞춘다)
        frame = build_features(extended, horizon=h, events=events, kbo_source=kbo_source,
                               as_of=as_of, observed_weather=known_through is not None)
        future_mask = frame["Date"] > last_obs
        frame = apply_scenario(frame, future_mask, sc)
        target_date = last_obs + pd.Timedelta(days=h)
        hit = frame.loc[frame["Date"] == target_date]
        if not hit.empty:
            rows[h] = hit.iloc[0]
    return rows


def forecast(df: pd.DataFrame, models: dict[tuple[str, int], TargetModel],
             horizon: int = DEFAULT_HORIZON, events=None, kbo_source=None,
             scenario: Scenario | None = None, targets=None,
             known_through=None) -> pd.DataFrame:
    """향후 `horizon` 일 예측 (지표별 중앙 추정 + 예측 구간).

    반환: Date, horizon, target, channel, label, metric, pred, lo, hi
    """
    targets = list(targets or TARGET_COLS)
    rows = future_frames(df, horizon, events=events, kbo_source=kbo_source,
                         scenario=scenario, known_through=known_through)

    # 속보 국면 시나리오는 모델이 아니라 과거 실측 배율로 반영한다
    sc = scenario or Scenario()
    effects = event_multipliers(df, events).get(sc.news_event, {}) if sc.news_event else {}

    out = []
    for h, row in rows.items():
        X = row.to_frame().T
        for t in targets:
            m = models.get((t, h))
            if m is None:
                continue
            Xf = X[m.feature_cols].astype(float)
            pred = float(m.predict(Xf)[0])
            interval = m.predict_interval(Xf)
            ch = channel_of(t)
            lo = float(min(interval.values(), key=lambda a: a[0])[0]) if interval else np.nan
            hi = float(max(interval.values(), key=lambda a: a[0])[0]) if interval else np.nan
            mult = effects.get(t, 1.0)
            pred, lo, hi = pred * mult, lo * mult, hi * mult
            out.append({
                "Date": row["Date"], "horizon": h, "target": t,
                "channel": ch.key, "label": ch.label,
                "metric": "2049" if is_2049(t) else "가구",
                "pred": max(pred, 0.0),
                "lo": max(min(lo, pred), 0.0), "hi": max(hi, pred),
            })

    return pd.DataFrame(out).sort_values(["metric", "horizon", "label"]).reset_index(drop=True)


def scenario_delta(df: pd.DataFrame, models, horizon: int, scenario: Scenario,
                   events=None, kbo_source=None) -> pd.DataFrame:
    """기본 시나리오 대비 변화량 (What-If 효과 크기)."""
    base = forecast(df, models, horizon, events, kbo_source, Scenario())
    alt = forecast(df, models, horizon, events, kbo_source, scenario)
    merged = base.merge(alt, on=["Date", "horizon", "target", "channel", "label", "metric"],
                        suffixes=("_base", "_alt"))
    merged["delta"] = merged["pred_alt"] - merged["pred_base"]
    return merged


def competition_frame(pred_df: pd.DataFrame, metric: str = "가구",
                      horizon: int = 1, when=None) -> pd.DataFrame:
    """그날의 편성·경쟁 구조와 예상 점유율.

    시간대 버킷이 아니라 방송 구간 겹침으로 계산한다. 평일과 주말의 경쟁 상대가
    다르고(TV조선은 주말에 19:00 뉴스7), 개편 이력에 따라 겹침도 달라진다.
    """
    sub = pred_df[(pred_df["metric"] == metric) & (pred_df["horizon"] == horizon)].copy()
    if sub.empty:
        return sub

    when = pd.Timestamp(when) if when is not None else pd.Timestamp(sub["Date"].iloc[0])
    ov = {k: v[0] for k, v in overlap_matrix(pd.Series([when])).items()}
    pred_of = dict(zip(sub["label"], sub["pred"]))
    by_label = {c.label: c for c in CHANNELS}

    rows = []
    for label, ch in by_label.items():
        if label not in pred_of:
            continue
        own = pred_of[label]
        rivals = {o.label: ov[(ch.key, o.key)] for o in CHANNELS
                  if o.key != ch.key and ov[(ch.key, o.key)] > 0.01}
        weighted = sum(pred_of.get(l, 0.0) * w for l, w in rivals.items())
        a0, a1 = ch.window(when)
        rows.append({
            "방송사": label,
            "방송 구간": f"{int(a0):02d}:{round(a0 % 1 * 60):02d}"
                       f"~{int(a1):02d}:{round(a1 % 1 * 60):02d}",
            "예측": round(own, 3),
            "겹치는 상대": ", ".join(f"{l} {w:.0%}" for l, w in rivals.items()) or "단독",
            "점유율": f"{100 * own / (own + weighted):.1f}%" if weighted else "100%",
        })
    return pd.DataFrame(rows).sort_values("방송 구간").reset_index(drop=True)
