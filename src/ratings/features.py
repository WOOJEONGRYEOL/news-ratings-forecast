"""피처 엔지니어링.

설계 원칙
---------
1) 외생 피처(달력·공휴일·일몰·KBO 편성)는 **미래 시점에도 확정적으로 알 수 있는 값**만
   사용한다. 예측 시점에 모르는 값을 넣으면 백테스트 성적만 좋아지고 실전에서 무너진다.
2) 자기회귀 피처(lag/rolling)는 예측 지평 h 에 맞춰 전부 `shift(h)` 이상으로 만든다.
   h=7 모델이 어제 시청률을 보게 두면 7일 예측 성능을 과대평가하게 된다.
3) 8개 지표의 lag 를 모든 모델이 공유한다 → 종편 4사 간 시청자 이동(cross-lag) 학습.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kbo, weather
from .config import (
    CHANNELS, KBO_ERA_START, LAGS, ROLL_WINDOWS, TARGET_COLS,
    overlap_matrix, schedule_frame,
)

SEOUL_LAT, SEOUL_LON, KST = 37.5665, 126.9780, 9.0
EPS = 1e-6


# --------------------------------------------------------------------------
# 천문: 서울 기준 실제 일몰 시각 (NOAA 근사식)
# --------------------------------------------------------------------------
def solar_times(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """서울 기준 일몰 시각(시, 24h 실수)과 낮 길이."""
    doy = dates.dayofyear.to_numpy(dtype=float)
    g = 2 * np.pi / 365.0 * (doy - 1)

    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))

    lat = np.radians(SEOUL_LAT)
    cos_ha = (np.cos(np.radians(90.833)) / (np.cos(lat) * np.cos(decl))
              - np.tan(lat) * np.tan(decl))
    ha = np.arccos(np.clip(cos_ha, -1.0, 1.0))          # radian
    ha_hours = np.degrees(ha) / 15.0

    solar_noon = 12.0 - SEOUL_LON / 15.0 + KST - eqtime / 60.0
    return pd.DataFrame(
        {"sunset_hour": solar_noon + ha_hours,
         "daylight_hours": 2 * ha_hours},
        index=dates,
    )


# --------------------------------------------------------------------------
# 달력 · 공휴일 · 연휴 구조
# --------------------------------------------------------------------------
def _kr_holidays(years) -> dict:
    """한국 공휴일 {date: name}. holidays 패키지가 없으면 빈 dict."""
    try:
        import holidays as _h
    except ImportError:
        return {}
    return dict(_h.country_holidays("KR", years=sorted(set(years))))


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """요일·계절·공휴일·연휴 길이 등 달력 파생."""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)

    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_friday"] = (idx.dayofweek == 4).astype(int)
    out["is_monday"] = (idx.dayofweek == 0).astype(int)
    out["month"] = idx.month
    out["day"] = idx.day
    out["dayofyear"] = idx.dayofyear
    out["weekofyear"] = idx.isocalendar().week.to_numpy(dtype=int)

    ang = 2 * np.pi * out["dayofyear"] / 365.25
    out["sin_year"], out["cos_year"] = np.sin(ang), np.cos(ang)
    dow_ang = 2 * np.pi * out["dayofweek"] / 7.0
    out["sin_dow"], out["cos_dow"] = np.sin(dow_ang), np.cos(dow_ang)

    hol = _kr_holidays(idx.year.unique())
    hol_dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(hol.keys())))) if hol \
        else pd.DatetimeIndex([])
    is_hol = idx.isin(hol_dates)
    out["is_holiday"] = is_hol.astype(int)
    out["is_holiday_eve"] = idx.isin(hol_dates - pd.Timedelta(days=1)).astype(int)
    out["is_holiday_after"] = idx.isin(hol_dates + pd.Timedelta(days=1)).astype(int)

    # 설날·추석 연휴(±2일): 시청 패턴이 통째로 달라지는 구간
    big = [d for d, name in hol.items()
           if any(k in name for k in ("설날", "추석", "New Year", "Chuseok"))]
    big_idx = pd.DatetimeIndex(sorted(pd.to_datetime(list(big)))) if big \
        else pd.DatetimeIndex([])
    big_window = pd.DatetimeIndex(
        sorted({d + pd.Timedelta(days=k) for d in big_idx for k in range(-2, 3)})
    ) if len(big_idx) else pd.DatetimeIndex([])
    out["is_big_holiday"] = idx.isin(big_window).astype(int)

    # 연휴 구조: 주말/공휴일이 연속으로 이어지는 덩어리의 길이와 그 안에서의 위치
    is_off = (out["is_weekend"] | out["is_holiday"]).to_numpy().astype(bool)
    run_id = np.cumsum(np.r_[True, is_off[1:] != is_off[:-1]])
    run = pd.Series(run_id, index=idx)
    run_len = run.map(run.value_counts())
    out["consecutive_off_days"] = np.where(is_off, run_len, 0)
    out["off_day_index"] = np.where(is_off, run.groupby(run).cumcount() + 1, 0)
    # 징검다리: 하루짜리 평일이 휴일 사이에 낀 경우
    out["is_bridge_day"] = ((~is_off) & (run_len == 1)).astype(int)

    if len(hol_dates):
        pos = np.searchsorted(hol_dates.to_numpy(), idx.to_numpy())
        nxt = np.where(pos < len(hol_dates), pos, len(hol_dates) - 1)
        prv = np.clip(pos - 1, 0, len(hol_dates) - 1)
        out["days_to_holiday"] = np.clip(
            (hol_dates.to_numpy()[nxt] - idx.to_numpy()) / np.timedelta64(1, "D"), 0, 30)
        out["days_from_holiday"] = np.clip(
            (idx.to_numpy() - hol_dates.to_numpy()[prv]) / np.timedelta64(1, "D"), 0, 30)
    else:
        out["days_to_holiday"] = out["days_from_holiday"] = 30.0

    return out


def temperature_proxy(dates: pd.DatetimeIndex) -> pd.Series:
    """서울 일평균 기온 계절 근사 (실측 API 연동 전 임시치)."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=float)
    return pd.Series(11.5 - 14.0 * np.cos(2 * np.pi * (doy - 25) / 365.25),
                     index=pd.DatetimeIndex(dates), name="temp_proxy")


# --------------------------------------------------------------------------
# 이벤트 (대형 속보 국면)
# --------------------------------------------------------------------------
# 사건 유형. events.csv 의 `type` 컬럼 값이며, 유형마다 별도 변수를 만든다.
# 같은 '속보 국면'이라도 채널별 반응이 정반대다 —
#   윤석열_사법: JTBC 2배, 나머지 3사 하락
#   이재명_사법: 채널A·MBN·TV조선 1.4~1.6배, JTBC 하락
# 한 변수에 뭉쳐두면 서로 상쇄돼 JTBC 신호만 남는다.
EVENT_TYPES = ("윤석열_사법", "이재명_사법", "대선_정국")


def event_features(dates: pd.DatetimeIndex, events: pd.DataFrame | None) -> pd.DataFrame:
    """events.csv(start, end, label, type, weight) -> 구간 플래그.

    미래 예측에서는 기본값 0 이며, What-If 시뮬레이터에서 유형을 골라 켤 수 있다.
    구간이 유형당 1~2 개뿐이라 통계적으로는 약하다 — 해석·시나리오 용도로 본다.
    """
    idx = pd.DatetimeIndex(dates)
    cols = ["is_news_event", "news_event_day"] + [f"event_{t}" for t in EVENT_TYPES]
    out = pd.DataFrame({c: np.zeros(len(idx)) for c in cols}, index=idx)
    if events is None or events.empty:
        return out

    for _, row in events.iterrows():
        start, end = pd.to_datetime(row["start"]), pd.to_datetime(row["end"])
        mask = (idx >= start) & (idx <= end)
        if not mask.any():
            continue
        weight = float(row.get("weight", 1.0) or 1.0)
        out.loc[mask, "is_news_event"] = weight
        out.loc[mask, "news_event_day"] = (idx[mask] - start).days + 1
        etype = str(row.get("type", "") or "").strip()
        if etype in EVENT_TYPES:
            out.loc[mask, f"event_{etype}"] = weight
    return out


# --------------------------------------------------------------------------
# 외생 피처 통합 (미래 날짜에도 계산 가능)
# --------------------------------------------------------------------------
def build_exogenous(dates, events: pd.DataFrame | None = None,
                    kbo_source=None, as_of=None, horizon: int = 1,
                    observed_weather: bool = False) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize()

    # 연휴 런 길이·다음 공휴일까지 일수는 범위 양끝에서 잘리면 값이 틀어진다.
    # 앞뒤 40일을 덧대어 계산한 뒤 요청 구간만 잘라 쓴다.
    pad = pd.date_range(idx.min() - pd.Timedelta(days=40),
                        idx.max() + pd.Timedelta(days=40), freq="D")
    parts = [calendar_features(pad), solar_times(pad), temperature_proxy(pad),
             event_features(pad, events)]
    exo = pd.concat(parts, axis=1).reindex(idx)

    k = (kbo.load_daily(idx, kbo_source, as_of=as_of) if kbo_source is not None
         else kbo.load_daily(idx, as_of=as_of)).set_index("Date")
    exo = exo.join(k, how="left")
    exo[kbo.DAILY_COLS] = exo[kbo.DAILY_COLS].fillna(0.0)

    # --- 기상 -------------------------------------------------------------
    # 지평 h 모델에는 **h일 전에 실제 나와 있던 예보**를 준다. 그날의 실측을 주면
    # 완벽 예지 누수다. 당일 수정 모드에서만 실측을 쓴다.
    try:
        wx = weather.load_for_horizon(idx, horizon, use_observed=observed_weather)
        exo = exo.join(wx)
    except (FileNotFoundError, KeyError, ValueError):
        for c in ("wx_precip", "wx_precip_evening", "wx_temp", "wx_cloud"):
            exo[c] = np.nan

    # 계절 근사치는 실측/예보가 있으면 그쪽을 우선한다
    exo["temp_effective"] = exo["wx_temp"].fillna(exo["temp_proxy"])
    exo["is_rainy"] = (exo["wx_precip"].fillna(0.0) >= 1.0).astype(float)
    exo["is_rainy_evening"] = (exo["wx_precip_evening"].fillna(0.0) >= 0.5).astype(float)
    # 로그 스케일 — 5mm 와 50mm 의 차이보다 0mm 와 5mm 의 차이가 크다
    exo["precip_log"] = np.log1p(exo["wx_precip"].fillna(0.0))
    exo["precip_evening_log"] = np.log1p(exo["wx_precip_evening"].fillna(0.0))

    # 상호작용: 일몰이 이른 계절 + 추운 날 = 귀가 후 TV 앞 체류 ↑
    exo["cold_and_dark"] = (20.0 - exo["temp_effective"]) * (20.0 - exo["sunset_hour"])
    # 야구가 19시 뉴스를 덮는 정도를 '서울 구장 여부'로 가중
    exo["kbo_pressure_19"] = exo["kbo_overlap_19"] * (1.0 + 0.5 * exo["kbo_seoul_games"])
    exo["kbo_pressure_21"] = exo["kbo_overlap_21"] * (1.0 + 0.5 * exo["kbo_seoul_games"])

    # --- 시변 편성표 -------------------------------------------------------
    # JTBC 뉴스룸은 2024-03-11 부터 21시대 → 18:50 로 옮겼다. 고정 슬롯으로 두면
    # 데이터의 22% 구간에서 경쟁 구조가 통째로 틀린다.
    sched = schedule_frame(idx)
    exo = exo.join(sched)

    # 채널쌍 겹침 — 평일/주말과 개편 이력을 모두 반영한다
    ov = overlap_matrix(idx)
    for ch in CHANNELS:
        # 그날 나와 방송 시간이 겹치는 상대가 얼마나 되는가 (0=단독, 3=전원 정면충돌)
        exo[f"{ch.key}_rival_weight"] = sum(
            ov[(ch.key, o.key)] for o in CHANNELS if o.key != ch.key)

    # 비 오는 날 야구가 편성돼 있다 = 취소 가능성. 사전 예측이 우천취소를 간접적으로
    # 잡을 수 있는 유일한 통로다.
    exo["kbo_rain_risk"] = exo["kbo_any_game"] * exo["precip_evening_log"]

    # 채널별 '내 방송 시간대와 야구 중계가 겹치는 시간' — 편성이 바뀌면 자동으로 따라간다
    start = exo["kbo_main_start"]
    playing = start > 0
    for ch in CHANNELS:
        air0, air1 = exo[f"{ch.key}_air_hour"], exo[f"{ch.key}_air_end"]
        lap = (np.minimum(start + kbo.GAME_HOURS, air1)
               - np.maximum(start, air0)).clip(lower=0.0)
        # 방송 길이로 정규화 — 35분짜리 주말 뉴스와 70분짜리 평일 뉴스를 같은 척도로
        exo[f"{ch.key}_kbo_overlap"] = np.where(playing, lap / (air1 - air0), 0.0)
        exo[f"{ch.key}_kbo_pressure"] = (exo[f"{ch.key}_kbo_overlap"]
                                         * (1.0 + 0.5 * exo["kbo_seoul_games"]))

    # --- 중계권 체제 이전 구간은 야구 변수를 결측 처리 -----------------------
    # 2024 시즌부터 티빙 유료 독점이 되며 야구 시청이 온라인 → TV 로 옮겨갔다.
    # 그 이전의 야구-뉴스 관계는 지금과 다르므로 학습에 쓰지 않는다.
    # 0 으로 두면 '경기 없음'이라는 다른 거짓 신호가 되므로 NaN 으로 둔다
    # (LightGBM 은 결측을 별도 분기로 처리한다).
    # 파생 변수까지 다 만든 **뒤에** 지워야 한다 — np.where(playing, ...) 같은 코드가
    # NaN 을 False 로 흘려보내 0 을 남기기 때문이다.
    pre_era = exo.index < pd.Timestamp(KBO_ERA_START)
    if pre_era.any():
        kbo_cols = [c for c in exo.columns if c.startswith("kbo_") or "_kbo_" in c]
        exo.loc[pre_era, kbo_cols] = np.nan

    exo.index.name = "Date"
    return exo


# --------------------------------------------------------------------------
# 자기회귀 피처 (지평 h 에 맞춰 shift)
# --------------------------------------------------------------------------
def build_autoregressive(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """h일 앞을 예측할 때 실제로 손에 쥘 수 있는 과거 시청률만으로 만든 피처."""
    h = int(horizon)
    src = df.set_index("Date")[TARGET_COLS]
    out = pd.DataFrame(index=src.index)

    for col in TARGET_COLS:
        s = src[col]
        base = s.shift(h)                       # 예측 시점에 알 수 있는 가장 최근 값

        for lag in LAGS:
            if lag >= h:
                out[f"{col}_lag_{lag}"] = s.shift(lag)

        for w in ROLL_WINDOWS:
            out[f"{col}_roll_mean_{w}"] = base.rolling(w).mean()
            out[f"{col}_roll_std_{w}"] = base.rolling(w).std()

        # 같은 요일 최근 4주 평균 — 요일 절벽이 큰 시청률에서 가장 강한 단일 신호
        weekly = [s.shift(7 * k) for k in range(1, 5) if 7 * k >= h]
        if weekly:
            out[f"{col}_dow_mean_4"] = pd.concat(weekly, axis=1).mean(axis=1)

        # 추세/편차
        out[f"{col}_momentum"] = base - s.shift(h + 7)
        out[f"{col}_vs_roll30"] = base - base.rolling(30).mean()

    # --- 방송 구간 겹침 기반 경쟁 구조 --------------------------------------
    # 19/21시대 버킷 대신 실제 겹침 비율로 가중한다. JTBC 평일이 19:50 이던 시절
    # 채널A(19:00)와는 10분만 겹쳤고(0.17), 18:50 으로 옮긴 뒤 0.83, 18:30 이 된
    # 지금은 0.50 이다. 주말엔 TV조선이 19:00 뉴스7 로 내려와 구조가 통째로 바뀐다.
    ov = overlap_matrix(out.index)
    lag_h = max(1, h)

    def _lag_col(col: str) -> str:
        c = f"{col}_lag_{lag_h}"
        return c if c in out.columns else f"{col}_roll_mean_7"

    for metric, cols in (("hh", [c.household_col for c in CHANNELS]),
                         ("2049", [c.col_2049 for c in CHANNELS])):
        lag_by_key = {ch.key: out[_lag_col(col)] for ch, col in zip(CHANNELS, cols)}
        for ch in CHANNELS:
            own = lag_by_key[ch.key]
            rivals = sum(lag_by_key[o.key] * ov[(ch.key, o.key)]
                         for o in CHANNELS if o.key != ch.key)
            apart = sum(lag_by_key[o.key] * (1.0 - ov[(ch.key, o.key)])
                        for o in CHANNELS if o.key != ch.key)
            out[f"{ch.key}_rivals_{metric}"] = rivals
            out[f"{ch.key}_slot_share_{metric}"] = own / (own + rivals + EPS)
            # 겹치지 않는 채널들의 합 = 그날 종편 뉴스 수요 총량의 대리 지표
            out[f"{ch.key}_apart_{metric}"] = apart

    # 종편 4사 전체 파이 (시간대 무관)
    out["all4_sum_hh"] = sum(out[_lag_col(c.household_col)] for c in CHANNELS)

    # 채널별 2049 비중 = 젊은 시청층 유입 지표
    for ch in CHANNELS:
        hh = f"{ch.household_col}_roll_mean_7"
        yg = f"{ch.col_2049}_roll_mean_7"
        out[f"{ch.key}_youth_ratio"] = out[yg] / (out[hh] + EPS)

    return out


def build_features(df: pd.DataFrame, horizon: int = 1,
                   events: pd.DataFrame | None = None,
                   kbo_source=None, as_of=None,
                   observed_weather: bool = False) -> pd.DataFrame:
    """외생 + 자기회귀 피처를 합친 학습/추론용 프레임 (타깃·Date 포함).

    `as_of` 이후 행에서는 예측 시점에 알 수 없는 외생 값(우천취소)이 0으로 지워진다.
    """
    exo = build_exogenous(df["Date"], events=events, kbo_source=kbo_source, as_of=as_of,
                          horizon=horizon, observed_weather=observed_weather)
    ar = build_autoregressive(df, horizon)

    out = df.set_index("Date").join(exo).join(ar)
    out.index.name = "Date"
    return out.reset_index()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """타깃·메타 컬럼을 제외한 실제 입력 피처 목록."""
    drop = set(TARGET_COLS) | {"Date", "날짜", "is_imputed"}
    return [c for c in frame.columns if c not in drop]
