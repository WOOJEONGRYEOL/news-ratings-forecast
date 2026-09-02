"""서울 기상 데이터 (Open-Meteo, 인증 불필요).

핵심 설계: **지평별로 그 시점에 실제 나와 있던 예보를 쓴다.**

  D+3 을 예측하는 모델에게 그날의 실측 강수량을 주면 완벽 예지 누수다. Open-Meteo
  Previous Runs API 는 '며칠 전 예보가 무엇이었는가'를 보관하고 있어, 지평 h 모델에
  `lead=h` 예보를 물려줄 수 있다. 당일 수정 모드에서만 실측(lead 0)을 쓴다.

  아카이브 깊이: D+1 예보는 2023-07 까지, D+3 이상은 2024 년부터. 부족한 구간은
  더 짧은 리드로 대체하고, 그것도 없으면 결측으로 둔다(모델이 알아서 처리).

저녁 시간대(18~22시) 강수를 따로 만든다. 하루 총강수량보다 이쪽이 '집에 머무는가'와
'야구가 취소되는가'에 직결된다.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DATA_DIR

SEOUL = {"latitude": 37.5665, "longitude": 126.9780, "timezone": "Asia/Seoul"}
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OBSERVED_CSV = DATA_DIR / "weather_observed.csv"
LEAD_CSV = DATA_DIR / "weather_leads.csv"
FUTURE_CSV = DATA_DIR / "weather_future.csv"

EVENING = range(18, 22)          # 18~21시 (메인뉴스 + 야구 중계 시간대)
MAX_LEAD = 7
CHUNK_DAYS = 400


def _get(url: str, params: dict, retries: int = 3) -> dict:
    query = urllib.parse.urlencode({**SEOUL, **params}, doseq=True)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{query}", timeout=90) as resp:  # noqa: S310
                body = json.loads(resp.read())
            if body.get("error"):
                raise RuntimeError(body.get("reason", "unknown"))
            return body
        except Exception as exc:                       # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo 요청 실패: {last}")


def _archive_limit() -> pd.Timestamp:
    """아카이브가 실제로 제공하는 마지막 날짜 (보통 어제/그저께)."""
    try:
        probe = _get(ARCHIVE_URL, {
            "start_date": (pd.Timestamp.today() - pd.Timedelta(days=10)).date().isoformat(),
            "end_date": pd.Timestamp.today().date().isoformat(),
            "daily": ["temperature_2m_mean"]})["daily"]
        ok = [t for t, v in zip(probe["time"], probe["temperature_2m_mean"]) if v is not None]
        if ok:
            return pd.Timestamp(ok[-1])
    except RuntimeError as exc:
        # 범위 초과 메시지에 상한이 들어 있다
        import re
        m = re.search(r"to (\d{4}-\d{2}-\d{2})", str(exc))
        if m:
            return pd.Timestamp(m.group(1))
    return pd.Timestamp.today() - pd.Timedelta(days=3)


def _chunks(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        stop = min(s + pd.Timedelta(days=CHUNK_DAYS - 1), e)
        yield s.date().isoformat(), stop.date().isoformat()
        s = stop + pd.Timedelta(days=1)


def _series(times, values) -> pd.Series:
    """시간별 값을 DatetimeIndex 시리즈로.

    주의: pd.Series(times) 로 만든 뒤 `.dt.normalize()` 로 groupby 하면
    RangeIndex 와 DatetimeIndex 가 어긋나 전부 NaN 이 된다. 반드시
    DatetimeIndex 를 만들어 인덱스 자체로 집계할 것.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(times)))
    return pd.Series(pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(),
                     index=idx)


def _daily_sum(s: pd.Series, prefix: str) -> pd.DataFrame:
    """일별 총합 + 저녁(18~21시) 총합."""
    day = s.groupby(s.index.normalize()).sum(min_count=1)
    eve = s[s.index.hour.isin(list(EVENING))]
    eve = eve.groupby(eve.index.normalize()).sum(min_count=1)
    out = pd.DataFrame({f"{prefix}_day": day, f"{prefix}_evening": eve})
    out.index.name = "Date"
    return out


def _daily_mean(s: pd.Series, name: str) -> pd.Series:
    out = s.groupby(s.index.normalize()).mean()
    out.name = name
    out.index.name = "Date"
    return out


def fetch_observed(start: str, end: str) -> pd.DataFrame:
    """실측 일별 기상 (ERA5 아카이브) + 저녁 강수."""
    frames = []
    for a, b in _chunks(start, end):
        daily = _get(ARCHIVE_URL, {
            "start_date": a, "end_date": b,
            "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
                      "precipitation_sum", "precipitation_hours", "snowfall_sum",
                      "cloud_cover_mean", "wind_speed_10m_max"],
        })["daily"]
        d = pd.DataFrame(daily).rename(columns={"time": "Date"})
        d["Date"] = pd.to_datetime(d["Date"])
        d = d.set_index("Date")

        hourly = _get(ARCHIVE_URL, {"start_date": a, "end_date": b,
                                    "hourly": ["precipitation"]})["hourly"]
        eve = _daily_sum(_series(hourly["time"], hourly["precipitation"]), "obs_precip")
        frames.append(d.join(eve[["obs_precip_evening"]]))
        time.sleep(0.5)

    out = pd.concat(frames)
    out.columns = [c if c.startswith("obs_") else f"wx_{c}" for c in out.columns]
    return out.reset_index()


def fetch_lead_forecasts(start: str, end: str, max_lead: int = MAX_LEAD) -> pd.DataFrame:
    """리드별 과거 예보 (previous_dayN). 컬럼: lead, Date, 예보값들."""
    rows = []
    for lead in range(1, max_lead + 1):
        suffix = f"_previous_day{lead}"
        for a, b in _chunks(start, end):
            try:
                h = _get(PREVIOUS_URL, {
                    "start_date": a, "end_date": b,
                    "hourly": [f"precipitation{suffix}", f"temperature_2m{suffix}",
                               f"cloud_cover{suffix}"],
                })["hourly"]
            except RuntimeError:
                continue
            precip = _daily_sum(_series(h["time"], h[f"precipitation{suffix}"]), "fc_precip")
            agg = precip.join(pd.concat([
                _daily_mean(_series(h["time"], h[f"temperature_2m{suffix}"]), "fc_temp"),
                _daily_mean(_series(h["time"], h[f"cloud_cover{suffix}"]), "fc_cloud"),
            ], axis=1))
            agg["lead"] = lead
            rows.append(agg.reset_index())
            time.sleep(0.5)

    if not rows:
        raise RuntimeError("과거 예보를 하나도 받지 못했습니다.")
    return pd.concat(rows, ignore_index=True)


def fetch_future(days: int = 16) -> pd.DataFrame:
    """앞으로 며칠간의 예보 (실제 운용용)."""
    h = _get(FORECAST_URL, {"forecast_days": days,
                            "hourly": ["precipitation", "temperature_2m", "cloud_cover"]})["hourly"]
    precip = _daily_sum(_series(h["time"], h["precipitation"]), "fc_precip")
    return precip.join(pd.concat([
        _daily_mean(_series(h["time"], h["temperature_2m"]), "fc_temp"),
        _daily_mean(_series(h["time"], h["cloud_cover"]), "fc_cloud"),
    ], axis=1)).reset_index()


def _merge_save(new: pd.DataFrame, path: Path, keys: list[str]) -> int:
    """기존 캐시와 합쳐 저장한다. 겹치는 행은 새 값으로 교체.

    통째로 덮어쓰면 안 된다 — 매일 최근 구간만 받아오는 운용에서 과거가 날아간다.
    (실제로 한 번 1,186일 → 401일로 잘렸다.)
    """
    if path.exists():
        try:
            old = pd.read_csv(path, parse_dates=["Date"])
            new = pd.concat([old, new], ignore_index=True)
        except (OSError, ValueError, KeyError):
            pass
    new = (new.sort_values("Date")
              .drop_duplicates(subset=keys, keep="last")
              .sort_values(keys)
              .reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return len(new)


def build_cache(start: str, end: str, max_lead: int = MAX_LEAD) -> None:
    """기상 캐시 갱신. 기존 파일이 있으면 **합친다**(덮어쓰지 않는다)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    limit = _archive_limit()
    end = min(pd.Timestamp(end), limit).date().isoformat()
    print(f"실측 기상 수집… ({start} ~ {end}, 아카이브 상한 {limit.date()})", flush=True)
    n = _merge_save(fetch_observed(start, end), OBSERVED_CSV, ["Date"])
    print(f"  실측 누적 {n}일", flush=True)

    print("리드별 과거 예보 수집…", flush=True)
    n = _merge_save(fetch_lead_forecasts(start, end, max_lead), LEAD_CSV, ["lead", "Date"])
    print(f"  예보 누적 {n}행", flush=True)

    print("미래 예보 수집…", flush=True)
    fetch_future().to_csv(FUTURE_CSV, index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------
def load_for_horizon(dates, horizon: int, use_observed: bool = False) -> pd.DataFrame:
    """지평 h 모델이 쓸 기상 피처.

    use_observed=True (당일 수정 모드) 면 실측을, 아니면 lead=h 예보를 쓴다.
    lead=h 가 없는 과거 구간은 더 짧은 리드로 대체한다.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize()
    cols = ["wx_precip", "wx_precip_evening", "wx_temp", "wx_cloud"]
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    out.index.name = "Date"

    if use_observed and OBSERVED_CSV.exists():
        obs = pd.read_csv(OBSERVED_CSV, parse_dates=["Date"]).set_index("Date")
        out["wx_precip"] = obs["wx_precipitation_sum"].reindex(idx)
        out["wx_precip_evening"] = obs["obs_precip_evening"].reindex(idx)
        out["wx_temp"] = obs["wx_temperature_2m_mean"].reindex(idx)
        out["wx_cloud"] = obs["wx_cloud_cover_mean"].reindex(idx)
        return out

    if LEAD_CSV.exists():
        leads = pd.read_csv(LEAD_CSV, parse_dates=["Date"])
        # 원하는 리드부터 짧은 쪽으로 내려가며 채운다
        for lead in sorted({horizon, *range(1, MAX_LEAD + 1)},
                           key=lambda x: (abs(x - horizon), x)):
            sub = leads.loc[leads["lead"] == lead].set_index("Date")
            if sub.empty:
                continue
            for dst, src in (("wx_precip", "fc_precip_day"),
                             ("wx_precip_evening", "fc_precip_evening"),
                             ("wx_temp", "fc_temp"), ("wx_cloud", "fc_cloud")):
                if src in sub.columns:
                    out[dst] = out[dst].fillna(sub[src].reindex(idx))
            if not out.isna().any().any():
                break

    if FUTURE_CSV.exists():
        fut = pd.read_csv(FUTURE_CSV, parse_dates=["Date"]).set_index("Date")
        for dst, src in (("wx_precip", "fc_precip_day"),
                         ("wx_precip_evening", "fc_precip_evening"),
                         ("wx_temp", "fc_temp"), ("wx_cloud", "fc_cloud")):
            if src in fut.columns:
                out[dst] = out[dst].fillna(fut[src].reindex(idx))

    return out
