"""KBO 프로야구 일정 -> 일자별 외생 변수.

kbo-forecast 프로젝트가 수집해 둔 네이버 스포츠 원본(games_YYYY_MM.json)을 그대로
활용한다. 실제 시작 시각·우천취소·구장이 들어 있어 요일 기반 휴리스틱보다 정확하다.
(2026시즌은 평일 18:30/19:00이 섞여 있어 휴리스틱으로는 잡히지 않는다.)
원본이 없으면 요일/계절 근사치로 자동 폴백한다.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import KBO_CACHE_DIR, KBO_DIRS, KBO_SCHEDULE_CSV

# 네이버 스포츠 공개 API (인증 불필요). 캐시 JSON 과 같은 소스라 형식이 동일하다.
NAVER_API = "https://api-gw.sports.naver.com/schedule/games"

GAME_HOURS = 3.25          # KBO 평균 경기 시간(연장 포함 근사)
SEOUL_STADIUMS = {"잠실", "고척"}
NEWS_WINDOW_19 = (19.0, 20.0)   # 채널A·MBN·JTBC 메인뉴스 방송 구간
NEWS_WINDOW_21 = (21.0, 22.0)   # TV조선 뉴스9 방송 구간

DAILY_COLS = [
    "kbo_games", "kbo_cancelled", "kbo_any_game", "kbo_seoul_games",
    "kbo_first_start", "kbo_main_start",
    "kbo_games_19", "kbo_games_21", "kbo_overlap_19", "kbo_overlap_21",
    "kbo_is_postseason",
]


def _overlap(start: float, win: tuple[float, float]) -> float:
    """경기 진행 구간 [start, start+GAME_HOURS] 과 뉴스 방송 구간의 겹침(시간)."""
    lo, hi = win
    return max(0.0, min(start + GAME_HOURS, hi) - max(start, lo))


AGG_COLS = [c for c in DAILY_COLS if c != "kbo_cancelled"]
SCHED_PREFIX = "sched__"


def _aggregate(games: pd.DataFrame) -> pd.DataFrame:
    """경기 목록 -> 일자별 집계 (취소 수 제외)."""
    g = games.copy()
    g["ov19"] = g["start"].map(lambda x: _overlap(x, NEWS_WINDOW_19))
    g["ov21"] = g["start"].map(lambda x: _overlap(x, NEWS_WINDOW_21))

    by_day = g.groupby("Date")
    agg = pd.DataFrame({
        "kbo_games": by_day.size(),
        "kbo_seoul_games": by_day["seoul"].sum(),
        "kbo_first_start": by_day["start"].min(),
        # 최빈 시작시각 = 그날 편성의 '주 시간대'
        "kbo_main_start": by_day["start"].agg(lambda x: x.mode().iat[0]),
        "kbo_games_19": by_day["ov19"].agg(lambda x: (x > 0).sum()),
        "kbo_games_21": by_day["ov21"].agg(lambda x: (x > 0).sum()),
        "kbo_overlap_19": by_day["ov19"].max(),
        "kbo_overlap_21": by_day["ov21"].max(),
    })
    agg["kbo_any_game"] = (agg["kbo_games"] > 0).astype(int)
    # 포스트시즌: 10~11월에 하루 1~2경기만 편성 (정규시즌은 하루 5경기)
    agg["kbo_is_postseason"] = (
        np.isin(agg.index.month, [10, 11]) & agg["kbo_games"].between(1, 2)
    ).astype(int)
    return agg[AGG_COLS]


def _games_frame(games: list[dict]) -> pd.DataFrame:
    """네이버 경기 dict 리스트 -> 경기 단위 프레임 (파일·실시간 공용)."""
    rows = []
    for g in games:
        ts = pd.to_datetime(g.get("gameDateTime") or g.get("gameDate"), errors="coerce")
        if pd.isna(ts):
            continue
        rows.append({
            "gameId": g.get("gameId") or
                      f"{g.get('gameDate')}{g.get('awayTeamCode')}{g.get('homeTeamCode')}",
            "Date": ts.normalize(),
            "start": ts.hour + ts.minute / 60.0,
            "cancel": bool(g.get("cancel")) or g.get("statusCode") == "CANCEL",
            "seoul": g.get("stadium") in SEOUL_STADIUMS,
        })
    if not rows:
        return pd.DataFrame(columns=["gameId", "Date", "start", "cancel", "seoul"])
    return pd.DataFrame(rows).drop_duplicates(subset=["gameId"], keep="last")


def _read_games(source_dir=None) -> pd.DataFrame:
    """games_*.json -> 경기 단위 프레임 (gameId 기준 중복 제거).

    여러 디렉터리를 순서대로 읽어 합친다. 뒤쪽 디렉터리가 우선이므로
    자체 수집본(KBO_CACHE_DIR)이 외부 수집본을 덮어쓴다.
    """
    dirs = [Path(source_dir)] if source_dir is not None else [Path(d) for d in KBO_DIRS]
    files = [f for d in dirs if d.exists() for f in sorted(d.glob("games_*.json"))]
    if not files:
        raise FileNotFoundError(
            "KBO 원본 JSON 없음. `python scripts/sync_kbo.py` 로 수집하세요.")

    games: list[dict] = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, list):
            games.extend(payload)

    out = _games_frame(games)
    if out.empty:
        raise ValueError("KBO 원본에서 유효한 경기를 추출하지 못했습니다.")
    return out


def parse_source(source_dir=None) -> pd.DataFrame:
    """일자별 KBO 집계.

    두 벌을 만든다.
      - `kbo_*`        : 실제 진행된 경기 기준 (사후에만 알 수 있다)
      - `sched__kbo_*` : 편성 기준. 취소 여부와 무관하며 몇 달 전에 확정된다.
    예측 시점 이후 구간에는 편성 기준을 써야 한다 — 그러지 않으면 "내일 비가 와서
    5경기가 다 취소된다"는 걸 미리 아는 모델이 된다.
    """
    games = _read_games(source_dir)
    all_days = pd.DatetimeIndex(sorted(set(games["Date"])))

    actual = _aggregate(games.loc[~games["cancel"]]).reindex(all_days)
    sched = _aggregate(games).reindex(all_days)
    sched.columns = [SCHED_PREFIX + c for c in sched.columns]

    out = actual.join(sched)
    out["kbo_cancelled"] = games.loc[games["cancel"]].groupby("Date").size()

    zero = [c for c in out.columns if "start" not in c]
    out[zero] = out[zero].fillna(0.0)
    starts = [c for c in out.columns if "start" in c]
    out[starts] = out[starts].fillna(-1.0)

    out.index.name = "Date"
    return out.reset_index()[["Date", *DAILY_COLS, *[SCHED_PREFIX + c for c in AGG_COLS]]]


def build_schedule_cache(source_dir=None,
                         out: Path = KBO_SCHEDULE_CSV) -> pd.DataFrame:
    """원본을 파싱해 data/kbo_schedule.csv 로 저장 (원본 접근 불가 환경 대비)."""
    df = parse_source(source_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def _heuristic(dates) -> pd.DataFrame:
    """원본이 없을 때 쓰는 근사 일정 (요일·월 기반)."""
    idx = pd.DatetimeIndex(dates)
    month, day, dow = idx.month, idx.day, idx.dayofweek
    in_season = (
        ((month == 3) & (day >= 22)) | np.isin(month, [4, 5, 6, 7, 8, 9, 10])
        | ((month == 11) & (day <= 10))
    )
    game_day = in_season & (dow != 0)     # 월요일은 KBO 정기 휴식일

    start = np.where(
        dow <= 4, 18.5,
        np.where(dow == 5, np.where(np.isin(month, [6, 7, 8]), 18.0, 17.0),
                 np.where(np.isin(month, [6, 7, 8]), 17.0, 14.0)),
    ).astype(float)
    start = np.where(game_day, start, -1.0)

    ov19 = np.array([_overlap(s, NEWS_WINDOW_19) if s > 0 else 0.0 for s in start])
    ov21 = np.array([_overlap(s, NEWS_WINDOW_21) if s > 0 else 0.0 for s in start])
    n = np.where(game_day, 5, 0)

    out = pd.DataFrame({
        "Date": idx,
        "kbo_games": n,
        "kbo_cancelled": 0.0,
        "kbo_any_game": game_day.astype(int),
        "kbo_seoul_games": np.where(game_day, 1, 0),
        "kbo_first_start": start,
        "kbo_main_start": start,
        "kbo_games_19": np.where(ov19 > 0, n, 0),
        "kbo_games_21": np.where(ov21 > 0, n, 0),
        "kbo_overlap_19": ov19,
        "kbo_overlap_21": ov21,
        "kbo_is_postseason": 0,
    })
    for c in AGG_COLS:                      # 휴리스틱은 취소를 모르니 편성=실제
        out[SCHED_PREFIX + c] = out[c]
    return out


# 예측 시점에 알 수 없는 컬럼: 내일 경기가 우천취소될지는 오늘 모른다.
UNKNOWABLE_AHEAD = ["kbo_cancelled"]


def fetch_live(start: str, end: str, timeout: int = 20, size: int = 600) -> list[dict]:
    """네이버 스포츠에서 해당 기간 경기를 **실시간으로** 가져온다.

    당일 우천취소는 캐시 파일에 없다 — 경기 1~2시간 전에 발표되기 때문이다.
    당일 수정 모드에서 이 함수로 직접 확인한다.
    """
    params = {
        "fields": "basic,stadium,statusInfo",
        "upperCategoryId": "kbaseball",
        "categoryId": "kbo",
        # 한 달은 최대 ~130 경기다. 기본 100 으로 두면 월 단위 수집이 조용히 잘린다.
        "fromDate": str(start), "toDate": str(end), "size": size,
    }
    url = f"{NAVER_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://m.sports.naver.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310
        body = json.loads(resp.read())
    if not body.get("success"):
        raise RuntimeError(f"네이버 API 응답 실패: {body.get('code')}")
    return body["result"]["games"]


def live_status(date, timeout: int = 20) -> dict:
    """특정 날짜의 KBO 편성/취소 현황 요약.

    반환: scheduled(편성), cancelled(취소), remaining(남은 경기), main_start(주력 시작 시각),
          games(경기별 요약 리스트), fetched_at
    """
    day = pd.Timestamp(date).date().isoformat()
    games = fetch_live(day, day, timeout=timeout)
    rows = []
    for g in games:
        ts = pd.to_datetime(g.get("gameDateTime") or g.get("gameDate"), errors="coerce")
        cancelled = bool(g.get("cancel")) or g.get("statusCode") == "CANCEL"
        rows.append({
            "시각": "" if pd.isna(ts) else f"{ts.hour:02d}:{ts.minute:02d}",
            "경기": f"{g.get('awayTeamName','?')}@{g.get('homeTeamName','?')}",
            "구장": g.get("stadium", ""),
            "상태": "취소" if cancelled else str(g.get("statusInfo") or g.get("statusCode") or ""),
            "_cancel": cancelled,
            "_start": None if pd.isna(ts) else ts.hour + ts.minute / 60.0,
        })

    live = [r for r in rows if not r["_cancel"]]
    starts = [r["_start"] for r in live if r["_start"] is not None]
    return {
        "date": day,
        "scheduled": len(rows),
        "cancelled": sum(r["_cancel"] for r in rows),
        "remaining": len(live),
        "main_start": max(set(starts), key=starts.count) if starts else -1.0,
        "games": rows,
        "fetched_at": pd.Timestamp.now().strftime("%H:%M"),
    }


def sync_schedule(seasons=None, refresh_months: int = 2,
                  dest: Path = KBO_CACHE_DIR) -> dict:
    """네이버에서 월별 일정을 받아 data/kbo_games/ 에 저장한다.

    외부 프로젝트(kbo-forecast) 수집에 의존하지 않기 위한 자체 캐시다.
    지나간 달의 결과는 바뀌지 않으므로 이미 있으면 건너뛰고, 최근 `refresh_months`
    개월과 미래 달만 다시 받는다 — 크롤링 예절이자 속도 문제다.
    """
    today = pd.Timestamp.today()
    seasons = list(seasons or range(2023, today.year + 1))
    dest.mkdir(parents=True, exist_ok=True)

    cutoff = (today - pd.DateOffset(months=refresh_months)).to_period("M")
    saved, skipped, failed = [], [], []

    for season in seasons:
        for month in range(3, 12):                 # KBO 정규시즌 3~11월
            period = pd.Period(f"{season}-{month:02d}", freq="M")
            if period > today.to_period("M"):
                continue
            path = dest / f"games_{season}_{month:02d}.json"
            if path.exists() and period < cutoff:
                skipped.append(path.name)
                continue
            last_day = period.days_in_month
            try:
                games = fetch_live(f"{season}-{month:02d}-01",
                                   f"{season}-{month:02d}-{last_day:02d}")
            except Exception as exc:               # noqa: BLE001
                failed.append(f"{path.name}: {exc}")
                continue
            if games:
                path.write_text(json.dumps(games, ensure_ascii=False), encoding="utf-8")
                saved.append(path.name)
            time.sleep(0.4)                        # 크롤링 예절

    return {"saved": saved, "skipped": len(skipped), "failed": failed}


def fetch_live_daily(start, end, timeout: int = 20) -> pd.DataFrame:
    """네이버 실시간 일정 -> `parse_source` 와 같은 형식의 일자별 집계.

    네이버는 약 4주치 미래 편성을 준다. 캐시가 끝난 뒤 구간을 요일 휴리스틱으로
    메우면 크게 틀린다 — 실제로는 목요일 17:00 경기도 있고(휴리스틱은 18:30),
    시즌 막바지엔 하루 5경기가 4·3·2 로 줄어든다.
    """
    games = _games_frame(fetch_live(pd.Timestamp(start).date().isoformat(),
                                    pd.Timestamp(end).date().isoformat(),
                                    timeout=timeout))
    if games.empty:
        return pd.DataFrame(columns=["Date", *DAILY_COLS,
                                     *[SCHED_PREFIX + c for c in AGG_COLS]])

    all_days = pd.DatetimeIndex(sorted(set(games["Date"])))
    actual = _aggregate(games.loc[~games["cancel"]]).reindex(all_days)
    sched = _aggregate(games).reindex(all_days)
    sched.columns = [SCHED_PREFIX + c for c in sched.columns]

    out = actual.join(sched)
    out["kbo_cancelled"] = games.loc[games["cancel"]].groupby("Date").size()
    zero = [c for c in out.columns if "start" not in c]
    out[zero] = out[zero].fillna(0.0)
    starts = [c for c in out.columns if "start" in c]
    out[starts] = out[starts].fillna(-1.0)
    out.index.name = "Date"
    return out.reset_index()[["Date", *DAILY_COLS, *[SCHED_PREFIX + c for c in AGG_COLS]]]


def load_daily(dates, source_dir=None,
               as_of: pd.Timestamp | None = None,
               use_live: bool = True) -> pd.DataFrame:
    """주어진 날짜 인덱스에 맞춘 KBO 일자별 피처.

    실측 일정 -> 캐시 CSV -> 휴리스틱 순으로 시도한다. 일정에 없는 날(비시즌·미편성)은
    '경기 없음'으로 채우고, 예측 지평이 수집된 일정보다 미래로 뻗는 구간만 휴리스틱으로
    메운다.

    `as_of` 를 주면 그 날짜 이후의 **우천취소 결과를 0으로 지운다.** 원본 JSON 에는
    경기가 끝난 뒤의 취소 여부가 들어 있어서, 그대로 쓰면 백테스트가 "내일 비가 와서
    경기가 취소된다"는 걸 미리 아는 셈이 된다. 예측/백테스트는 반드시 `as_of` 를 넘길 것.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize()

    sched: pd.DataFrame | None = None
    try:
        sched = parse_source(source_dir)
    except (FileNotFoundError, ValueError):
        if KBO_SCHEDULE_CSV.exists():
            sched = pd.read_csv(KBO_SCHEDULE_CSV, parse_dates=["Date"])

    if sched is None or sched.empty:
        return _heuristic(idx)

    covered_max = pd.DatetimeIndex(sched["Date"]).max()
    aligned = sched.set_index("Date").reindex(idx)

    beyond = idx > covered_max
    if beyond.any():
        fb = None
        if use_live:
            try:
                live = fetch_live_daily(idx[beyond].min(), idx[beyond].max())
                if not live.empty:
                    fb = live.set_index("Date").reindex(idx[beyond])
            except Exception:                       # noqa: BLE001  (네트워크·API 장애)
                fb = None
        if fb is None or fb["kbo_games"].isna().all():
            fb = _heuristic(idx[beyond]).set_index("Date")
        else:
            # 실시간에 없는 날(=편성 없음)은 0 으로
            miss = fb["kbo_games"].isna()
            if miss.any():
                zeros = _heuristic(fb.index[miss]).set_index("Date")
                zeros[[c for c in zeros.columns if c != "Date"]] = 0.0
                for c in zeros.columns:
                    if "start" in c:
                        zeros[c] = -1.0
                fb.loc[miss, zeros.columns] = zeros.to_numpy()
        fill_cols = [c for c in aligned.columns if c in fb.columns]
        aligned.loc[beyond, fill_cols] = fb[fill_cols].to_numpy()

    aligned = aligned.fillna({c: 0.0 for c in aligned.columns if "start" not in c})
    aligned = aligned.fillna({c: -1.0 for c in aligned.columns if "start" in c})
    for c in ("kbo_first_start", "kbo_main_start"):
        aligned.loc[aligned[c] <= 0, c] = -1.0

    # 경기 수·시작시각 등은 **항상 편성 기준**으로 통일한다.
    # 학습 때만 '실제 진행된 경기 수'를 쓰면 추론 때와 변수의 의미가 달라져
    # (학습: 취소 반영된 사후값 / 추론: 편성값) 조용히 어긋난다.
    sched_cols = [SCHED_PREFIX + c for c in AGG_COLS]
    if set(sched_cols).issubset(aligned.columns):
        aligned[AGG_COLS] = aligned[sched_cols].to_numpy()

    # 취소는 그날이 돼야 아는 값이라 예측 시점 이후로는 0(=아직 모름)으로 둔다.
    if as_of is not None:
        aligned.loc[idx > pd.Timestamp(as_of), UNKNOWABLE_AHEAD] = 0.0

    aligned.index.name = "Date"
    return aligned.reset_index()[["Date", *DAILY_COLS]]
