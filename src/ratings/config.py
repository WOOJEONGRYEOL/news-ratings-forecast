"""종편 4사 메인뉴스 시청률 예측 - 전역 설정."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RATINGS_CSV = DATA_DIR / "종편_4사_메인_시청률.csv"
KBO_SCHEDULE_CSV = DATA_DIR / "kbo_schedule.csv"

# KBO 일정 원본. 자체 수집본을 우선하고, kbo-forecast 프로젝트 수집본이 있으면
# 함께 읽어 보완한다(같은 네이버 API 소스라 형식이 동일하다).
# 자체 수집이 있으면 외부 프로젝트가 멈춰도 문제없다 — scripts/sync_kbo.py 참고.
KBO_CACHE_DIR = DATA_DIR / "kbo_games"
KBO_SOURCE_DIR = Path.home() / "kbo-forecast" / "data"
KBO_DIRS = (KBO_SOURCE_DIR, KBO_CACHE_DIR)   # 뒤쪽이 우선(중복 시 keep="last")

DATE_COL = "날짜"


def sheet_id() -> str:
    """시청률 구글 시트 ID.

    코드에 박아두지 않는다 — 이 ID 하나면 원본 시트를 통째로 내려받을 수 있다.
    찾는 순서: 환경변수 → 로컬 파일(git 제외) → Streamlit secrets.
    """
    import os

    env = os.environ.get("RATINGS_SHEET_ID")
    if env:
        return env.strip()

    local = DATA_DIR / "sheet_id.txt"
    if local.exists():
        return local.read_text(encoding="utf-8").strip()

    try:
        import streamlit as st
        return str(st.secrets["ratings_sheet_id"]).strip()
    except Exception:                                  # noqa: BLE001
        pass

    raise RuntimeError(
        "시청률 시트 ID 를 찾을 수 없습니다. 다음 중 하나로 지정하세요.\n"
        "  - 환경변수  RATINGS_SHEET_ID=<시트ID>\n"
        f"  - 파일      {DATA_DIR / 'sheet_id.txt'}\n"
        '  - Streamlit secrets   ratings_sheet_id = "<시트ID>"'
    )


def hm(hour: int, minute: int = 0) -> float:
    """18:55 -> 18.9167 (24시간 실수)."""
    return hour + minute / 60.0


@dataclass(frozen=True)
class SlotPeriod:
    """한 채널이 특정 시점부터 쓰던 편성 (시작일 포함).

    평일/주말이 다르고 방송 길이도 채널마다 다르다 (평일 55~70분, 주말 35~60분).
    MBN 주말은 '뉴스센터', TV조선 주말은 '뉴스7' 로 아예 다른 프로그램이다.
    """

    start: str                       # YYYY-MM-DD, 이 날부터 적용
    weekday: tuple[float, float]     # (시작, 종료)
    weekend: tuple[float, float]
    note: str = ""

    def window(self, is_weekend: bool) -> tuple[float, float]:
        return self.weekend if is_weekend else self.weekday


@dataclass(frozen=True)
class Channel:
    """방송사 메인뉴스 메타데이터."""

    key: str
    label: str
    household_col: str
    col_2049: str
    schedule: tuple[SlotPeriod, ...]     # 편성 이력 (시작일 오름차순)
    color: str = "#888888"               # 차트 색 (밝은/어두운 테마 모두 대비 확보)

    def at(self, when) -> SlotPeriod:
        ts = pd.Timestamp(when)
        current = self.schedule[0]
        for period in self.schedule:
            if ts >= pd.Timestamp(period.start):
                current = period
        return current

    def window(self, when) -> tuple[float, float]:
        ts = pd.Timestamp(when)
        return self.at(ts).window(ts.dayofweek >= 5)

    def air_hour(self, when) -> float:
        return self.window(when)[0]


# --- 편성 이력 -------------------------------------------------------------
# 출처: 사용자 제공(2026-08). 현재 구간은 실측 방송 시각, 과거 구간은 변경 이력의
# 시작 시각에 해당 채널의 현재 방송 길이를 적용해 재구성했다.
#
# 이력과 실측이 5분씩 어긋나는 곳은 **실측을 채택**했다 (사용자 지시).
#   - 채널A 평일: 이력 "19:00 변동 없음" → 실측 18:55~19:55
#   - JTBC 평일: 이력 24.12.16 부터 18:30 → 실측 18:35~19:45
#   - JTBC 주말: 이력에 25.02.08 부터 18:20 이 있으나 실측은 18:25~19:00.
#                25.01.11 값(18:25)과 같아 별도 구간을 두지 않았다.
#   - 채널A 주말 26.01.10 개편: 나무위키 18:30/18:25 불일치를 실측 18:25 로 확정
#
# 과거 구간의 **방송 길이**는 현재 길이를 그대로 적용했다 (종료 시각 미확인).
CHANNELS: tuple[Channel, ...] = (
    Channel("channelA", "채널A 뉴스A", "뉴스A", "뉴스A(2049)", (
        SlotPeriod("2000-01-01", (hm(18, 55), hm(19, 55)), (hm(19, 0), hm(19, 35))),
        SlotPeriod("2026-01-10", (hm(18, 55), hm(19, 55)), (hm(18, 25), hm(19, 0)),
                   "주말 18:25 로 이동"),
    ), color="#2563EB"),        # 블루
    Channel("jtbc", "JTBC 뉴스룸", "JTBC뉴스룸", "JTBC뉴스룸(2049)", (
        SlotPeriod("2000-01-01", (hm(19, 50), hm(21, 0)), (hm(18, 0), hm(18, 35))),
        SlotPeriod("2024-03-11", (hm(18, 50), hm(20, 0)), (hm(18, 40), hm(19, 15)),
                   "평일 18:50 / 주말 18:40"),
        SlotPeriod("2024-07-01", (hm(18, 50), hm(20, 0)), (hm(18, 30), hm(19, 5)),
                   "주말 18:30 ('2024년 7월' 로만 확인돼 1일로 근사"),
        SlotPeriod("2024-12-16", (hm(18, 35), hm(19, 45)), (hm(18, 30), hm(19, 5)),
                   "평일 18:35"),
        SlotPeriod("2025-01-11", (hm(18, 35), hm(19, 45)), (hm(18, 25), hm(19, 0)),
                   "주말 18:25 — 실측 기준 현재까지 유지"),
    ), color="#A855F7"),        # 바이올렛
    Channel("mbn", "MBN 뉴스7", "MBN뉴스7", "MBN뉴스7(2049)", (
        # 주말은 '뉴스센터' 라는 별개 프로그램이 19:30 에 나간다
        SlotPeriod("2000-01-01", (hm(19, 0), hm(20, 0)), (hm(19, 30), hm(20, 5))),
    ), color="#F97316"),        # 오렌지
    Channel("tvchosun", "TV조선 뉴스9", "TV조선뉴스9", "TV조선뉴스9(2049)", (
        # 주말은 '뉴스7' 이라는 별개 프로그램. 평일엔 21시대 단독이지만
        # 주말엔 채널A·JTBC 와 정면으로 붙는다.
        SlotPeriod("2000-01-01", (hm(21, 0), hm(21, 55)), (hm(18, 57), hm(19, 57))),
    ), color="#DC2626"),        # 레드
)

HOUSEHOLD_COLS = [c.household_col for c in CHANNELS]
COLS_2049 = [c.col_2049 for c in CHANNELS]
TARGET_COLS = HOUSEHOLD_COLS + COLS_2049

# 화면 어디서나 같은 색을 쓰도록 한 곳에서 정의한다
PALETTE = {c.label: c.color for c in CHANNELS}
PALETTE_BY_KEY = {c.key: c.color for c in CHANNELS}

BY_HOUSEHOLD = {c.household_col: c for c in CHANNELS}
BY_2049 = {c.col_2049: c for c in CHANNELS}
BY_TARGET = {**BY_HOUSEHOLD, **BY_2049}


def channel_of(target: str) -> Channel:
    return BY_TARGET[target]


def is_2049(target: str) -> bool:
    return target in BY_2049


def schedule_frame(dates) -> pd.DataFrame:
    """날짜별 편성표. 컬럼: {key}_air_hour, {key}_air_end, {key}_air_len,
    {key}_days_since_change."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).normalize()
    weekend = idx.dayofweek >= 5
    out = pd.DataFrame(index=idx)

    for ch in CHANNELS:
        start = np.empty(len(idx))
        end = np.empty(len(idx))
        since = np.full(len(idx), 999.0)
        for period in ch.schedule:
            eff = pd.Timestamp(period.start)
            mask = idx >= eff
            if not mask.any():
                continue
            start[mask] = np.where(weekend[mask], period.weekend[0], period.weekday[0])
            end[mask] = np.where(weekend[mask], period.weekend[1], period.weekday[1])
            if eff > idx.min():
                since[mask] = np.clip((idx[mask] - eff).days, 0, 999)
        out[f"{ch.key}_air_hour"] = start
        out[f"{ch.key}_air_end"] = end
        out[f"{ch.key}_air_len"] = end - start
        out[f"{ch.key}_days_since_change"] = np.minimum(since, 365)

    out.index.name = "Date"
    return out


def overlap_matrix(dates) -> dict[tuple[str, str], np.ndarray]:
    """날짜별 채널쌍 방송 구간 겹침 (내 방송 시간 중 상대와 겹치는 비율, 0~1).

    19/21시대라는 거친 버킷 대신 실제 구간을 쓴다. 비대칭이다 — 짧은 프로그램이
    긴 프로그램에 완전히 먹히면 전자는 1.0, 후자는 그보다 작다.
    """
    sched = schedule_frame(dates)
    out = {}
    for i in CHANNELS:
        a0 = sched[f"{i.key}_air_hour"].to_numpy()
        a1 = sched[f"{i.key}_air_end"].to_numpy()
        span = np.maximum(a1 - a0, 1e-6)
        for j in CHANNELS:
            if i.key == j.key:
                continue
            b0 = sched[f"{j.key}_air_hour"].to_numpy()
            b1 = sched[f"{j.key}_air_end"].to_numpy()
            inter = np.clip(np.minimum(a1, b1) - np.maximum(a0, b0), 0.0, None)
            out[(i.key, j.key)] = inter / span
    return out


# KBO 중계권 체제 변경.
# 2006~2023: 네이버 컨소시엄 무료 온라인 중계
# 2024~2026: 티빙 유료 독점 (2024 정규시즌 개막 3/23)
#
# 유료화로 '티빙을 안 보던 시청자가 유선 스포츠 채널로 이동'하면서, 야구가 TV 뉴스를
# 잠식하는 구조 자체가 달라졌다. 그래서 이 날 이전 구간에서는 야구 변수를 결측으로 둔다
# (0 으로 두면 '경기가 없었다'는 다른 거짓말이 된다).
KBO_ERA_START = "2024-03-23"

# --- 피처 생성 파라미터 -------------------------------------------------
LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLL_WINDOWS = (7, 14, 30)
MIN_TRAIN_DAYS = 180        # 백테스트 최소 학습 구간
DEFAULT_HORIZON = 7         # 기본 예측 지평 (일)
QUANTILES = (0.1, 0.9)      # 예측 구간
