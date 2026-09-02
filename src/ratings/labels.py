"""피처 이름 -> 사람이 읽는 한글 설명.

피처가 조합으로 생성되므로(타깃 8개 × 시차 7종 × …) 사전만으로는 감당이 안 된다.
패턴 규칙 + 예외 사전으로 처리한다.
"""
from __future__ import annotations

import re

from .config import CHANNELS

# 채널 식별자 -> 짧은 표기
SHORT = {"channelA": "채널A", "jtbc": "JTBC", "mbn": "MBN", "tvchosun": "TV조선"}
_COL_SHORT = {**{c.household_col: f"{SHORT[c.key]}(가구)" for c in CHANNELS},
              **{c.col_2049: f"{SHORT[c.key]}(2049)" for c in CHANNELS}}

EXACT = {
    # --- 달력 ---
    "dayofweek": "요일 (0=월 … 6=일)",
    "is_weekend": "주말 여부",
    "is_friday": "금요일 여부",
    "is_monday": "월요일 여부",
    "month": "월",
    "day": "일",
    "dayofyear": "연중 며칠째",
    "weekofyear": "연중 몇 주째",
    "sin_year": "계절 위치 (사인)",
    "cos_year": "계절 위치 (코사인)",
    "sin_dow": "요일 위치 (사인)",
    "cos_dow": "요일 위치 (코사인)",
    # --- 공휴일 ---
    "is_holiday": "공휴일 여부",
    "is_holiday_eve": "공휴일 전날",
    "is_holiday_after": "공휴일 다음날",
    "is_big_holiday": "설날·추석 연휴 구간",
    "consecutive_off_days": "연휴 길이 (이어지는 휴일 수)",
    "off_day_index": "연휴 중 몇째 날",
    "is_bridge_day": "징검다리 평일",
    "days_to_holiday": "다음 공휴일까지 남은 일수",
    "days_from_holiday": "직전 공휴일로부터 지난 일수",
    # --- 천문·기상 ---
    "sunset_hour": "일몰 시각 (서울)",
    "daylight_hours": "낮 길이",
    "temp_proxy": "기온 계절 근사치",
    "temp_effective": "기온 (실측/예보)",
    "cold_and_dark": "추위 × 이른 일몰 (귀가 후 체류 지표)",
    "wx_precip": "하루 강수량",
    "wx_precip_evening": "저녁 18~21시 강수량",
    "wx_temp": "기온 (실측/예보)",
    "wx_cloud": "운량",
    "is_rainy": "비 오는 날",
    "is_rainy_evening": "저녁에 비",
    "precip_log": "하루 강수량 (로그)",
    "precip_evening_log": "저녁 강수량 (로그)",
    # --- 야구 ---
    "kbo_games": "야구 편성 경기 수",
    "kbo_cancelled": "야구 우천취소 경기 수",
    "kbo_any_game": "야구 경기 있는 날",
    "kbo_seoul_games": "서울 구장(잠실·고척) 경기 수",
    "kbo_first_start": "야구 최초 경기 시작 시각",
    "kbo_main_start": "야구 주력 경기 시작 시각",
    "kbo_games_19": "19시대에 진행 중인 야구 경기 수",
    "kbo_games_21": "21시대에 진행 중인 야구 경기 수",
    "kbo_overlap_19": "야구가 19시대를 덮는 시간",
    "kbo_overlap_21": "야구가 21시대를 덮는 시간",
    "kbo_pressure_19": "19시대 야구 압박 (서울 구장 가중)",
    "kbo_pressure_21": "21시대 야구 압박 (서울 구장 가중)",
    "kbo_is_postseason": "포스트시즌",
    "kbo_rain_risk": "우천취소 위험 (야구 편성 × 저녁 비 예보)",
    # --- 이벤트 ---
    "is_news_event": "속보 국면 (유형 무관)",
    "news_event_day": "속보 국면 며칠째",
    "event_윤석열_사법": "속보: 윤석열 사법 국면 (계엄·탄핵·내란재판)",
    "event_이재명_사법": "속보: 이재명 사법 국면 (체포동의안·영장)",
    "event_대선_정국": "속보: 대선 정국 (후보 단일화 등)",
    "all4_sum_hh": "종편 4사 시청률 합계 (가구)",
}

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?P<col>.+)_lag_(?P<n>\d+)$"), "{col} · {n}일 전 시청률"),
    (re.compile(r"^(?P<col>.+)_roll_mean_(?P<w>\d+)$"), "{col} · 최근 {w}일 평균"),
    (re.compile(r"^(?P<col>.+)_roll_std_(?P<w>\d+)$"), "{col} · 최근 {w}일 변동성"),
    (re.compile(r"^(?P<col>.+)_dow_mean_4$"), "{col} · 최근 4주 같은 요일 평균"),
    (re.compile(r"^(?P<col>.+)_momentum$"), "{col} · 주간 추세 (전주 대비)"),
    (re.compile(r"^(?P<col>.+)_vs_roll30$"), "{col} · 30일 평균 대비 편차"),
]

_CHANNEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?P<key>\w+?)_rivals_hh$"), "{ch} · 동시간대 경쟁사 시청률 합 (가구)"),
    (re.compile(r"^(?P<key>\w+?)_rivals_2049$"), "{ch} · 동시간대 경쟁사 시청률 합 (2049)"),
    (re.compile(r"^(?P<key>\w+?)_slot_share_hh$"), "{ch} · 동시간대 점유율 (가구)"),
    (re.compile(r"^(?P<key>\w+?)_slot_share_2049$"), "{ch} · 동시간대 점유율 (2049)"),
    (re.compile(r"^(?P<key>\w+?)_apart_hh$"), "{ch} · 안 겹치는 채널들 합 (가구)"),
    (re.compile(r"^(?P<key>\w+?)_apart_2049$"), "{ch} · 안 겹치는 채널들 합 (2049)"),
    (re.compile(r"^(?P<key>\w+?)_rival_weight$"), "{ch} · 동시간대 경쟁 강도"),
    (re.compile(r"^(?P<key>\w+?)_air_hour$"), "{ch} · 방송 시작 시각"),
    (re.compile(r"^(?P<key>\w+?)_air_end$"), "{ch} · 방송 종료 시각"),
    (re.compile(r"^(?P<key>\w+?)_air_len$"), "{ch} · 방송 길이"),
    (re.compile(r"^(?P<key>\w+?)_days_since_change$"), "{ch} · 편성 변경 후 경과일"),
    (re.compile(r"^(?P<key>\w+?)_kbo_overlap$"), "{ch} · 야구 중계와 겹치는 비율"),
    (re.compile(r"^(?P<key>\w+?)_kbo_pressure$"), "{ch} · 야구 압박 (서울 구장 가중)"),
    (re.compile(r"^(?P<key>\w+?)_youth_ratio$"), "{ch} · 2049 비중"),
]

GROUP_LABELS = {
    "kbo": "야구 전체",
    "kbo_schedule": "야구 편성 (경기 수·시작 시각)",
    "kbo_shock": "야구 우천취소",
    "astro": "일몰·기온",
    "holiday": "공휴일·연휴",
    "slot": "동시간대 경쟁",
    "sched_extra": "편성 파생 (방송 길이·경과일 등)",
    "event": "대형 속보 국면",
    "weather": "기상 전체",
    "weather_extra": "기상 파생",
    "rain_risk": "우천취소 위험",
    "(없음 · 전체 변수)": "(없음 · 전체 변수)",
}


def humanize(feature: str) -> str:
    """피처 이름 하나를 한글 설명으로."""
    if feature in EXACT:
        return EXACT[feature]

    for pattern, template in _PATTERNS:
        m = pattern.match(feature)
        if m:
            col = m.group("col")
            pretty = _COL_SHORT.get(col, col)
            return template.format(col=pretty, **{k: v for k, v in m.groupdict().items()
                                                  if k != "col"})

    for pattern, template in _CHANNEL_PATTERNS:
        m = pattern.match(feature)
        if m and m.group("key") in SHORT:
            return template.format(ch=SHORT[m.group("key")])

    return feature        # 못 알아본 건 원래 이름 그대로


def humanize_group(group: str) -> str:
    return GROUP_LABELS.get(group, group)


TARGET_LABELS = {
    **{c.household_col: f"{c.label} · 가구" for c in CHANNELS},
    **{c.col_2049: f"{c.label} · 2049" for c in CHANNELS},
}


def humanize_target(col: str) -> str:
    """'JTBC뉴스룸(2049)' -> 'JTBC 뉴스룸 · 2049'."""
    return TARGET_LABELS.get(col, col)
