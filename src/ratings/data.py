"""시청률 원본 로드 · 정제 · 결측일 보정."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATE_COL, RATINGS_CSV, TARGET_COLS


def _parse_dates(raw: pd.Series) -> pd.Series:
    """'230701' / '20230701' / '2023-07-01' 등 혼재된 날짜 표기를 모두 흡수."""
    s = raw.astype(str).str.strip().str.replace(r"[^\d]", "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    six = s.str.len() == 6
    if six.any():
        out.loc[six] = pd.to_datetime(s[six], format="%y%m%d", errors="coerce")
    eight = s.str.len() == 8
    if eight.any():
        out.loc[eight] = pd.to_datetime(s[eight], format="%Y%m%d", errors="coerce")

    rest = out.isna()
    if rest.any():
        out.loc[rest] = pd.to_datetime(raw[rest], errors="coerce")
    return out


def load_ratings(path: str | Path | None = None) -> pd.DataFrame:
    """CSV/XLSX -> 일자 연속 · 결측 보간된 시청률 프레임.

    반환 컬럼: Date, 8개 타깃, is_imputed(그 날 관측이 실제로 없었는지)
    """
    path = Path(path) if path is not None else RATINGS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"시청률 데이터가 없습니다: {path}\n"
            f"  → 원본 CSV를 {path} 위치에 두거나 --data 로 경로를 지정하세요."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise UnicodeDecodeError("csv", b"", 0, 1, f"인코딩 판별 실패: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    if DATE_COL not in df.columns:
        raise KeyError(f"'{DATE_COL}' 컬럼이 없습니다. 실제 컬럼: {list(df.columns)}")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"타깃 컬럼 누락: {missing}\n실제 컬럼: {list(df.columns)}")

    df["Date"] = _parse_dates(df[DATE_COL])
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date").reset_index(drop=True)

    for col in TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 메인 뉴스가 정확히 0.000 으로 찍히는 건 시청률이 아니라 결방/미집계다.
    # (예: 2026-05-12 JTBC 뉴스룸) 0을 실측으로 두면 모델이 허구의 급락을 학습한다.
    df[TARGET_COLS] = df[TARGET_COLS].mask(df[TARGET_COLS] <= 0.0)

    # 달력상 빠진 날짜를 채워 일자 연속성 확보 (lag_7 등이 요일과 어긋나지 않게)
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    observed = set(df["Date"])
    df = df.set_index("Date").reindex(full_range)
    df.index.name = "Date"

    # 관측 자체가 없던 날 + 값이 비어 있던 날을 모두 보정 대상으로 표시
    df["is_imputed"] = (~df.index.isin(observed)) | df[TARGET_COLS].isna().any(axis=1)
    df[TARGET_COLS] = df[TARGET_COLS].interpolate(method="linear", limit_direction="both")

    return df.reset_index()[["Date", *TARGET_COLS, "is_imputed"]]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """채널별 평일/주말 평균과 결측 보정 현황 요약."""
    weekend = df["Date"].dt.dayofweek >= 5
    rows = []
    for col in TARGET_COLS:
        wd, we = df.loc[~weekend, col], df.loc[weekend, col]
        rows.append({
            "지표": col,
            "평균": round(df[col].mean(), 4),
            "평일평균": round(wd.mean(), 4),
            "주말평균": round(we.mean(), 4),
            "주말낙폭%": round((we.mean() / wd.mean() - 1) * 100, 1),
            "최소": round(df[col].min(), 3),
            "최대": round(df[col].max(), 3),
        })
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame) -> dict:
    return {
        "start": df["Date"].min().date().isoformat(),
        "end": df["Date"].max().date().isoformat(),
        "days": int(len(df)),
        "imputed_days": int(df["is_imputed"].sum()),
        "imputed_dates": [d.date().isoformat() for d in df.loc[df["is_imputed"], "Date"]],
    }


def outlier_dates(df: pd.DataFrame, col: str, z: float = 3.0, window: int = 60) -> pd.Series:
    """이동 로버스트 z-score 기준 급등/급락일 (대형 이슈 플래그 자동 생성용)."""
    s = df.set_index("Date")[col]
    med = s.rolling(window, center=True, min_periods=15).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=15).median()
    score = (s - med) / (1.4826 * mad.replace(0, np.nan))
    return score[score.abs() > z]
