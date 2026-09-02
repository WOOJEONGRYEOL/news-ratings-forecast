"""롤링 오리진 백테스트.

'기준일 C에 서서 C+1 ~ C+H 를 예측한다'는 실제 운용 상황을 그대로 재현한다.
C 이후 데이터는 학습에도 피처 계산에도 절대 들어가지 않는다.
베이스라인(전일값·지난주 동요일·최근 4주 동요일 평균) 대비 개선폭까지 함께 낸다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_TRAIN_DAYS, TARGET_COLS
from .features import build_features, feature_columns
from .model import fit_target, training_frame

# KBO 블록은 성격이 다른 두 덩어리로 갈린다. 뭉뚱그려 재면 결론이 상쇄된다.
#  - kbo_schedule: 편성 자체. (요일, 월)로 거의 결정돼 달력 피처와 공선적이다.
#  - kbo_shock:    우천취소. 같은 요일·같은 계절 안에서 변하는 진짜 외생 충격이다.
#    (다만 취소는 비 때문에 생기고 비는 그 자체로 재택 시청을 늘리므로,
#     이 변수의 효과는 '야구 이탈 + 강우' 합이지 야구만의 효과가 아니다.)
KBO_SHOCK_COLS = {"kbo_cancelled"}

FEATURE_GROUPS = {
    "kbo": lambda c: c.startswith("kbo_"),
    "kbo_schedule": lambda c: c.startswith("kbo_") and c not in KBO_SHOCK_COLS,
    "kbo_shock": lambda c: c in KBO_SHOCK_COLS,
    "astro": lambda c: c in {"sunset_hour", "daylight_hours", "temp_proxy", "cold_and_dark"},
    "holiday": lambda c: c.startswith(("is_holiday", "consecutive_off", "days_to_holiday",
                                       "days_from_holiday", "off_day_index"))
                         or c in {"is_big_holiday", "is_bridge_day"},
    "slot": lambda c: ("slot_share" in c) or c.endswith("_rivals_hh") or
                      c.endswith("_rivals_2049") or c.endswith("rival_weight"),
    # 편성 파생 중 '있으면 좋을 것 같은' 것들 — 실제로 기여하는지 따로 잰다
    "sched_extra": lambda c: c.endswith(("_air_end", "_air_len", "_days_since_change"))
                             or "_apart_" in c,
    "event": lambda c: c.startswith(("is_news_event", "news_event_day", "event_")),
    # 유형별 이벤트 변수. 양성 일수가 5~21일뿐이라 min_child_samples=20 에 걸려
    # 어느 모델도 split 하지 못한다(실측 확인). 모델에선 빼고 What-If 는 실측 배율로 간다.
    "event_type": lambda c: c.startswith("event_"),
    # 기상 전체 (실측/예보 공통)
    "weather": lambda c: c.startswith(("wx_", "precip_", "is_rainy"))
                         or c in {"temp_effective", "kbo_rain_risk", "cold_and_dark"},
    # 비 x 야구 = 취소 위험. 사전 예측이 우천취소를 간접적으로 잡는 유일한 통로다.
    "rain_risk": lambda c: c == "kbo_rain_risk",
    # 근거가 얇은 기상 파생. 남길 것은 kbo_rain_risk(취소 통로)와
    # temp_effective(가짜 계절근사치를 실측으로 대체) 둘뿐이다.
    "weather_extra": lambda c: (c.startswith(("wx_", "precip_", "is_rainy"))
                                and c != "kbo_rain_risk"),
}


def _cutoffs(dates: pd.Series, n_folds: int, horizon: int, step: int | None) -> list[pd.Timestamp]:
    """마지막에서부터 step 간격으로 떨어진 기준일들."""
    step = step or horizon
    last = dates.max()
    first_allowed = dates.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    outs = []
    for i in range(n_folds):
        c = last - pd.Timedelta(days=horizon + i * step)
        if c < first_allowed:
            break
        outs.append(c)
    return sorted(outs)


def _baselines(df: pd.DataFrame, cutoff: pd.Timestamp, target: str,
               target_date: pd.Timestamp) -> dict[str, float]:
    hist = df.loc[df["Date"] <= cutoff].set_index("Date")[target]
    same_dow = [target_date - pd.Timedelta(days=7 * k) for k in range(1, 5)]
    same_dow = [hist.get(d, np.nan) for d in same_dow]
    return {
        "naive_last": float(hist.iloc[-1]) if len(hist) else np.nan,
        "snaive_7": float(same_dow[0]) if not pd.isna(same_dow[0]) else np.nan,
        "dow_mean_4": float(np.nanmean(same_dow)) if np.any(~pd.isna(same_dow)) else np.nan,
    }


def _future_row(hist: pd.DataFrame, cutoff: pd.Timestamp, h: int,
                events, kbo_source, as_of: pd.Timestamp | None = None,
                observed_weather: bool = False) -> pd.DataFrame:
    """cutoff 기준 D+h 시점의 피처 1행 (타깃은 비운 채로 계산).

    `as_of` 는 **외생 변수의 관측 시점**만 정한다. 시청률 시차 피처는 언제나
    cutoff 까지만 쓴다(horizon 이 통제). 따라서 as_of 를 대상일로 올려도
    타깃 누수는 생기지 않고, '그날 낮에 우천취소가 확정된 뒤 예측을 고친다'는
    상황만 재현된다.
    """
    future = pd.DataFrame({"Date": pd.date_range(cutoff + pd.Timedelta(days=1),
                                                 periods=h, freq="D")})
    for t in TARGET_COLS:
        future[t] = np.nan
    future["is_imputed"] = True

    full = pd.concat([hist, future], ignore_index=True)
    frame = build_features(full, horizon=h, events=events, kbo_source=kbo_source,
                           as_of=as_of if as_of is not None else cutoff,
                           observed_weather=observed_weather)
    return frame.loc[frame["Date"] == cutoff + pd.Timedelta(days=h)]


def run(df: pd.DataFrame, horizon: int = 7, n_folds: int = 12, step: int | None = None,
        targets=None, events=None, kbo_source=None, objective: str = "l1",
        drop_groups: tuple[str, ...] = (), half_life: float | None = None,
        same_day: bool = False, progress=None) -> pd.DataFrame:
    """폴드 x 지평 x 타깃 예측 결과를 긴 형식으로 반환.

    same_day=False  사전 예측 — 기준일에 서서 D+h 를 낸다. 그날의 우천취소는 모른다.
    same_day=True   당일 수정 — 대상일 낮에 취소가 확정된 뒤 예측을 고친다.
                    시청률 시차는 그대로 기준일까지만 쓴다. 두 모드의 차이가
                    '취소를 아는 것의 가치'다.
    """
    targets = list(targets or TARGET_COLS)
    cutoffs = _cutoffs(df["Date"], n_folds, horizon, step)
    if not cutoffs:
        raise ValueError("학습 구간이 부족해 백테스트를 만들 수 없습니다.")

    drop_fns = [FEATURE_GROUPS[g] for g in drop_groups if g in FEATURE_GROUPS]
    records = []
    total, done = len(cutoffs) * horizon, 0

    for cutoff in cutoffs:
        hist = df.loc[df["Date"] <= cutoff].reset_index(drop=True)
        for h in range(1, horizon + 1):
            done += 1
            if progress is not None:
                progress(done / total, f"{cutoff.date()} · D+{h}")

            target_date = cutoff + pd.Timedelta(days=h)
            actual = df.loc[df["Date"] == target_date]
            if actual.empty or bool(actual["is_imputed"].iat[0]):
                continue    # 관측이 없던 날은 채점 대상에서 제외

            train_frame = training_frame(hist, h, events=events, kbo_source=kbo_source)
            if len(train_frame) < 60:
                continue

            feats = feature_columns(train_frame)
            if drop_fns:
                feats = [c for c in feats if not any(fn(c) for fn in drop_fns)]

            X = _future_row(hist, cutoff, h, events, kbo_source,
                            as_of=target_date if same_day else cutoff,
                            observed_weather=same_day)
            if X.empty or bool(X[feats].isna().any(axis=1).iat[0]):
                continue
            Xf = X[feats].astype(float)

            for t in targets:
                kw = {} if half_life is None else {"half_life": half_life}
                m = fit_target(train_frame, t, h, feats, with_quantiles=False,
                               objective=objective, **kw)
                records.append({
                    "cutoff": cutoff, "Date": target_date, "horizon": h, "target": t,
                    "actual": float(actual[t].iat[0]),
                    "model": max(float(m.predict(Xf)[0]), 0.0),
                    **_baselines(df, cutoff, t, target_date),
                })

    return pd.DataFrame(records)


def _errors(res: pd.DataFrame, col: str) -> dict[str, float]:
    err = res[col] - res["actual"]
    denom = res["actual"].replace(0, np.nan)
    return {
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE%": float((err.abs() / denom).mean() * 100),
        "bias": float(err.mean()),
    }


def summarize(res: pd.DataFrame, by: tuple[str, ...] = ("target",)) -> pd.DataFrame:
    """타깃(또는 지평)별 모델·베이스라인 오차와 개선율."""
    method_cols = ["model", "naive_last", "snaive_7", "dow_mean_4"]
    rows = []
    for keys, grp in res.groupby(list(by), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(by, keys))
        rec["n"] = len(grp)
        for c in method_cols:
            if c not in grp.columns:
                continue
            sub = grp.dropna(subset=[c])
            if sub.empty:
                continue
            e = _errors(sub, c)
            rec[f"{c}_MAE"] = round(e["MAE"], 4)
            rec[f"{c}_RMSE"] = round(e["RMSE"], 4)
        if "model_MAE" in rec and "snaive_7_MAE" in rec:
            rec["vs_snaive7_MAE%"] = round(
                (1 - rec["model_MAE"] / max(rec["snaive_7_MAE"], 1e-9)) * 100, 1)
        if "model_MAE" in rec and "dow_mean_4_MAE" in rec:
            rec["vs_dow4_MAE%"] = round(
                (1 - rec["model_MAE"] / max(rec["dow_mean_4_MAE"], 1e-9)) * 100, 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def ablation(df: pd.DataFrame, groups=("kbo", "astro", "holiday", "slot"),
             by_target: bool = False, **kw) -> pd.DataFrame:
    """피처 그룹을 하나씩 빼면서 오차가 얼마나 달라지는지 측정.

    순열 중요도('모델이 얼마나 의존하는가')와 결론이 갈릴 수 있다. 이쪽이
    '표본 밖에서 실제로 도움이 되는가'에 대한 답이다. 악화폭이 음수면
    그 그룹은 빼는 편이 낫다는 뜻이다.
    """
    full = run(df, **kw)
    base_mae = _errors(full, "model")["MAE"]
    rows = [{"제외 그룹": "(없음 · 전체 피처)", "MAE": round(base_mae, 4), "악화폭%": 0.0}]
    per_target = [full.assign(변형="(없음 · 전체 피처)")] if by_target else []

    for g in groups:
        res = run(df, drop_groups=(g,), **kw)
        mae = _errors(res, "model")["MAE"]
        rows.append({"제외 그룹": g, "MAE": round(mae, 4),
                     "악화폭%": round((mae / base_mae - 1) * 100, 2)})
        if by_target:
            per_target.append(res.assign(변형=g))

    summary = pd.DataFrame(rows).sort_values("악화폭%", ascending=False).reset_index(drop=True)
    if not by_target:
        return summary

    # 타깃별로 어느 그룹이 도움이 되는지 갈리는지 확인한다
    allres = pd.concat(per_target, ignore_index=True)
    allres["abs_err"] = (allres["model"] - allres["actual"]).abs()
    wide = allres.pivot_table(index="target", columns="변형", values="abs_err", aggfunc="mean")
    base_col = "(없음 · 전체 피처)"
    out = pd.DataFrame({"지표": wide.index, "전체 피처 MAE": wide[base_col].round(4)})
    for g in groups:
        if g in wide.columns:
            out[f"{g} 제외 악화폭%"] = ((wide[g] / wide[base_col] - 1) * 100).round(2).to_numpy()
    return summary, out.reset_index(drop=True)
