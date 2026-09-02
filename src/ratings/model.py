"""LightGBM 기반 (타깃 × 예측지평) 모델 학습.

- 지평별 직접 예측(direct multi-horizon): h일 뒤를 예측하는 전용 모델을 따로 학습한다.
  재귀 예측처럼 오차가 누적되지 않고, 각 지평에서 실제 사용 가능한 정보만 쓴다.
- 보간으로 채운 날은 학습 타깃에서 제외한다(피처 계산에는 그대로 사용).
- 최근 데이터에 지수 가중을 줘 편성·시청 환경 변화를 따라가게 한다.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import QUANTILES, TARGET_COLS
from .features import build_features, feature_columns

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:                                    # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LGBM = False

# 표본 가중 반감기(일). None = 균등 가중.
# 2024Q4 국면 변화 때문에 최근 데이터에 무게를 실어야 할 것 같지만, 측정해 보면 반대다.
#   균등 0.1449 < 반감기 400일 0.1485 < 반감기 180일 0.1500  (6폴드 롤링 오리진 MAE)
# 표본이 1,100행뿐이라 과거를 깎아 유효 표본을 줄이는 손해가 더 크고, 수준 변화는 이미
# lag/rolling 피처가 흡수한다. 재측정은 `scripts/tune.py`.
HALF_LIFE_DAYS: float | None = None
BASE_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=15,          # 표본이 ~1,100행뿐이라 얕게 유지
    min_child_samples=20,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.65,
    reg_alpha=0.1,
    reg_lambda=1.0,
    verbose=-1,
    n_jobs=1,               # 이 크기 데이터에선 스레드 동기화 비용이 더 크다
    random_state=42,
)


@dataclass
class TargetModel:
    """단일 (타깃, 지평) 예측기 + 선택적 분위 모델."""

    target: str
    horizon: int
    feature_cols: list[str]
    estimator: object
    quantiles: dict[float, object] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(X[self.feature_cols]), dtype=float)

    def predict_interval(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        return {q: np.asarray(m.predict(X[self.feature_cols]), dtype=float)
                for q, m in self.quantiles.items()}


def _make_estimator(objective: str = "l1", alpha: float | None = None, **over):
    params = {**BASE_PARAMS, **over}
    if HAS_LGBM:
        if alpha is not None:
            return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
        return lgb.LGBMRegressor(objective=objective, **params)

    # LightGBM 미설치 환경 폴백 (libomp 없는 macOS 등)
    from sklearn.ensemble import HistGradientBoostingRegressor
    loss = "quantile" if alpha is not None else \
        {"l1": "absolute_error", "l2": "squared_error"}.get(objective, "squared_error")
    kw = dict(max_iter=params["n_estimators"], learning_rate=params["learning_rate"],
              max_leaf_nodes=params["num_leaves"], min_samples_leaf=params["min_child_samples"],
              l2_regularization=params["reg_lambda"], random_state=params["random_state"])
    if alpha is not None:
        kw["quantile"] = alpha
    return HistGradientBoostingRegressor(loss=loss, **kw)


def sample_weights(dates: pd.Series,
                   half_life: float | None = HALF_LIFE_DAYS) -> np.ndarray | None:
    """최근일수록 큰 가중치 (지수 감쇠). half_life 가 None 이면 균등 가중."""
    if half_life is None:
        return None
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-np.log(2.0) * age / half_life)


def training_frame(df: pd.DataFrame, horizon: int, events=None,
                   kbo_source=None) -> pd.DataFrame:
    """지평 h 용 피처 프레임 (학습 가능한 행만).

    전부 과거 행이라 `as_of` 를 마지막 관측일로 둔다 — 아무것도 지워지지 않는다.
    """
    frame = build_features(df, horizon=horizon, events=events, kbo_source=kbo_source,
                           as_of=df["Date"].max())
    feats = feature_columns(frame)
    # 야구 변수는 중계권 체제 이전 구간에서 의도적으로 결측이다. 이걸로 행을 버리면
    # 2023 년치가 통째로 날아간다 — LightGBM 이 결측을 직접 처리하도록 남겨 둔다.
    required = [c for c in feats if not (c.startswith("kbo_") or "_kbo_" in c)]
    return frame.dropna(subset=required).reset_index(drop=True)


def fit_target(frame: pd.DataFrame, target: str, horizon: int,
               feature_cols: list[str] | None = None,
               with_quantiles: bool = True,
               objective: str = "l1",
               half_life: float | None = HALF_LIFE_DAYS) -> TargetModel:
    feature_cols = feature_cols or feature_columns(frame)

    fit_rows = frame
    if "is_imputed" in frame.columns:
        fit_rows = frame.loc[~frame["is_imputed"].astype(bool)]
    fit_rows = fit_rows.dropna(subset=[target])

    X, y = fit_rows[feature_cols], fit_rows[target].to_numpy(dtype=float)
    w = sample_weights(fit_rows["Date"], half_life)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = _make_estimator(objective=objective)
        est.fit(X, y, sample_weight=w)

        qmodels = {}
        if with_quantiles:
            for q in QUANTILES:
                qm = _make_estimator(alpha=q)
                qm.fit(X, y, sample_weight=w)
                qmodels[q] = qm

    return TargetModel(target=target, horizon=horizon, feature_cols=list(feature_cols),
                       estimator=est, quantiles=qmodels)


# 기본으로 빼는 피처 그룹.
#
# `sched_extra` = 편성 파생 20개 (air_end / air_len / days_since_change / _apart_).
# 실측 편성 구조로 바꾸면서 피처가 181 → 225 개로 늘자 MAE 가 0.1577 → 0.1603 으로
# 밀렸다. 이 20개를 덜어내니 0.1586 으로 회복돼 고정 슬롯 모델과 통계적 동률이 된다
# (CI [-0.0016, +0.0032]). 남긴 피처(air_hour, rival_weight, 겹침 가중 rivals)와
# 대체로 중복이라 뺀다. 구조는 맞게 유지하면서 희석만 줄이는 선택이다.
#
# 반면 KBO 그룹은 빼지 않는다. 5폴드 애블레이션에서 빼는 게 나아 보였지만 10폴드에서
# 뒤집혔다 (빼면 +2.42% 나빠짐, CI [+0.0005, +0.0076]). 재현되지 않는 측정으로
# 피처를 버리지 않는다. 실험은 `run_backtest.py --drop <그룹>` 으로 언제든 가능하다.
# `weather_extra` = 근거가 얇은 기상 파생 8개. 기상을 전부 넣으면 235 피처에서
# 0.1606 으로 밀리고, kbo_rain_risk + temp_effective 둘만 남기면 0.1586 으로
# 무기상과 **정확히 동률**이 된다 (차이 +0.0000, CI [-0.0023, +0.0023]).
# 정확도 이득은 없지만 비용도 없어서, 기온 What-If 을 살리려고 둘은 남긴다.
# `event_type` = 유형별 속보 변수. 양성 일수가 5~21일뿐이라 min_child_samples=20 에
# 걸려 **어느 모델도 split 하지 못한다** (split 횟수 0으로 실측 확인). 학습에서 빼고,
# What-If 는 과거 실측 배율을 직접 적용하는 방식으로 처리한다 — forecast.EVENT_EFFECT.
DEFAULT_DROP_GROUPS: tuple[str, ...] = ("sched_extra", "weather_extra", "event_type")
ADVANCE_DROP_GROUPS = DEFAULT_DROP_GROUPS   # 이전 이름 호환


def fit_all(df: pd.DataFrame, horizons=range(1, 8), targets=None,
            events=None, kbo_source=None, with_quantiles: bool = True,
            objective: str = "l1", drop_groups: tuple[str, ...] = DEFAULT_DROP_GROUPS,
            progress=None) -> dict[tuple[str, int], TargetModel]:
    """(타깃, 지평) 조합 전체 학습."""
    from .backtest import FEATURE_GROUPS

    targets = list(targets or TARGET_COLS)
    horizons = list(horizons)
    models: dict[tuple[str, int], TargetModel] = {}
    drop_fns = [FEATURE_GROUPS[g] for g in drop_groups if g in FEATURE_GROUPS]

    total, done = len(targets) * len(horizons), 0
    for h in horizons:
        frame = training_frame(df, h, events=events, kbo_source=kbo_source)
        feats = feature_columns(frame)
        if drop_fns:
            feats = [c for c in feats if not any(fn(c) for fn in drop_fns)]
        for t in targets:
            models[(t, h)] = fit_target(frame, t, h, feats,
                                        with_quantiles=with_quantiles, objective=objective)
            done += 1
            if progress is not None:
                progress(done / total, f"{t} · D+{h}")
    return models


def permutation_importance(model: TargetModel, frame: pd.DataFrame,
                           n_repeats: int = 5, top: int = 15,
                           seed: int = 0) -> pd.DataFrame:
    """홀드아웃 구간에서 피처를 섞었을 때 MAE 가 얼마나 나빠지는지로 기여도 측정."""
    rng = np.random.default_rng(seed)
    X = frame[model.feature_cols].copy()
    y = frame[model.target].to_numpy(dtype=float)

    base = float(np.mean(np.abs(model.estimator.predict(X) - y)))
    rows = []
    for col in model.feature_cols:
        if X[col].nunique(dropna=False) <= 1:
            continue
        deltas = []
        original = X[col].to_numpy(copy=True)
        for _ in range(n_repeats):
            X[col] = rng.permutation(original)
            deltas.append(float(np.mean(np.abs(model.estimator.predict(X) - y))) - base)
        X[col] = original
        rows.append({"feature": col, "importance": float(np.mean(deltas))})

    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out["share"] = out["importance"].clip(lower=0) / max(out["importance"].clip(lower=0).sum(), 1e-9)
    return out.head(top).reset_index(drop=True)
