"""설정 후보를 롤링 오리진으로 비교한다 (손실함수 · 표본 가중 반감기).

2024Q4 전후로 채널별 시청률 수준이 크게 달라졌기 때문에, 최근 데이터에 얼마나 무게를
실을지가 실제 성능을 좌우한다. 감으로 정하지 말고 측정한다.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from ratings import backtest, data as rdata          # noqa: E402
from ratings.config import REPORT_DIR                # noqa: E402

CONFIGS = [
    ("l1 · 반감기 400일", dict(objective="l1", half_life=400.0)),
    ("l1 · 반감기 180일", dict(objective="l1", half_life=180.0)),
    ("l1 · 가중 없음",    dict(objective="l1", half_life=None)),
    ("l2 · 반감기 400일", dict(objective="l2", half_life=400.0)),
]


def main() -> None:
    df = rdata.load_ratings()
    rows = []
    for name, kw in CONFIGS:
        t0 = time.time()
        res = backtest.run(df, horizon=7, n_folds=6, step=28, **kw)
        err = (res["model"] - res["actual"]).abs()
        base = (res["snaive_7"] - res["actual"]).abs()
        rows.append({
            "설정": name,
            "MAE": round(float(err.mean()), 4),
            "RMSE": round(float(((res["model"] - res["actual"]) ** 2).mean() ** 0.5), 4),
            "vs 지난주동요일%": round((1 - err.mean() / base.mean()) * 100, 1),
            "초": round(time.time() - t0),
        })
        print(f"  {name:20s} MAE {rows[-1]['MAE']}  ({rows[-1]['초']}초)", flush=True)

    out = pd.DataFrame(rows).sort_values("MAE")
    print()
    print(out.to_string(index=False))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORT_DIR / "tuning.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
