"""롤링 오리진 백테스트 실행 CLI.

  .venv/bin/python run_backtest.py --folds 10 --horizon 7
  .venv/bin/python run_backtest.py --ablation          # 피처 그룹 기여도
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

from ratings import backtest, data as rdata           # noqa: E402
from ratings.config import REPORT_DIR                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="시청률 CSV 경로")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--step", type=int, default=21, help="기준일 간격(일)")
    ap.add_argument("--objective", default="l1", choices=["l1", "l2"])
    ap.add_argument("--same-day", action="store_true",
                    help="당일 수정 모드 — 대상일 우천취소 확정 후 예측")
    ap.add_argument("--ablation", action="store_true", help="피처 그룹별 기여도 측정")
    ap.add_argument("--groups", default=None, help="쉼표 구분 (기본: kbo,astro,holiday,slot)")
    ap.add_argument("--drop", default=None, help="학습에서 제외할 피처 그룹 (쉼표 구분)")
    args = ap.parse_args()

    df = rdata.load_ratings(args.data)
    cov = rdata.coverage(df)
    print(f"데이터: {cov['start']} ~ {cov['end']}  ({cov['days']}일, 보정 {cov['imputed_days']}일)\n")

    t0 = time.time()
    started = [0]

    def bar(frac: float, msg: str) -> None:
        if int(frac * 40) > started[0]:
            started[0] = int(frac * 40)
            el = time.time() - t0
            eta = el / max(frac, 1e-9) - el
            print(f"\r  [{'█' * started[0]:<40}] {frac:5.0%}  {msg}  ETA {eta/60:4.1f}분",
                  end="", flush=True)

    if args.ablation:
        print("피처 그룹 애블레이션 (그룹을 빼면 오차가 얼마나 나빠지는가)\n")
        groups = tuple(args.groups.split(",")) if args.groups else \
            ("kbo", "astro", "holiday", "slot")
        summary, per_target = backtest.ablation(
            df, groups=groups, by_target=True, horizon=args.horizon,
            n_folds=args.folds, step=args.step, objective=args.objective)
        print(summary.to_string(index=False))
        print("\n=== 지표별 (악화폭이 음수면 그 그룹은 빼는 편이 낫다) ===")
        print(per_target.to_string(index=False))
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(REPORT_DIR / "ablation.csv", index=False, encoding="utf-8-sig")
        per_target.to_csv(REPORT_DIR / "ablation_by_target.csv", index=False,
                          encoding="utf-8-sig")
        return

    from ratings.model import DEFAULT_DROP_GROUPS
    drop = tuple(args.drop.split(",")) if args.drop else DEFAULT_DROP_GROUPS
    res = backtest.run(df, horizon=args.horizon, n_folds=args.folds, step=args.step,
                       objective=args.objective, same_day=args.same_day,
                       drop_groups=drop, progress=bar)
    print(f"\n\n총 {time.time() - t0:.0f}초 · 평가 표본 {len(res):,}건\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = ("_sameday" if args.same_day else "") + ("_drop" if args.drop else "")
    res.to_csv(REPORT_DIR / f"backtest_raw{tag}.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    by_t = backtest.summarize(res, ("target",))
    by_h = backtest.summarize(res, ("horizon",))
    print("=== 지표별 ===")
    print(by_t.to_string(index=False))
    print("\n=== 예측 지평별 ===")
    print(by_h.to_string(index=False))

    by_t.to_csv(REPORT_DIR / f"backtest_by_target{tag}.csv", index=False, encoding="utf-8-sig")
    by_h.to_csv(REPORT_DIR / f"backtest_by_horizon{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\n리포트 저장: {REPORT_DIR}")


if __name__ == "__main__":
    main()
