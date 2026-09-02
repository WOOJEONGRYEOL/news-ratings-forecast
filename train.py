"""모델 학습 후 models/ 에 저장 (대시보드는 저장본을 불러 즉시 뜬다).

  .venv/bin/python train.py
  .venv/bin/python train.py --horizon 14 --no-quantiles
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

from ratings import data as rdata, model as rmodel        # noqa: E402
from ratings.config import MODEL_DIR, RATINGS_CSV         # noqa: E402

BUNDLE = MODEL_DIR / "models.pkl"


def load_events(path: Path | None = None):
    """data/events.csv (start,end,label,weight) 가 있으면 읽어온다."""
    import pandas as pd
    path = path or (RATINGS_CSV.parent / "events.csv")
    if not Path(path).exists():
        return None
    ev = pd.read_csv(path)
    return ev if {"start", "end"}.issubset(ev.columns) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--objective", default="l1", choices=["l1", "l2"])
    ap.add_argument("--no-quantiles", action="store_true")
    ap.add_argument("--drop", default=None,
                    help="학습에서 제외할 피처 그룹 (쉼표 구분, 실험용)")
    args = ap.parse_args()

    df = rdata.load_ratings(args.data)
    events = load_events()
    cov = rdata.coverage(df)
    print(f"데이터 {cov['start']} ~ {cov['end']} ({cov['days']}일)"
          + (f" · 이벤트 구간 {len(events)}개" if events is not None else ""))

    t0 = time.time()
    seen = [0]

    def bar(frac, msg):
        if int(frac * 40) > seen[0]:
            seen[0] = int(frac * 40)
            print(f"\r  [{'█' * seen[0]:<40}] {frac:5.0%}  {msg}", end="", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    drop = tuple(args.drop.split(",")) if args.drop else rmodel.DEFAULT_DROP_GROUPS

    # 사전 예측과 당일 수정은 **같은 모델**을 쓴다. 차이는 학습이 아니라 추론 시점의
    # as_of 게이팅(우천취소를 아는가)에만 있다. 모드별로 피처를 나눠 봤지만
    # 그 근거가 된 애블레이션이 재현되지 않아 되돌렸다 — README 3장 참고.
    models = rmodel.fit_all(df, horizons=range(1, args.horizon + 1), events=events,
                            with_quantiles=not args.no_quantiles,
                            objective=args.objective, drop_groups=drop, progress=bar)
    with BUNDLE.open("wb") as fh:
        pickle.dump({"models": models, "horizon": args.horizon,
                     "data_end": cov["end"], "objective": args.objective,
                     "drop_groups": drop}, fh)
    print(f"\n\n{len(models)}개 모델 저장 → {BUNDLE}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
