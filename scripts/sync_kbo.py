"""KBO 월별 일정을 네이버에서 받아 data/kbo_games/ 에 저장한다.

  .venv/bin/python scripts/sync_kbo.py                # 2023~올해
  .venv/bin/python scripts/sync_kbo.py --from 2021    # 더 과거까지
  .venv/bin/python scripts/sync_kbo.py --all          # 이미 받은 달도 전부 다시

외부 프로젝트(kbo-forecast) 수집에 의존하지 않기 위한 자체 캐시다.
지나간 달의 결과는 바뀌지 않으므로 기본적으로 최근 2개월과 미래 달만 다시 받는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratings import kbo                                  # noqa: E402
from ratings.config import KBO_CACHE_DIR                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2023, help="시작 시즌")
    ap.add_argument("--all", action="store_true", help="이미 받은 달도 전부 다시 받기")
    args = ap.parse_args()

    seasons = range(args.start, pd.Timestamp.today().year + 1)
    res = kbo.sync_schedule(seasons, refresh_months=999 if args.all else 2)

    print(f"저장 {len(res['saved'])}개 · 건너뜀 {res['skipped']}개")
    if res["saved"]:
        print("  " + ", ".join(res["saved"][-6:]) + (" …" if len(res["saved"]) > 6 else ""))
    for f in res["failed"]:
        print(f"  실패: {f}")

    sched = kbo.build_schedule_cache()
    print(f"\n일정 캐시 재생성: {len(sched)}일 "
          f"({sched['Date'].min().date()} ~ {sched['Date'].max().date()})")
    print(f"자체 수집본: {KBO_CACHE_DIR}")


if __name__ == "__main__":
    main()
