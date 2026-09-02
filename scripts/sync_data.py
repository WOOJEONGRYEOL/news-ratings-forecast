"""구글 시트에서 최신 시청률을 내려받아 data/종편_4사_메인_시청률.csv 를 갱신한다.

  .venv/bin/python scripts/sync_data.py            # 받아서 검증 후 교체
  .venv/bin/python scripts/sync_data.py --retrain  # 갱신됐으면 모델까지 재학습

시트가 '링크가 있는 모든 사용자' 공개 상태여야 한다. 기존 파일은 교체 전에 검증하고,
행 수가 줄거나 컬럼이 어긋나면 교체하지 않는다.

시트 ID 는 코드에 없다 — `RATINGS_SHEET_ID` 환경변수, `data/sheet_id.txt`,
Streamlit secrets 중 하나로 지정한다 (config.sheet_id 참고).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratings import data as rdata                              # noqa: E402
from ratings.config import PROJECT_ROOT, RATINGS_CSV, sheet_id  # noqa: E402


def export_url() -> str:
    """시트를 CSV 로 내려받는 주소. ID 는 config.sheet_id() 가 해결한다."""
    return f"https://docs.google.com/spreadsheets/d/{sheet_id()}/export?format=csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="기본값: 시트 ID 로 조립")
    ap.add_argument("--retrain", action="store_true", help="갱신 시 모델 재학습")
    args = ap.parse_args()

    url = args.url or export_url()
    before = None
    if RATINGS_CSV.exists():
        before = rdata.coverage(rdata.load_ratings())
        print(f"현재: {before['start']} ~ {before['end']} ({before['days']}일)")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    print("내려받는 중…")
    urllib.request.urlretrieve(url, tmp_path)          # noqa: S310  (고정 구글 도메인)

    # 교체 전에 파싱이 되는지, 데이터가 줄지 않았는지 확인한다
    try:
        fresh = rdata.load_ratings(tmp_path)
    except (KeyError, ValueError, UnicodeDecodeError) as exc:
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"받은 파일을 읽을 수 없습니다 — 교체하지 않았습니다.\n  {exc}")

    after = rdata.coverage(fresh)
    if before and after["days"] < before["days"]:
        tmp_path.unlink(missing_ok=True)
        sys.exit(f"행 수가 줄었습니다 ({before['days']} → {after['days']}일). "
                 f"시트를 확인하세요 — 교체하지 않았습니다.")

    if before and after["end"] == before["end"] and after["days"] == before["days"]:
        tmp_path.unlink(missing_ok=True)
        print("변경 없음.")
        return

    RATINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_path), RATINGS_CSV)
    added = after["days"] - (before["days"] if before else 0)
    print(f"갱신: {after['start']} ~ {after['end']} ({after['days']}일, +{added}일)")

    if args.retrain:
        print("\n모델 재학습…")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "train.py")], check=True)


if __name__ == "__main__":
    main()
