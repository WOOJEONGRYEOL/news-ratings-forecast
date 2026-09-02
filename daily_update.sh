#!/usr/bin/env bash
# 매일 아침 실행: 시트에서 전날 시청률을 받아 갱신되면 모델을 다시 학습한다.
#
#   ./daily_update.sh
#
# cron 등록 예 (매일 08:10):
#   10 8 * * * ./daily_update.sh >> logs/daily.log 2>&1
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

echo "=== $(date '+%Y-%m-%d %H:%M') ==="
# sync_data.py 는 파싱과 행 수를 검증한 뒤에만 교체한다(시트가 깨져도 기존 데이터 보존).
# --retrain 은 실제로 갱신됐을 때만 학습을 돌린다.
.venv/bin/python scripts/sync_data.py --retrain

# KBO 월별 일정 자체 수집 (최근 2개월 + 미래 달만 다시 받는다)
.venv/bin/python scripts/sync_kbo.py

# 기상: 실측 아카이브와 리드별 예보를 최근 구간만 다시 받는다
.venv/bin/python -c "
import sys, warnings; sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
import pandas as pd
from ratings import weather
start = (pd.Timestamp.today() - pd.Timedelta(days=400)).date().isoformat()
weather.build_cache(start, pd.Timestamp.today().date().isoformat())
print('기상 데이터 갱신')
"
echo "완료"
