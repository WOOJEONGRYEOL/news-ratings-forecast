#!/usr/bin/env bash
# 대시보드 실행. 모델이 없으면 먼저 학습한다.
set -euo pipefail
cd "$(dirname "$0")"
[[ -f models/models.pkl ]] || .venv/bin/python train.py
exec .venv/bin/streamlit run app.py
