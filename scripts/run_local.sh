#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [ ! -f "${ROOT_DIR}/.env" ]; then
  echo "[!] .env not found. Copy .env.example to .env and customise before running." >&2
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate" 2>/dev/null || true

python -m app.cli fetch
python -m app.cli preprocess
python -m app.cli build-index
uvicorn app.main:app --host 0.0.0.0 --port "${QA_API_PORT:-8000}"
