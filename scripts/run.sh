#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -n "${PY:-}" ]]; then
  PY="$PY"
elif [[ -x "$HOME/.openclaw/workspace/psyprofile-platform/backend/.venv/bin/python" ]]; then
  PY="$HOME/.openclaw/workspace/psyprofile-platform/backend/.venv/bin/python"
else
  echo "先创建虚拟环境并安装依赖: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ ! -f apps/web/index.html ]] || [[ apps/web/api.js -nt apps/web/index.html ]] || [[ apps/web/core.js -nt apps/web/index.html ]] || [[ apps/web/system.js -nt apps/web/index.html ]]; then
  "$PY" apps/web/build.py
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec "$PY" apps/api/main.py
