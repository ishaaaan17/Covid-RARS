#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
"$VENV_DIR/bin/python" -m pip install -r requirements-dev.txt
INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-auto}"
if [ "$INSTALL_OPTIONAL" != "skip" ]; then
  "$VENV_DIR/bin/python" -m pip install -r requirements-optional.txt
fi
"$VENV_DIR/bin/python" -m pip install -e .

INSTALL_TORCH="${INSTALL_TORCH:-auto}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
if [ "$INSTALL_TORCH" != "skip" ]; then
  if ! "$VENV_DIR/bin/python" - <<'PY'
try:
    import torch  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    "$VENV_DIR/bin/python" -m pip install torch --index-url "$TORCH_INDEX_URL"
  fi
fi

"$VENV_DIR/bin/python" scripts/48_run_strong_baseline_e2e.py "$@"
