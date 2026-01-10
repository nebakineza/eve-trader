#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR_DEFAULT="$ROOT_DIR/.venv"
VENV_DIR="${VENV_DIR:-$VENV_DIR_DEFAULT}"
REQ_FILE_DEFAULT="$ROOT_DIR/requirements.venv.prod.txt"
REQ_FILE="${REQ_FILE:-$REQ_FILE_DEFAULT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing python3" >&2
  exit 2
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 3
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[venv] Creating: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel >/dev/null
python -m pip install -r "$REQ_FILE"

echo "[venv] Ready: $(python -c 'import sys; print(sys.executable)')"
python -c "import cv2, numpy as np; print('cv2', cv2.__version__); print('np', np.__version__)"
