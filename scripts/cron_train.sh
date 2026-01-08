#!/usr/bin/env bash
# scripts/cron_train.sh
#
# Continuous Training wrapper for SkyNet (RTX 5090).
#
# Behavior:
# - Download latest refinery parquet from Debian (size-gated)
# - Activate .venv-blackwell
# - Train with walk-forward validation (handled inside oracle.train_model)
# - On acceptance, ensure models/oracle_v1_latest.pt exists and scp it to Debian ~/eve-trader/models/
# - Ensure oracle/indicators.py is identical on Debian (sha sync)
#
# Usage:
#   ./scripts/cron_train.sh            # run once
#   ./scripts/cron_train.sh --print-cron
#
# Env overrides:
# - DEBIAN_HOST (default 192.168.14.105)
# - DEBIAN_USER (default seb)
# - DEBIAN_REPO_DIR (default ~/eve-trader)
# - PARQUET_URL (default http://192.168.14.105:8001/training_data_cleaned.parquet)
# - MIN_PARQUET_BYTES (default 17000000)
# - VENV_PATH (default .venv-blackwell)
# - EPOCHS (default 50)
# - LOOKBACK_MINUTES (default 120)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEBIAN_HOST="${DEBIAN_HOST:-192.168.14.105}"
DEBIAN_USER="${DEBIAN_USER:-seb}"
DEBIAN_REPO_DIR="${DEBIAN_REPO_DIR:-~/eve-trader}"
PARQUET_URL="${PARQUET_URL:-http://192.168.14.105:8001/training_data_cleaned.parquet}"
MIN_PARQUET_BYTES="${MIN_PARQUET_BYTES:-17000000}"
VENV_PATH="${VENV_PATH:-.venv-blackwell}"
EPOCHS="${EPOCHS:-50}"
LOOKBACK_MINUTES="${LOOKBACK_MINUTES:-120}"

PRINT_CRON=0
for arg in "$@"; do
  case "$arg" in
    --print-cron) PRINT_CRON=1 ;;
    *) ;;
  esac
done

if [[ "$PRINT_CRON" == "1" ]]; then
  cat <<EOF
# Run every 4 hours (SkyNet cron):
#   crontab -e
# Add:
0 */4 * * * cd $ROOT_DIR && ./scripts/cron_train.sh >> logs/cron_train.log 2>&1
EOF
  exit 0
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 2; }
}

require_cmd python3
require_cmd ssh
require_cmd scp

# curl or wget for parquet retrieval
if command -v curl >/dev/null 2>&1; then
  DL_CMD="curl"
elif command -v wget >/dev/null 2>&1; then
  DL_CMD="wget"
else
  echo "Need curl or wget" >&2
  exit 2
fi

log "Starting continuous training cycle"
log "Parquet URL: $PARQUET_URL"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
PARQUET_PATH="$ROOT_DIR/data/refinery_${TS}.parquet"

log "Downloading parquet -> $PARQUET_PATH"
if [[ "$DL_CMD" == "curl" ]]; then
  curl -fsSL --connect-timeout 10 --max-time 300 -o "$PARQUET_PATH" "$PARQUET_URL"
else
  wget -q -O "$PARQUET_PATH" "$PARQUET_URL"
fi

SIZE="$(stat -c %s "$PARQUET_PATH" 2>/dev/null || echo 0)"
if [[ "$SIZE" -lt "$MIN_PARQUET_BYTES" ]]; then
  echo "Parquet too small ($SIZE bytes; min $MIN_PARQUET_BYTES). Aborting." >&2
  exit 3
fi
log "Parquet size OK: $SIZE bytes"

# Activate venv
if [[ ! -d "$ROOT_DIR/$VENV_PATH" ]]; then
  echo "Venv not found at $ROOT_DIR/$VENV_PATH" >&2
  exit 4
fi
# shellcheck disable=SC1090
source "$ROOT_DIR/$VENV_PATH/bin/activate"

log "Training (epochs=$EPOCHS lookback=$LOOKBACK_MINUTES)"
set +e
python3 -m oracle.train_model \
  --parquet "$PARQUET_PATH" \
  --epochs "$EPOCHS" \
  --lookback-minutes "$LOOKBACK_MINUTES" \
  --set-live
RC=$?
set -e

if [[ "$RC" != "0" ]]; then
  echo "Training rejected/failed (rc=$RC); not syncing model." >&2
  exit "$RC"
fi

LATEST_LOCAL="$ROOT_DIR/models/oracle_v1_latest.pt"
if [[ ! -f "$LATEST_LOCAL" ]]; then
  echo "Expected latest model at $LATEST_LOCAL but not found" >&2
  exit 5
fi

log "Sync indicators.py to Debian (sha check)"
LOCAL_SHA="$(sha256sum "$ROOT_DIR/oracle/indicators.py" | awk '{print $1}')"
REMOTE_SHA="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$DEBIAN_USER@$DEBIAN_HOST" "sha256sum $DEBIAN_REPO_DIR/oracle/indicators.py 2>/dev/null | awk '{print \$1}'" || true)"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  log "Remote indicators mismatch (local=$LOCAL_SHA remote=${REMOTE_SHA:-missing}); syncing"
  scp "$ROOT_DIR/oracle/indicators.py" "$DEBIAN_USER@$DEBIAN_HOST:$DEBIAN_REPO_DIR/oracle/indicators.py"
fi

log "Sync oracle_v1_latest.pt to Debian models/"
scp "$LATEST_LOCAL" "$DEBIAN_USER@$DEBIAN_HOST:$DEBIAN_REPO_DIR/models/oracle_v1_latest.pt"

log "Done"
