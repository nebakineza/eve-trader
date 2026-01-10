#!/usr/bin/env bash
# scripts/induction_watchdog.sh
#
# Autonomous induction watchdog:
# - If oracle:model_status is IDLE for > 4 hours, trigger a high-energy induction run
#   (100 epochs, higher learning rate) via scripts/cron_train.sh.
#
# Env overrides:
# - REDIS_URL (default redis://192.168.14.105:6379/0)
# - IDLE_HOURS_THRESHOLD (default 4)
# - INDUCTION_EPOCHS (default 100)
# - INDUCTION_LR (default 0.003)
# - LOOKBACK_MINUTES (passed through to cron_train.sh if set)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REDIS_URL="${REDIS_URL:-redis://192.168.14.105:6379/0}"
IDLE_HOURS_THRESHOLD="${IDLE_HOURS_THRESHOLD:-4}"
INDUCTION_EPOCHS="${INDUCTION_EPOCHS:-100}"
INDUCTION_LR="${INDUCTION_LR:-0.005}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 2; }
}

require_cmd python3

# Use python+redis-py to avoid depending on redis-cli/jq.
# redis-py is already used throughout the stack.
py_out="$(
python3 - <<'PY'
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import redis
except Exception as e:
    print(json.dumps({"ok": False, "error": f"redis_import_failed: {e}"}))
    sys.exit(0)

redis_url = os.getenv("REDIS_URL", "redis://192.168.14.105:6379/0")
r = redis.Redis.from_url(redis_url, decode_responses=True)

status = None
metrics_raw = None
try:
    status = r.get("oracle:model_status")
except Exception:
    status = None

try:
    metrics_raw = r.get("oracle:last_training_metrics")
except Exception:
    metrics_raw = None

last_ts = None
if metrics_raw:
    try:
        obj = json.loads(metrics_raw)
        # train_model.py writes metrics["timestamp"] as YYYYMMDDTHHMMSSZ
        last_ts = obj.get("timestamp")
    except Exception:
        last_ts = None

last_dt = None
if last_ts:
    s = str(last_ts).strip()
    try:
        last_dt = datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        last_dt = None

age_hours = None
if last_dt is not None:
    age_hours = (datetime.now(tz=timezone.utc) - last_dt).total_seconds() / 3600.0

print(json.dumps({
    "ok": True,
    "status": status,
    "metrics_ts": last_ts,
    "age_hours": age_hours,
}))
PY
)"

if [[ -z "$py_out" ]]; then
  echo "Watchdog probe failed (no output)" >&2
  exit 3
fi

ok="$(python3 -c 'import json,sys; print(str(bool(json.loads(sys.stdin.read()).get("ok"))).lower())' <<<"$py_out")"
if [[ "$ok" != "true" ]]; then
  err="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("error","unknown"))' <<<"$py_out")"
  echo "Watchdog probe failed: $err" >&2
  exit 3
fi

status="$(python3 -c 'import json,sys; v=json.loads(sys.stdin.read()).get("status"); print("" if v is None else str(v))' <<<"$py_out")"
age_hours="$(python3 -c 'import json,sys; v=json.loads(sys.stdin.read()).get("age_hours"); print("" if v is None else str(v))' <<<"$py_out")"

log "oracle:model_status=${status:-UNKNOWN} age_hours=${age_hours:-N/A}"

# Only trigger when explicitly IDLE, and only if we can compute age.
if [[ "${status^^}" != "IDLE" ]]; then
  log "No action (status is not IDLE)"
  exit 0
fi

if [[ -z "$age_hours" ]]; then
  log "No action (missing last_training_metrics timestamp)"
  exit 0
fi

# Compare as floats via python
should_trigger="$(python3 -c 'import sys; a=float(sys.argv[1]); t=float(sys.argv[2]); print("1" if a>t else "0")' "$age_hours" "$IDLE_HOURS_THRESHOLD")"
if [[ "$should_trigger" != "1" ]]; then
  log "No action (IDLE duration below threshold)"
  exit 0
fi

log "IDLE > ${IDLE_HOURS_THRESHOLD}h; triggering induction run (epochs=$INDUCTION_EPOCHS lr=$INDUCTION_LR)"

cd "$ROOT_DIR"
EPOCHS="$INDUCTION_EPOCHS" LEARNING_RATE="$INDUCTION_LR" ./scripts/cron_train.sh
