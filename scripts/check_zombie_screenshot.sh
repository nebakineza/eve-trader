#!/usr/bin/env bash
set -euo pipefail

# Quick sanity check: does Redis contain a non-empty system:zombie:screenshot payload?
# Works whether Redis is local or the compose container.

KEY="${ZOMBIE_SHOT_REDIS_KEY:-system:zombie:screenshot}"
MIN_BYTES="${ZOMBIE_SHOT_MIN_REDIS_BYTES:-50000}"

if docker ps --format '{{.Names}}' | grep -q '^eve-trader-redis$'; then
  docker exec eve-trader-redis redis-cli --raw EXISTS "$KEY"
  n=$(docker exec eve-trader-redis redis-cli --raw STRLEN "$KEY" | tr -d '\r')
  echo "$n"
  if [[ "$n" -lt "$MIN_BYTES" ]]; then
    echo "[!] Screenshot payload too small: $n < $MIN_BYTES" >&2
    exit 2
  fi
  exit 0
fi

if command -v redis-cli >/dev/null 2>&1; then
  redis-cli --raw EXISTS "$KEY"
  n=$(redis-cli --raw STRLEN "$KEY" | tr -d '\r')
  echo "$n"
  if [[ "$n" -lt "$MIN_BYTES" ]]; then
    echo "[!] Screenshot payload too small: $n < $MIN_BYTES" >&2
    exit 2
  fi
  exit 0
fi

echo "[!] Neither docker redis nor redis-cli found." >&2
exit 1
