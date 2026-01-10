import argparse
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WarHistorian")


def _redis_client(redis_url: str):
    import redis

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _parse_kill_time(raw: Any) -> datetime | None:
    if raw is None:
        return None
    try:
        s = str(raw).strip()
        if not s:
            return None
        # zKillboard commonly returns ISO timestamps like "2026-01-08T12:34:56Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def fetch_isk_destroyed_last_window(
    *,
    lookback_minutes: int,
    max_pages: int,
    sleep_between_requests_s: float,
    user_agent: str,
) -> tuple[float, str]:
    """Best-effort estimate of total ISK destroyed over the last lookback window.

    Implementation notes:
    - Uses zKillboard public API, fetching newest kills and summing zkb.totalValue.
    - Stops paging once the oldest kill in a page is older than the window.
    - Rate-limits politely to avoid getting blocked.

    If zKillboard changes schema/endpoints, this returns 0.0 rather than raising.
    """

    # MCP-first: if a kongyo2 OSINT MCP endpoint is configured, prefer it.
    # Expected JSON shape (best-effort): {"isk_destroyed": <float>} or {"warfare_isk_destroyed": <float>}.
    mcp_endpoint = os.getenv("KONGYO2_OSINT_MCP_ENDPOINT", "").strip()
    if mcp_endpoint:
        try:
            resp = requests.get(
                mcp_endpoint,
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=20,
            )
            if resp.status_code == 200:
                payload = resp.json() or {}
                if isinstance(payload, dict):
                    v = payload.get("isk_destroyed")
                    if v is None:
                        v = payload.get("warfare_isk_destroyed")
                    if v is None:
                        v = payload.get("system:macro:warfare")
                    val = _as_float(v)
                    if val is not None and val >= 0:
                        return float(val), "kongyo2_osint_mcp"
        except Exception:
            # Fall back to zKillboard.
            pass

    base_url = os.getenv("WAR_ZKILL_URL", "https://zkillboard.com/api/kills/").rstrip("/") + "/"
    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=int(lookback_minutes))

    esi_base = os.getenv("ESI_BASE_URL", "https://esi.evetech.net/latest").rstrip("/")
    esi_time_lookup = os.getenv("WAR_ZKILL_ESI_TIME_LOOKUP", "1").strip().lower() in {"1", "true", "yes"}
    esi_time_lookup_max = int(os.getenv("WAR_ZKILL_ESI_TIME_LOOKUP_MAX", "30"))
    esi_lookups = 0

    def _killmail_time_via_esi(*, killmail_id: int, km_hash: str) -> datetime | None:
        nonlocal esi_lookups
        if not esi_time_lookup:
            return None
        if esi_time_lookup_max > 0 and esi_lookups >= esi_time_lookup_max:
            return None
        if not killmail_id or not km_hash:
            return None
        esi_lookups += 1
        url = f"{esi_base}/killmails/{int(killmail_id)}/{str(km_hash).strip()}/"
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                return None
            payload = resp.json() or {}
            if isinstance(payload, dict):
                return _parse_kill_time(payload.get("killmail_time") or payload.get("killmailTime"))
        except Exception:
            return None
        return None

    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json",
    }

    total = 0.0
    for page in range(1, int(max_pages) + 1):
        # zKillboard paging format: /api/kills/page/2/
        url = base_url
        if page > 1:
            url = base_url + f"page/{page}/"

        try:
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.status_code != 200:
                logger.warning("zKillboard HTTP %s for %s", resp.status_code, url)
                break
            data = resp.json()
            if not isinstance(data, list) or not data:
                break
        except Exception as e:
            logger.warning("zKillboard fetch failed: %s", e)
            break

        oldest_in_page = None
        for item in data:
            if not isinstance(item, dict):
                continue
            dt = _parse_kill_time(item.get("killmail_time") or item.get("killmailTime"))
            if dt is None:
                # zKillboard schema variant: may only include killmail_id + zkb.hash.
                try:
                    killmail_id = int(item.get("killmail_id") or item.get("killmailID") or 0)
                except Exception:
                    killmail_id = 0
                zkb = item.get("zkb") or {}
                km_hash = zkb.get("hash") if isinstance(zkb, dict) else None
                if killmail_id and km_hash:
                    dt = _killmail_time_via_esi(killmail_id=killmail_id, km_hash=str(km_hash))
            if dt is None:
                continue
            if oldest_in_page is None or dt < oldest_in_page:
                oldest_in_page = dt
            if dt < cutoff:
                continue

            zkb = item.get("zkb") or {}
            value = _as_float(zkb.get("totalValue") or zkb.get("total_value"))
            if value is None or value < 0:
                continue
            total += float(value)

        # Stop if we already covered the full window.
        if oldest_in_page is not None and oldest_in_page < cutoff:
            break

        time.sleep(max(0.0, float(sleep_between_requests_s)))

    return float(total), ("zkillboard" if esi_lookups == 0 else "zkillboard+esi")


def publish_warfare_macro(*, redis_url: str, isk_destroyed: float, ttl_seconds: int, source: str) -> None:
    r = _redis_client(redis_url)
    now = datetime.now(tz=timezone.utc)

    # Defensive: do not clobber a fresh non-zero value with 0.0.
    # zKillboard occasionally rate-limits or returns schema variants; a transient fetch failure
    # would otherwise force the dashboard to show 0 even when a recent non-zero signal exists.
    try:
        new_v = float(isk_destroyed)
        if new_v <= 0.0:
            prev_v = _as_float(r.get("system:macro:warfare"))
            prev_ts = _as_float(r.get("system:macro:warfare:updated_at"))
            if prev_v is not None and prev_v > 0.0 and prev_ts is not None:
                age_s = max(0.0, float(now.timestamp()) - float(prev_ts))
                if age_s < max(60.0, float(ttl_seconds) * 2.0):
                    logger.warning(
                        "Skipping overwrite of fresh non-zero warfare (prev=%.2f age=%.0fs) with %.2f (source=%s)",
                        float(prev_v),
                        float(age_s),
                        float(new_v),
                        str(source),
                    )
                    return
    except Exception:
        pass
    r.set("system:macro:warfare", str(float(isk_destroyed)), ex=int(ttl_seconds))
    r.set("system:macro:warfare:updated_at", str(now.timestamp()), ex=int(ttl_seconds))
    r.set("system:macro:warfare:source", str(source), ex=int(ttl_seconds))


def main() -> None:
    p = argparse.ArgumentParser(description="War-Historian: publish last-60m ISK destroyed to Redis")
    p.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        help="Redis URL (defaults to REDIS_URL)",
    )
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.getenv("WAR_POLL_SECONDS", "300")),
        help="Poll interval seconds",
    )
    p.add_argument(
        "--lookback-minutes",
        type=int,
        default=int(os.getenv("WAR_LOOKBACK_MINUTES", "60")),
        help="Window size in minutes",
    )
    p.add_argument(
        "--ttl-seconds",
        type=int,
        default=int(os.getenv("WAR_TTL_SECONDS", "600")),
        help="Redis TTL seconds",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=int(os.getenv("WAR_ZKILL_MAX_PAGES", "3")),
        help="Max zKillboard pages to fetch per poll",
    )
    p.add_argument(
        "--sleep-between-requests",
        type=float,
        default=float(os.getenv("WAR_ZKILL_SLEEP_SECONDS", "1.1")),
        help="Sleep between zKillboard requests",
    )
    p.add_argument(
        "--user-agent",
        default=os.getenv("WAR_USER_AGENT", "eve-trader/war-historian (contact: ops)"),
        help="HTTP User-Agent for zKillboard",
    )
    p.add_argument("--once", action="store_true", help="Run once and exit")

    args = p.parse_args()

    logger.info("War-Historian started (lookback=%dm poll=%ds)", args.lookback_minutes, args.poll_seconds)

    while True:
        isk, src = fetch_isk_destroyed_last_window(
            lookback_minutes=args.lookback_minutes,
            max_pages=args.max_pages,
            sleep_between_requests_s=args.sleep_between_requests,
            user_agent=args.user_agent,
        )
        try:
            publish_warfare_macro(
                redis_url=args.redis_url,
                isk_destroyed=isk,
                ttl_seconds=args.ttl_seconds,
                source=src,
            )
            logger.info("Published system:macro:warfare=%.2f (source=%s)", float(isk), src)
        except Exception as e:
            logger.warning("Redis publish failed: %s", e)

        if args.once:
            return
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
