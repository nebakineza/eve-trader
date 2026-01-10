import json
import logging
import os
import sys
import time
from datetime import datetime

import redis
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.arduino_connector import ArduinoBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ArduinoBridgeService")


def _redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://cache:6379/0")
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _set_status(r: redis.Redis, status: str, detail: str | None = None) -> None:
    # Dashboard reads zombie:hid_status (GREEN => online).
    r.set("zombie:hid_status", status)
    r.set("zombie:hid_last_seen", datetime.utcnow().isoformat() + "Z")
    if detail is not None:
        r.set("zombie:hid_detail", detail)


def _is_hardware_locked(r: redis.Redis) -> bool:
    """Check if HARDWARE_LOCK toggle is enabled (safe mode)."""
    try:
        val = r.get("system:hardware_lock")
        return val is None or str(val).lower() in {"true", "1", "yes"}
    except Exception:
        return True  # default to locked/safe


def _get_pending_signals(r: redis.Redis, confidence_threshold: float = 0.65):
    """
    Scan market_radar:* keys for high-confidence signals.
    Returns list of dicts: {type_id, signal_type, confidence, price, predicted}
    """
    signals = []
    try:
        keys = r.keys("market_radar:*")
        for k in keys:
            raw = r.get(k)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                conf = float(obj.get("confidence", 0.0) or 0.0)
                if conf >= confidence_threshold:
                    predicted = float(obj.get("predicted", 0.0) or 0.0)
                    price = float(obj.get("price", 0.0) or 0.0)
                    if predicted > price:
                        signal_type = "BUY"
                    elif predicted < price:
                        signal_type = "SELL"
                    else:
                        signal_type = None
                    if signal_type:
                        signals.append({
                            "type_id": obj.get("type_id"),
                            "signal_type": signal_type,
                            "confidence": conf,
                            "price": price,
                            "predicted": predicted,
                        })
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Signal scan error: {e}")
    return signals


def _handshake_requested(r: redis.Redis) -> bool:
    try:
        v = r.get("system:hardware:handshake")
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "go", "now"} or s != ""
    except Exception:
        return False


def _consume_handshake_request(r: redis.Redis) -> None:
    # One-shot semantics: delete first so we never double-fire.
    try:
        r.delete("system:hardware:handshake")
    except Exception:
        pass


def _db_url() -> str | None:
    v = os.getenv("DATABASE_URL", "").strip()
    return v or None


def _winsorized_sharpe_4h(
    *,
    horizon_minutes: int,
    lookback_hours: float,
    winsor_lo: float,
    winsor_hi: float,
    stop_loss_pct: float,
) -> float | None:
    db_url = _db_url()
    if not db_url:
        return None

    # Pull recent resolved trades and compute returns in Python (winsorized).
    query = """
        SELECT
            t.actual_price_at_time AS entry_price,
            mh.close AS exit_price,
            t.signal_type,
            t.forced_exit
        FROM shadow_trades t
        LEFT JOIN LATERAL (
            SELECT close
            FROM market_history
            WHERE type_id = t.type_id
              AND \"timestamp\" >= t.timestamp + (%s || ' minutes')::interval
            ORDER BY \"timestamp\" ASC
            LIMIT 1
        ) mh ON TRUE
        WHERE t.virtual_outcome IN ('WIN', 'LOSS')
          AND t.timestamp >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY t.timestamp DESC
        LIMIT 200;
    """

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (int(horizon_minutes), float(lookback_hours)))
                rows = cur.fetchall()
    except Exception:
        return None

    rets: list[float] = []
    for entry_price, exit_price, signal_type, forced_exit in rows:
        try:
            forced = bool(forced_exit)
            if forced:
                ret = -abs(float(stop_loss_pct))
            else:
                ep = float(entry_price or 0.0)
                xp = float(exit_price or 0.0)
                if ep <= 0 or xp <= 0:
                    continue
                stype = str(signal_type or "").upper()
                if stype == "BUY":
                    ret = (xp - ep) / ep
                else:
                    ret = (ep - xp) / ep

            if ret < winsor_lo:
                ret = winsor_lo
            elif ret > winsor_hi:
                ret = winsor_hi
            rets.append(float(ret))
        except Exception:
            continue

    if len(rets) < 5:
        return 0.0

    mu = sum(rets) / float(len(rets))
    var = sum((x - mu) ** 2 for x in rets) / float(max(1, len(rets) - 1))
    sigma = var ** 0.5
    if sigma <= 0:
        return 0.0
    return float(mu / sigma)


def main() -> int:
    serial_port = os.getenv("ARDUINO_PORT", "/dev/ttyACM0")
    baud_rate = int(os.getenv("ARDUINO_BAUD", "9600"))
    once = os.getenv("ONCE", "0").strip() in {"1", "true", "yes"}
    pulse_test = os.getenv("ARDUINO_PULSE_TEST", "0").strip() in {"1", "true", "yes"}
    dry_run = os.getenv("ARDUINO_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}
    retry_seconds = float(os.getenv("RETRY_SECONDS", "5"))
    confidence_threshold = float(os.getenv("ARDUINO_CONFIDENCE_THRESHOLD", "0.85"))
    poll_seconds = float(os.getenv("ARDUINO_POLL_SECONDS", "10"))

    # Bardol multi-strategy gate: require winsorized Sharpe over recent window.
    sharpe_gate_enabled = os.getenv("ARDUINO_SHARPE_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes"}
    min_sharpe_4h = float(os.getenv("ARDUINO_MIN_SHARPE_4H", "1.5"))
    sharpe_lookback_hours = float(os.getenv("ARDUINO_SHARPE_LOOKBACK_HOURS", "4"))
    sharpe_refresh_seconds = float(os.getenv("ARDUINO_SHARPE_REFRESH_SECONDS", "60"))
    sharpe_horizon_minutes = int(os.getenv("ARDUINO_SHARPE_HORIZON_MINUTES", "60"))
    sharpe_winsor_lo = float(os.getenv("ARDUINO_SHARPE_WINSOR_LO", "-1.0"))
    sharpe_winsor_hi = float(os.getenv("ARDUINO_SHARPE_WINSOR_HI", "10.0"))
    sharpe_stop_loss_pct = float(os.getenv("SHADOW_STOP_LOSS_PCT", "0.015"))

    last_sharpe_check_ts = 0.0
    last_sharpe_value: float | None = None

    # Handshake macro tuning (defaults match operator confirmed sequence)
    handshake_end_cmd = os.getenv("ARDUINO_HANDSHAKE_END_CMD", "END")
    handshake_accept_cmd = os.getenv("ARDUINO_HANDSHAKE_ACCEPT_CMD", "ENTER")
    handshake_delay_ms = int(os.getenv("ARDUINO_HANDSHAKE_DELAY_MS", "500"))

    r = None
    try:
        r = _redis_client()
    except Exception as e:
        logger.error(f"Redis unavailable: {e}")

    def publish(status: str, detail: str | None = None):
        if r is None:
            return
        try:
            _set_status(r, status=status, detail=detail)
        except Exception:
            pass

    bridge: ArduinoBridge | None = None

    if dry_run:
        logger.warning("🧪 ARDUINO_DRY_RUN enabled: no serial commands will be sent")

        # Ensure we never attempt to open a real serial port in dry-run.
        serial_port = ""

        class _DryRunBridge:
            port = "DRY_RUN"

            def is_active(self) -> bool:
                return True

            def send_command(self, cmd: str) -> bool:
                logger.warning("🧪 DRY_RUN would send command: %s", cmd)
                return True

        bridge = _DryRunBridge()  # type: ignore[assignment]

    logger.info(
        "ArduinoBridge starting (mode=%s conf>=%.2f sharpe_gate=%s sharpe>=%.2f lookback_h=%.2f refresh_s=%.0f horizon_min=%d)",
        "DRY_RUN" if dry_run else "PRODUCTION",
        float(confidence_threshold),
        "ON" if sharpe_gate_enabled else "OFF",
        float(min_sharpe_4h),
        float(sharpe_lookback_hours),
        float(sharpe_refresh_seconds),
        int(sharpe_horizon_minutes),
    )

    while True:
        try:
            # (Re)connect to Arduino if needed
            if not dry_run and (bridge is None or not bridge.is_active()):
                bridge = ArduinoBridge(baud_rate=baud_rate)
                # If user specified a fixed port, prefer it.
                if serial_port:
                    try:
                        import serial

                        bridge.port = serial_port
                        bridge.connection = serial.Serial(serial_port, baud_rate, timeout=1)
                        time.sleep(2)
                    except Exception as e:
                        bridge.connection = None
                        logger.error(f"Failed to open {serial_port}: {e}")

            # In dry-run, force an always-active bridge and do not publish SERIAL mode.
            if dry_run:
                if bridge is None:
                    class _DryRunBridge:
                        port = "DRY_RUN"

                        def is_active(self) -> bool:
                            return True

                        def send_command(self, cmd: str) -> bool:
                            logger.warning("🧪 DRY_RUN would send command: %s", cmd)
                            return True

                    bridge = _DryRunBridge()  # type: ignore[assignment]
                publish("GREEN", detail=json.dumps({"port": "DRY_RUN", "mode": "DRY_RUN", "ok": True}))
            else:
                if bridge.is_active():
                    detail = json.dumps({"port": bridge.port, "mode": "SERIAL", "ok": True})
                    publish("GREEN", detail=detail)
                else:
                    publish(
                        "AMBER",
                        detail=json.dumps({"port": bridge.port if bridge else None, "mode": "SERIAL", "ok": False}),
                    )
                    logger.warning("🟠 Arduino OFFLINE (no serial connection)")
                    if once:
                        return 1
                    time.sleep(retry_seconds)
                    continue

            # EULA Handshake (hardware macro): END -> wait -> RETURN/ENTER
            # Safety: never allow a queued handshake to execute later after an unlock.
            if r and _handshake_requested(r):
                if _is_hardware_locked(r):
                    _consume_handshake_request(r)
                    logger.warning("📜 Hardware handshake requested but HARDWARE_LOCK is enabled; discarding request.")
                    if once:
                        return 0
                    time.sleep(poll_seconds)
                    continue

                _consume_handshake_request(r)
                logger.info("📜 Hardware handshake requested: sending %s then %s", handshake_end_cmd, handshake_accept_cmd)
                ok1 = bridge.send_command(handshake_end_cmd)
                time.sleep(max(0.0, float(handshake_delay_ms) / 1000.0))
                ok2 = bridge.send_command(handshake_accept_cmd)
                try:
                    r.lpush(
                        "arduino:command_log",
                        json.dumps(
                            {
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "command": "HARDWARE_HANDSHAKE",
                                "sequence": [handshake_end_cmd, handshake_accept_cmd],
                                "delay_ms": int(handshake_delay_ms),
                                "ok": bool(ok1 and ok2),
                            }
                        ),
                    )
                    r.ltrim("arduino:command_log", 0, 499)
                except Exception:
                    pass
                # After handshake, wait one poll interval to avoid accidental rapid repeats.
                if once:
                    return 0 if (ok1 and ok2) else 1
                time.sleep(poll_seconds)
                continue

            # Check HARDWARE_LOCK
            if r and _is_hardware_locked(r):
                # Locked: do not send any commands.
                if once:
                    logger.warning("Hardware locked; skipping signal dispatch (ONCE=1).")
                else:
                    logger.debug("Hardware locked; skipping signal dispatch.")
                if once:
                    return 0
                time.sleep(poll_seconds)
                continue

            # One-shot safety pulse test (Shift+C => Open Cargo)
            # Must only fire when HARDWARE_LOCK is disabled and there exists a prediction signal
            # at or above the confidence threshold.
            if pulse_test and r:
                signals = _get_pending_signals(r, confidence_threshold=confidence_threshold)
                if not signals:
                    logger.warning(
                        "Pulse test armed but no signals met confidence threshold (>= %.2f); not firing.",
                        confidence_threshold,
                    )
                    return 2

                # Bardol gate applies to all hardware actions in PRODUCTION mode.
                if sharpe_gate_enabled:
                    now_ts = time.time()
                    if last_sharpe_value is None or (now_ts - last_sharpe_check_ts) >= sharpe_refresh_seconds:
                        last_sharpe_value = _winsorized_sharpe_4h(
                            horizon_minutes=sharpe_horizon_minutes,
                            lookback_hours=sharpe_lookback_hours,
                            winsor_lo=sharpe_winsor_lo,
                            winsor_hi=sharpe_winsor_hi,
                            stop_loss_pct=sharpe_stop_loss_pct,
                        )
                        last_sharpe_check_ts = now_ts

                    sharpe_val = float(last_sharpe_value) if last_sharpe_value is not None else None
                    if sharpe_val is None:
                        logger.warning("🛑 Bardol gate: Sharpe unavailable; rejecting pulse test.")
                        return 3
                    if sharpe_val < min_sharpe_4h:
                        logger.warning(
                            "🛑 Bardol gate: winsorized 4h Sharpe %.3f < %.3f; rejecting pulse test (conf=%.2f type_id=%s)",
                            sharpe_val,
                            min_sharpe_4h,
                            float(signals[0].get("confidence", 0.0) or 0.0),
                            str(signals[0].get("type_id")),
                        )
                        return 3
                logger.info(
                    "🧪 Pulse test firing SHIFT_C (Open Cargo) gated by conf=%.2f (type_id=%s)",
                    float(signals[0].get("confidence", 0.0) or 0.0),
                    str(signals[0].get("type_id")),
                )
                ok = bridge.send_command("SHIFT_C")
                if ok:
                    try:
                        r.lpush(
                            "arduino:command_log",
                            json.dumps(
                                {
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "command": "SHIFT_C",
                                    "mode": "PULSE_TEST",
                                    "type_id": signals[0].get("type_id"),
                                    "confidence": float(signals[0].get("confidence", 0.0) or 0.0),
                                }
                            ),
                        )
                        r.ltrim("arduino:command_log", 0, 499)
                    except Exception:
                        pass
                    return 0
                return 1

            # Scan for high-confidence signals
            if r:
                signals = _get_pending_signals(r, confidence_threshold=confidence_threshold)
                for sig in signals:
                    # Bardol gate: block execution when recent winsorized Sharpe is too low.
                    if sharpe_gate_enabled:
                        now_ts = time.time()
                        if last_sharpe_value is None or (now_ts - last_sharpe_check_ts) >= sharpe_refresh_seconds:
                            last_sharpe_value = _winsorized_sharpe_4h(
                                horizon_minutes=sharpe_horizon_minutes,
                                lookback_hours=sharpe_lookback_hours,
                                winsor_lo=sharpe_winsor_lo,
                                winsor_hi=sharpe_winsor_hi,
                                stop_loss_pct=sharpe_stop_loss_pct,
                            )
                            last_sharpe_check_ts = now_ts

                        sharpe_val = float(last_sharpe_value) if last_sharpe_value is not None else None
                        if sharpe_val is None:
                            logger.warning("🛑 Bardol gate: Sharpe unavailable; rejecting trade.")
                            break
                        if sharpe_val < min_sharpe_4h:
                            logger.warning(
                                "🛑 Bardol gate: winsorized 4h Sharpe %.3f < %.3f; rejecting trade (conf=%.2f type_id=%s)",
                                sharpe_val,
                                min_sharpe_4h,
                                float(sig.get("confidence", 0.0) or 0.0),
                                str(sig.get("type_id")),
                            )
                            break

                    cmd = f"CLICK_{sig['signal_type']}"
                    logger.info(f"📡 Dispatching {cmd} for type_id={sig['type_id']} conf={sig['confidence']:.2f}")
                    if bridge.send_command(cmd):
                        # Record dispatch in Redis for audit
                        try:
                            r.lpush("arduino:command_log", json.dumps({
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "command": cmd,
                                "type_id": sig["type_id"],
                                "confidence": sig["confidence"],
                            }))
                            r.ltrim("arduino:command_log", 0, 499)  # keep last 500
                        except Exception:
                            pass
                    # Only one command per poll cycle to avoid rapid-fire
                    break

            if once:
                return 0

        except Exception as e:
            publish("AMBER", detail=json.dumps({"error": str(e)}))
            logger.error(f"Arduino bridge loop error: {e}")
            if once:
                return 1

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

