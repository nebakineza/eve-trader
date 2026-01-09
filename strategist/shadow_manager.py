import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import sqlalchemy
from sqlalchemy import text


logger = logging.getLogger("ShadowManager")


@dataclass(frozen=True)
class ShadowConfig:
    horizon_minutes: int = 60
    confidence_threshold: float = 0.6
    fee_rate: float = 0.025  # 2026 simulated tax/fee


class ShadowManager:
    def __init__(self, database_url: str, config: ShadowConfig | None = None):
        self.config = config or ShadowConfig(
            horizon_minutes=int(os.getenv("SHADOW_HORIZON_MINUTES", "60")),
            confidence_threshold=float(os.getenv("SHADOW_CONFIDENCE_THRESHOLD", "0.6")),
            fee_rate=float(os.getenv("SHADOW_FEE_RATE", "0.025")),
        )

        # Strategist typically passes a sync postgres URL already.
        sync_url = database_url.replace("asyncpg", "psycopg2")
        self.engine = sqlalchemy.create_engine(sync_url)

    def ensure_schema(self) -> None:
        sql_path = os.path.join(os.getcwd(), "storage", "shadow_ledger.sql")
        try:
            with open(sql_path, "r", encoding="utf-8") as f:
                ddl = f.read()
        except FileNotFoundError:
            logger.warning("Shadow ledger SQL not found; skipping schema init")
            return

        with self.engine.begin() as conn:
            # Execute statement-by-statement for compatibility.
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if not stmt:
                    continue
                conn.execute(text(stmt))

    def maybe_record_intended_trade(
        self,
        *,
        type_id: int,
        predicted_price: float | None,
        current_price: float | None,
        confidence: float,
    ) -> bool:
        if confidence is None or confidence < self.config.confidence_threshold:
            return False

        if current_price is None or predicted_price is None:
            return False

        signal_type = "BUY" if predicted_price >= current_price else "SELL"

        edge_pct = 0.0
        try:
            if float(current_price) > 0:
                edge_pct = (float(predicted_price) - float(current_price)) / float(current_price)
        except Exception:
            edge_pct = 0.0

        try:
            reasoning = (
                f"{signal_type} | conf={float(confidence):.3f} | "
                f"pred={float(predicted_price):.4f} vs px={float(current_price):.4f} | "
                f"edge={edge_pct:+.2%}"
            )
        except Exception:
            reasoning = f"{signal_type} | conf={confidence}"

        # Dedupe: avoid spamming repeated signals for the same type_id.
        # If there is already a PENDING entry within the last horizon window, skip.
        with self.engine.begin() as conn:
            existing = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM shadow_trades
                    WHERE type_id = :tid
                      AND virtual_outcome = 'PENDING'
                      AND timestamp >= NOW() - (:horizon || ' minutes')::interval
                    LIMIT 1
                    """
                ),
                {"tid": int(type_id), "horizon": int(self.config.horizon_minutes)},
            ).fetchone()

            if existing:
                return False

            conn.execute(
                text(
                    """
                    INSERT INTO shadow_trades (type_id, signal_type, predicted_price, actual_price_at_time, reasoning, virtual_outcome)
                    VALUES (:tid, :signal_type, :predicted, :entry, :reasoning, 'PENDING')
                    """
                ),
                {
                    "tid": int(type_id),
                    "signal_type": signal_type,
                    "predicted": float(predicted_price),
                    "entry": float(current_price),
                    "reasoning": reasoning,
                },
            )

        logger.info(
            "Shadow trade recorded: type_id=%s signal=%s conf=%.3f entry=%.4f predicted=%.4f",
            type_id,
            signal_type,
            confidence,
            current_price,
            predicted_price,
        )
        return True

    def settle_pending_trades(self, *, stop_loss_pct_override: float | None = None) -> int:
        horizon = timedelta(minutes=self.config.horizon_minutes)

        stop_loss_pct = float(os.getenv("SHADOW_STOP_LOSS_PCT", "0.015"))
        if stop_loss_pct_override is not None:
            try:
                ov = float(stop_loss_pct_override)
                if 0.0 < ov < 1.0:
                    stop_loss_pct = ov
            except Exception:
                pass
        winsor_lo = float(os.getenv("SHADOW_RETURN_WINSOR_LO", "-1.0"))
        winsor_hi = float(os.getenv("SHADOW_RETURN_WINSOR_HI", "10.0"))

        # Stop-loss first: if price moves adversely beyond threshold, force an early exit.
        # Then, for trades older than horizon, find the first market_history close at/after timestamp+horizon.
        # If data isn't available yet, keep it pending.
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT timestamp, type_id, signal_type, actual_price_at_time
                    FROM shadow_trades
                    WHERE virtual_outcome = 'PENDING'
                    ORDER BY timestamp ASC
                    LIMIT 200
                    """
                ),
                {},
            ).fetchall()

            settled = 0
            for ts, type_id, signal_type, entry_price in rows:
                entry = float(entry_price) if entry_price is not None else 0.0
                if entry <= 0.0:
                    continue

                # Only check stop-loss while the trade is still within the horizon window.
                older = conn.execute(
                    text(
                        """
                        SELECT 1
                        WHERE :ts <= NOW() - (:horizon || ' minutes')::interval
                        """
                    ),
                    {"ts": ts, "horizon": int(self.config.horizon_minutes)},
                ).fetchone()

                forced = False

                # Forced-exit check (best-effort):
                # BUY: if low <= entry*(1-stop_loss_pct)
                # SELL: if high >= entry*(1+stop_loss_pct)
                if not older:
                    try:
                        if str(signal_type).upper() == "BUY":
                            threshold = entry * (1.0 - stop_loss_pct)
                            row = conn.execute(
                                text(
                                    """
                                    SELECT MIN(low)
                                    FROM market_history
                                    WHERE type_id = :tid
                                      AND timestamp >= :ts
                                      AND timestamp <= (:ts + (:horizon || ' minutes')::interval)
                                    """
                                ),
                                {"tid": int(type_id), "ts": ts, "horizon": int(self.config.horizon_minutes)},
                            ).fetchone()
                            min_low = float(row[0]) if row and row[0] is not None else None
                            if min_low is not None and min_low > 0 and min_low <= threshold:
                                forced = True
                        else:
                            threshold = entry * (1.0 + stop_loss_pct)
                            row = conn.execute(
                                text(
                                    """
                                    SELECT MAX(high)
                                    FROM market_history
                                    WHERE type_id = :tid
                                      AND timestamp >= :ts
                                      AND timestamp <= (:ts + (:horizon || ' minutes')::interval)
                                    """
                                ),
                                {"tid": int(type_id), "ts": ts, "horizon": int(self.config.horizon_minutes)},
                            ).fetchone()
                            max_high = float(row[0]) if row and row[0] is not None else None
                            if max_high is not None and max_high > 0 and max_high >= threshold:
                                forced = True
                    except Exception:
                        forced = False

                if forced:
                    conn.execute(
                        text(
                            """
                            UPDATE shadow_trades
                            SET virtual_outcome = 'LOSS', forced_exit = TRUE
                            WHERE timestamp = :ts AND type_id = :tid AND virtual_outcome = 'PENDING'
                            """
                        ),
                        {"ts": ts, "tid": int(type_id)},
                    )
                    settled += 1
                    logger.info(
                        "Shadow trade forced-exit: type_id=%s signal=%s entry=%.4f stop_loss_pct=%.4f",
                        type_id,
                        signal_type,
                        entry,
                        stop_loss_pct,
                    )
                    continue

                # Only settle on horizon once old enough.
                if not older:
                    continue

                exit_row = conn.execute(
                    text(
                        """
                        SELECT close
                        FROM market_history
                        WHERE type_id = :tid
                          AND timestamp >= (:ts + (:horizon || ' minutes')::interval)
                        ORDER BY timestamp ASC
                        LIMIT 1
                        """
                    ),
                    {"tid": int(type_id), "ts": ts, "horizon": int(self.config.horizon_minutes)},
                ).fetchone()

                if not exit_row:
                    continue

                exit_price = float(exit_row[0])
                entry = float(entry_price) if entry_price is not None else 0.0

                if signal_type == "BUY":
                    gross = exit_price - entry
                    outcome = "WIN" if gross > 0 else "LOSS"
                else:
                    gross = entry - exit_price
                    outcome = "WIN" if gross > 0 else "LOSS"

                net = gross * (1.0 - self.config.fee_rate)

                # Normalized return (winsorized) for audit consistency.
                ret = (net / entry) if entry > 0 else 0.0
                if ret < winsor_lo:
                    ret = winsor_lo
                elif ret > winsor_hi:
                    ret = winsor_hi

                conn.execute(
                    text(
                        """
                        UPDATE shadow_trades
                        SET virtual_outcome = :outcome
                        WHERE timestamp = :ts AND type_id = :tid
                        """
                    ),
                    {"outcome": outcome, "ts": ts, "tid": int(type_id)},
                )

                settled += 1
                logger.info(
                    "Shadow trade settled: type_id=%s signal=%s entry=%.4f exit=%.4f outcome=%s net_alpha=%.4f ret=%.6f",
                    type_id,
                    signal_type,
                    entry,
                    exit_price,
                    outcome,
                    net,
                    ret,
                )

        return settled
