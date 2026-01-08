import argparse
import json
import os
import glob
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


DEFAULT_PARQUET_URL = "http://192.168.14.105:8001/training_data_cleaned.parquet"


def sync_to_host(*, host: str, dest_dir: str) -> None:
    """Sync trained weights to the Debian host and restart the oracle container.

    Required behavior:
    - scp models/oracle_v1_*.pt seb@192.168.14.105:~/eve-trader/models/
    - restart oracle container on the server to load new weights

    This is best-effort: training should not be lost if the network is down.
    """

    user_host = f"seb@{host}"
    dest = f"{user_host}:{dest_dir}"

    # Ensure destination exists
    try:
        subprocess.run(
            ["ssh", user_host, f"mkdir -p {dest_dir}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        pass

    # Copy weights
    try:
        subprocess.run(
            ["scp"] + glob.glob("models/oracle_v1_*.pt") + [dest],
            check=True,
        )
        print(f"✅ Synced weights to {dest}")
    except Exception as e:
        print(f"⚠️ Weight sync failed (scp): {e}")
        return

    # Restart strategist on the server to load the new weights
    # (explicit requirement)
    remote_restart = "docker restart eve-trader-strategist"
    try:
        subprocess.run(["ssh", user_host, remote_restart], check=False)
        print("✅ Triggered remote restart to load new weights")
    except Exception as e:
        print(f"⚠️ Remote restart failed: {e}")


def _require_torch():
    try:
        import torch
        return torch
    except Exception as e:
        raise RuntimeError(
            "PyTorch is required for training. Install torch on the RTX 5090 environment. "
            f"Original error: {e}"
        )


@dataclass
class TrainingConfig:
    parquet_path: Path
    redis_url: str
    model_dir: Path
    lookback_minutes: int
    horizon_minutes: int
    bucket_minutes: int
    max_types: int
    max_samples: int
    batch_size: int
    epochs: int
    learning_rate: float
    set_live_after_train: bool


def _redis_client(redis_url: str):
    import redis

    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


def _is_http_url(value: str) -> bool:
    try:
        u = urlparse(value)
        return u.scheme in {"http", "https"}
    except Exception:
        return False


def _download_parquet(url: str, out_path: Path) -> None:
    import requests

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out_path)


def _cleanup_downloaded_parquets(data_dir: Path, keep: Path | None) -> None:
    # Only touch the files we generate automatically.
    pattern = str(data_dir / "oracle_training_*.parquet")
    for p in glob.glob(pattern):
        try:
            fp = Path(p)
            if keep is not None and fp.resolve() == keep.resolve():
                continue
            fp.unlink(missing_ok=True)
        except Exception:
            pass


def _build_time_series(df_orders: pd.DataFrame, bucket_minutes: int, max_types: int) -> pd.DataFrame:
    df = df_orders.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "type_id", "price", "volume_remaining", "is_buy_order"])

    df["type_id"] = df["type_id"].astype(int)
    df["price"] = df["price"].astype(float)
    df["volume_remaining"] = df["volume_remaining"].astype(int)
    df["is_buy_order"] = df["is_buy_order"].astype(bool)

    bucket = f"{int(bucket_minutes)}min"
    df["t"] = df["timestamp"].dt.floor(bucket)

    # Limit to most-active types by row count (keeps training bounded)
    type_counts = df.groupby("type_id").size().sort_values(ascending=False)
    keep_types = type_counts.head(max_types).index
    df = df[df["type_id"].isin(keep_types)]

    buy = df[df["is_buy_order"]]
    sell = df[~df["is_buy_order"]]

    best_bid = buy.groupby(["t", "type_id"])["price"].max().rename("best_bid")
    best_ask = sell.groupby(["t", "type_id"])["price"].min().rename("best_ask")

    buy_vol = buy.groupby(["t", "type_id"])["volume_remaining"].sum().rename("buy_vol")
    sell_vol = sell.groupby(["t", "type_id"])["volume_remaining"].sum().rename("sell_vol")

    features = pd.concat([best_bid, best_ask, buy_vol, sell_vol], axis=1).reset_index()

    features["market_price"] = features["best_ask"].fillna(features["best_bid"]).fillna(0.0)
    features["imbalance"] = (features["buy_vol"].fillna(0) + 1.0) / (features["sell_vol"].fillna(0) + 1.0)

    # Velocity per type over time buckets
    features = features.sort_values(["type_id", "t"])
    features["velocity"] = features.groupby("type_id")["market_price"].pct_change().replace([np.inf, -np.inf], np.nan)
    features["velocity"] = features["velocity"].fillna(0.0)

    # Keep only meaningful prices
    features = features[features["market_price"] > 0]
    return features


def _build_time_series_from_history(df_history: pd.DataFrame, bucket_minutes: int, max_types: int) -> pd.DataFrame:
    df = df_history.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "type_id"])

    df["type_id"] = df["type_id"].astype(int)

    # Price proxy: prefer avg_price, fall back to close
    price_col = None
    if "avg_price" in df.columns:
        price_col = "avg_price"
    elif "close" in df.columns:
        price_col = "close"
    else:
        raise RuntimeError("market_history Parquet missing avg_price/close price columns")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    bucket = f"{int(bucket_minutes)}min"
    df["t"] = df["timestamp"].dt.floor(bucket)

    # Limit to most-active types by row count (keeps training bounded)
    type_counts = df.groupby("type_id").size().sort_values(ascending=False)
    keep_types = type_counts.head(max_types).index
    df = df[df["type_id"].isin(keep_types)]

    # There should already be ~1 row per (t, type_id); average defensively
    grouped = df.groupby(["t", "type_id"], as_index=False)
    out = grouped.agg(
        market_price=(price_col, "mean"),
        order_book_imbalance=(
            "order_book_imbalance" if "order_book_imbalance" in df.columns else price_col,
            "mean",
        ),
    )

    # If we had to use a dummy column for imbalance, normalize to neutral
    if "order_book_imbalance" not in df.columns:
        out["order_book_imbalance"] = 1.0

    out["imbalance"] = pd.to_numeric(out["order_book_imbalance"], errors="coerce").fillna(1.0)

    out = out.sort_values(["type_id", "t"])
    out["velocity"] = out.groupby("type_id")["market_price"].pct_change().replace([np.inf, -np.inf], np.nan)
    out["velocity"] = out["velocity"].fillna(0.0)

    out = out[out["market_price"] > 0]
    return out[["t", "type_id", "market_price", "imbalance", "velocity"]]


def _make_sequences(
    df_feat: pd.DataFrame,
    lookback_steps: int,
    horizon_steps: int,
    max_samples: int,
):
    # Output arrays
    x_list = []
    h_list = []
    d_list = []
    y_list = []

    total = 0
    for _, g in df_feat.groupby("type_id"):
        g = g.sort_values("t")
        if len(g) < (lookback_steps + horizon_steps + 1):
            continue

        v = g["velocity"].to_numpy(dtype=np.float32)
        im = g["imbalance"].to_numpy(dtype=np.float32)
        price = g["market_price"].to_numpy(dtype=np.float32)

        t = pd.to_datetime(g["t"]).dt
        hours = t.hour.to_numpy(dtype=np.int64)
        days = t.dayofweek.to_numpy(dtype=np.int64)

        for end_idx in range(lookback_steps - 1, len(g) - horizon_steps):
            start_idx = end_idx - lookback_steps + 1
            target_idx = end_idx + horizon_steps

            x = np.stack([v[start_idx : end_idx + 1], im[start_idx : end_idx + 1]], axis=1)
            x_list.append(x)
            h_list.append(hours[start_idx : end_idx + 1])
            d_list.append(days[start_idx : end_idx + 1])
            y_list.append(price[target_idx])
            total += 1
            if max_samples and total >= max_samples:
                break

        if max_samples and total >= max_samples:
            break

    if total == 0:
        raise RuntimeError("No training sequences could be generated (insufficient time series depth).")

    X = np.asarray(x_list, dtype=np.float32)
    H = np.asarray(h_list, dtype=np.int64)
    D = np.asarray(d_list, dtype=np.int64)
    Y = np.asarray(y_list, dtype=np.float32)
    return X, H, D, Y


def train(cfg: TrainingConfig) -> str:
    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    from oracle.model import OracleTemporalTransformer

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires cuda.")

    r = _redis_client(cfg.redis_url)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    r.set("oracle:model_status", "TRAINING")

    gpu_name = None
    gpu_capability = None
    gpu_warning = None
    try:
        gpu_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        gpu_capability = f"{major}.{minor}"
        # RTX 5090 (sm_120) may require a newer/nightly Torch build.
        # Avoid spamming operators with a warning unless this Torch build does not advertise sm_120 support.
        if major >= 12:
            arch_list = None
            try:
                arch_list = torch.cuda.get_arch_list()
            except Exception:
                arch_list = None

            arch_ok = False
            if arch_list:
                arch_ok = any(a in {"sm_120", "compute_120"} for a in arch_list)

            if not arch_ok:
                torch_ver = getattr(torch, "__version__", "unknown")
                cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
                gpu_warning = (
                    f"Detected compute capability sm_{major}{minor} (e.g., RTX 5090). "
                    f"This Torch build may not include sm_120 kernels (torch={torch_ver}, cuda={cuda_ver}, arch_list={arch_list}). "
                    "If you hit errors like 'no kernel image is available for execution on the device', install a Torch build that supports sm_120."
                )
    except Exception:
        pass

    try:
        df_orders = pd.read_parquet(cfg.parquet_path)
        # Support both raw order snapshots and aggregated market_history exports.
        if {"price", "volume_remaining", "is_buy_order"}.issubset(df_orders.columns):
            df_feat = _build_time_series(df_orders, bucket_minutes=cfg.bucket_minutes, max_types=cfg.max_types)
        elif {"avg_price", "order_book_imbalance"}.intersection(df_orders.columns) and {"timestamp", "type_id"}.issubset(df_orders.columns):
            df_feat = _build_time_series_from_history(df_orders, bucket_minutes=cfg.bucket_minutes, max_types=cfg.max_types)
        elif {"close", "timestamp", "type_id"}.issubset(df_orders.columns):
            df_feat = _build_time_series_from_history(df_orders, bucket_minutes=cfg.bucket_minutes, max_types=cfg.max_types)
        else:
            raise RuntimeError(
                "Unsupported Parquet schema. Expected either market_orders-style columns "
                "(timestamp, type_id, price, volume_remaining, is_buy_order) or market_history-style columns "
                "(timestamp, type_id, avg_price/close, order_book_imbalance)."
            )

        lookback_steps = int(cfg.lookback_minutes / cfg.bucket_minutes)
        horizon_steps = int(cfg.horizon_minutes / cfg.bucket_minutes)
        if lookback_steps < 2 or horizon_steps < 1:
            raise ValueError("Invalid lookback/horizon for the selected bucket size.")

        unique_buckets = int(df_feat["t"].nunique()) if (not df_feat.empty and "t" in df_feat.columns) else 0
        min_required = lookback_steps + horizon_steps + 1
        if unique_buckets < min_required:
            # Emergency induction mode: allow bootstrapping the training pipeline from a single snapshot.
            # This is gated behind an env var so normal operators still get the safety check.
            allow_snapshot = os.getenv("ORACLE_ALLOW_SNAPSHOT_TRAINING", "0") == "1"
            if not allow_snapshot:
                raise RuntimeError(
                    "No training sequences could be generated (insufficient time series depth). "
                    f"Have {unique_buckets} time buckets; need at least {min_required}. "
                    "This typically means the Parquet represents a single snapshot, not rolling history. "
                    "Collect rolling features for >=6 hours (or reduce --lookback-minutes) and re-export."
                )

            if df_feat.empty or "t" not in df_feat.columns:
                raise RuntimeError("Snapshot training requested but feature frame is empty.")

            # Create synthetic buckets by repeating the latest bucket forward in time.
            # This allows sequence construction and proves GPU training is wired correctly.
            base_t = pd.to_datetime(df_feat["t"].max())
            synthetic = []
            for i in range(min_required):
                chunk = df_feat.copy()
                chunk["t"] = base_t + pd.to_timedelta(i * cfg.bucket_minutes, unit="m")
                synthetic.append(chunk)
            df_feat = pd.concat(synthetic, ignore_index=True)

        X, H, D, Y = _make_sequences(
            df_feat,
            lookback_steps=lookback_steps,
            horizon_steps=horizon_steps,
            max_samples=cfg.max_samples,
        )

        X_t = torch.from_numpy(X)
        H_t = torch.from_numpy(H)
        D_t = torch.from_numpy(D)
        Y_t = torch.from_numpy(Y).unsqueeze(-1)

        ds = TensorDataset(X_t, H_t, D_t, Y_t)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        model = OracleTemporalTransformer(input_dim=2, forecast_horizon=1).to(device)
        model.train()

        optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        loss_fn = torch.nn.MSELoss()

        last_loss = None
        for epoch in range(cfg.epochs):
            running = 0.0
            seen = 0
            for xb, hb, db, yb in dl:
                xb = xb.to(device)
                hb = hb.to(device)
                db = db.to(device)
                yb = yb.to(device)

                out = model(xb, hb, db)["predicted_price"]
                loss = loss_fn(out, yb)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                batch_size = xb.shape[0]
                running += float(loss.item()) * batch_size
                seen += batch_size

            last_loss = running / max(1, seen)
            print(f"Epoch {epoch+1}/{cfg.epochs} | loss={last_loss:.6f}")

        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.model_dir / f"oracle_v1_{ts}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "created_at": ts,
                "device": str(device),
                "input_dim": 2,
                "lookback_minutes": cfg.lookback_minutes,
                "horizon_minutes": cfg.horizon_minutes,
                "bucket_minutes": cfg.bucket_minutes,
            },
            model_path,
        )

        # Auto-sync weights to the Debian host (Body) and restart oracle to load.
        # Defaults match the production host unless overridden.
        sync_host = os.getenv("EVE_TRADER_SYNC_HOST", "192.168.14.105")
        sync_dest = os.getenv("EVE_TRADER_SYNC_DEST", "~/eve-trader/models/")
        sync_to_host(host=sync_host, dest_dir=sync_dest)

        metrics = {
            "timestamp": ts,
            "loss": last_loss,
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
            "gpu_warning": gpu_warning,
            "samples": int(len(ds)),
            "rows_in_parquet": int(len(df_orders)),
            "model_path": str(model_path),
            "lookback_minutes": cfg.lookback_minutes,
            "horizon_minutes": cfg.horizon_minutes,
            "bucket_minutes": cfg.bucket_minutes,
            "max_types": cfg.max_types,
        }
        r.set("oracle:last_training_metrics", json.dumps(metrics))
        r.set("oracle:model_latest_path", str(model_path))
        r.set("oracle:model_status", "LIVE" if cfg.set_live_after_train else "IDLE")
        return str(model_path)
    except Exception as e:
        err_metrics = {
            "timestamp": ts,
            "error": str(e),
            "device": str(device),
            "gpu_name": gpu_name,
            "gpu_capability": gpu_capability,
            "gpu_warning": gpu_warning,
            "parquet_path": str(cfg.parquet_path),
            "lookback_minutes": cfg.lookback_minutes,
            "horizon_minutes": cfg.horizon_minutes,
            "bucket_minutes": cfg.bucket_minutes,
        }
        try:
            r.set("oracle:last_training_metrics", json.dumps(err_metrics))
        except Exception:
            pass
        r.set("oracle:model_status", "IDLE")
        raise


def main():
    p = argparse.ArgumentParser(description="Train Oracle model from cleaned Parquet on RTX 5090.")
    p.add_argument(
        "--parquet",
        default=os.getenv("TRAINING_PARQUET", ""),
        help="Local parquet path. If omitted, the trainer will download from --parquet-url.",
    )
    p.add_argument(
        "--parquet-url",
        default=os.getenv("TRAINING_PARQUET_URL", DEFAULT_PARQUET_URL),
        help="Remote Parquet URL (Debian exporter).",
    )
    p.add_argument(
        "--data-dir",
        default=os.getenv("TRAINING_DATA_DIR", "data"),
        help="Directory for downloaded Parquet files.",
    )
    p.add_argument(
        "--cleanup-parquets",
        action="store_true",
        default=True,
        help="After successful training, delete downloaded oracle_training_*.parquet files.",
    )
    p.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://192.168.14.105:6379/0"),
        help="Redis URL for status + metrics logging (dashboard reads these).",
    )
    p.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "models"),
        help="Directory to write versioned models.",
    )
    p.add_argument("--lookback-minutes", type=int, default=int(os.getenv("LOOKBACK_MINUTES", "360")))
    p.add_argument("--horizon-minutes", type=int, default=int(os.getenv("HORIZON_MINUTES", "60")))
    p.add_argument("--bucket-minutes", type=int, default=int(os.getenv("BUCKET_MINUTES", "5")))
    p.add_argument("--max-types", type=int, default=int(os.getenv("MAX_TYPES", "500")))
    p.add_argument("--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", "200000")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "256")))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "3")))
    p.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001")))
    p.add_argument("--set-live", action="store_true", help="Set oracle:model_status=LIVE after training completes.")
    args = p.parse_args()

    # Decide parquet source.
    downloaded_path = None
    parquet_path = None
    if args.parquet and str(args.parquet).strip():
        if _is_http_url(args.parquet):
            # Support passing URL via --parquet for convenience.
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            downloaded_path = Path(args.data_dir) / f"oracle_training_{ts}.parquet"
            _download_parquet(args.parquet, downloaded_path)
            parquet_path = downloaded_path
        else:
            parquet_path = Path(args.parquet)
    else:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        downloaded_path = Path(args.data_dir) / f"oracle_training_{ts}.parquet"
        _download_parquet(args.parquet_url, downloaded_path)
        parquet_path = downloaded_path

    cfg = TrainingConfig(
        parquet_path=parquet_path,
        redis_url=args.redis_url,
        model_dir=Path(args.model_dir),
        lookback_minutes=args.lookback_minutes,
        horizon_minutes=args.horizon_minutes,
        bucket_minutes=args.bucket_minutes,
        max_types=args.max_types,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        set_live_after_train=bool(args.set_live),
    )

    try:
        model_path = train(cfg)
        print(f"✅ Saved model: {model_path}")
    finally:
        # Always cleanup downloaded parquet artifacts to keep the 5090 box lean.
        if bool(args.cleanup_parquets):
            _cleanup_downloaded_parquets(Path(args.data_dir), keep=None)
            if downloaded_path is not None:
                try:
                    downloaded_path.unlink(missing_ok=True)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
