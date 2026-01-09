#!/usr/bin/env python3
import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil


@dataclass(frozen=True)
class Candidate:
    path: Path
    mtime: float
    size: int


def _bytes(n: int) -> str:
    # Simple humanizer
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n = n / 1024
    return f"{n:.1f}TB"


def _disk_used_ratio(root: Path) -> float:
    try:
        disk = shutil.disk_usage(str(root))
        if disk.total <= 0:
            return 0.0
        return float(disk.used) / float(disk.total)
    except Exception:
        return 0.0


def _find_candidates(data_dir: Path, pattern: str) -> list[Candidate]:
    out: list[Candidate] = []
    for p in data_dir.glob(pattern):
        try:
            st = p.stat()
            out.append(Candidate(path=p, mtime=float(st.st_mtime), size=int(st.st_size)))
        except Exception:
            continue
    out.sort(key=lambda c: (c.mtime, c.path.as_posix()))
    return out


def _delete(p: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY_RUN delete {p}")
        return
    p.unlink(missing_ok=True)


def run_once(*, data_dir: Path, keep_hours: int, max_ratio: float, target_ratio: float, pattern: str, dry_run: bool) -> int:
    # We keep the CLI surface compatible with existing systemd units.
    # Behavior upgrade:
    # - Guardrail triggers on overall disk utilization (used/total), not just data_dir size.
    # - When guard trips, prune both old parquet exports and logs to prevent SSH lockout.

    root = Path(__file__).resolve().parents[1]
    data_dir = data_dir.resolve()
    logs_dir = (root / "logs").resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        return 2

    # 1) Standard retention: delete matching exports older than keep_hours.
    if keep_hours > 0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=int(keep_hours))
        cutoff_ts = cutoff.timestamp()

        # Parquet exports (data/)
        parquet_patterns = [
            str(pattern or "refinery_*.parquet"),
            "oracle_training_*.parquet",
            "training_data_*.parquet",
        ]
        for pat in parquet_patterns:
            for c in _find_candidates(data_dir, pat):
                if c.mtime < cutoff_ts:
                    print(
                        f"retention delete {c.path.as_posix()} size={_bytes(c.size)} mtime={datetime.fromtimestamp(c.mtime, tz=timezone.utc).isoformat()}"
                    )
                    _delete(c.path, dry_run=dry_run)

        # Logs (logs/)
        if logs_dir.exists() and logs_dir.is_dir():
            for c in _find_candidates(logs_dir, "*.log"):
                if c.mtime < cutoff_ts:
                    print(
                        f"retention delete {c.path.as_posix()} size={_bytes(c.size)} mtime={datetime.fromtimestamp(c.mtime, tz=timezone.utc).isoformat()}"
                    )
                    _delete(c.path, dry_run=dry_run)

    # 2) Guardrail: if overall disk utilization exceeds max_ratio, delete oldest exports/logs until target.
    ratio = _disk_used_ratio(data_dir)
    if ratio < max_ratio:
        disk = shutil.disk_usage(str(data_dir))
        print(
            f"ok disk_used={_bytes(int(disk.used))} disk_total={_bytes(int(disk.total))} ratio={ratio:.3f} (<{max_ratio})"
        )
        return 0

    disk = shutil.disk_usage(str(data_dir))
    print(
        f"GUARD TRIP disk_used={_bytes(int(disk.used))} disk_total={_bytes(int(disk.total))} ratio={ratio:.3f} (>= {max_ratio})"
    )

    candidates: list[Candidate] = []
    # Mix candidates across data/ and logs/, oldest-first.
    for pat in [str(pattern or "refinery_*.parquet"), "oracle_training_*.parquet", "training_data_*.parquet"]:
        candidates.extend(_find_candidates(data_dir, pat))
    if logs_dir.exists() and logs_dir.is_dir():
        candidates.extend(_find_candidates(logs_dir, "*.log"))
    candidates.sort(key=lambda c: (c.mtime, c.path.as_posix()))

    deleted = 0
    while candidates and ratio >= target_ratio:
        c = candidates.pop(0)
        print(f"guard delete {c.path.as_posix()} size={_bytes(c.size)}")
        _delete(c.path, dry_run=dry_run)
        deleted += 1
        ratio = _disk_used_ratio(data_dir)

    disk2 = shutil.disk_usage(str(data_dir))
    print(
        f"guard done deleted={deleted} disk_used={_bytes(int(disk2.used))} ratio={ratio:.3f} target<{target_ratio}"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Disk-aware 48h cleanup + hard guard for eve-trader/data")
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--keep-hours", type=int, default=int(os.getenv("DISK_GUARD_KEEP_HOURS", "48")))
    p.add_argument("--max-ratio", type=float, default=float(os.getenv("DISK_GUARD_MAX_RATIO", "0.75")))
    p.add_argument("--target-ratio", type=float, default=float(os.getenv("DISK_GUARD_TARGET_RATIO", "0.70")))
    p.add_argument("--pattern", default=os.getenv("DISK_GUARD_PATTERN", "refinery_*.parquet"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.keep_hours <= 0:
        print("ERROR: --keep-hours must be > 0", file=sys.stderr)
        return 2
    if not (0.0 < args.target_ratio <= args.max_ratio <= 1.0):
        print("ERROR: require 0 < target_ratio <= max_ratio <= 1", file=sys.stderr)
        return 2

    return run_once(
        data_dir=Path(args.data_dir),
        keep_hours=int(args.keep_hours),
        max_ratio=float(args.max_ratio),
        target_ratio=float(args.target_ratio),
        pattern=str(args.pattern),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
