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


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Avoid descending into heavy/critical state.
        # We never delete inside these, and they can be root-owned.
        dirnames[:] = [d for d in dirnames if d not in {"postgres", "redis"}]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                total += p.stat().st_size
            except Exception:
                continue
    return int(total)


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
    data_dir = data_dir.resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        return 2

    disk = shutil.disk_usage(str(data_dir))
    disk_total = int(disk.total)

    # 1) Standard retention: delete matching files older than keep_hours.
    if keep_hours > 0:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=int(keep_hours))
        cutoff_ts = cutoff.timestamp()
        candidates = _find_candidates(data_dir, pattern)
        old = [c for c in candidates if c.mtime < cutoff_ts]
        for c in old:
            print(f"retention delete {c.path.name} size={_bytes(c.size)} mtime={datetime.fromtimestamp(c.mtime, tz=timezone.utc).isoformat()}")
            _delete(c.path, dry_run=dry_run)

    # 2) Guardrail: if data_dir grows beyond max_ratio of total disk, delete oldest matching files until target.
    data_size = _dir_size_bytes(data_dir)
    ratio = (data_size / disk_total) if disk_total > 0 else 0.0

    if ratio < max_ratio:
        print(f"ok data_size={_bytes(data_size)} disk_total={_bytes(disk_total)} ratio={ratio:.3f} (<{max_ratio})")
        return 0

    print(f"GUARD TRIP data_size={_bytes(data_size)} disk_total={_bytes(disk_total)} ratio={ratio:.3f} (>= {max_ratio})")
    candidates = _find_candidates(data_dir, pattern)
    deleted = 0
    while candidates and ratio >= target_ratio:
        c = candidates.pop(0)
        print(f"guard delete {c.path.name} size={_bytes(c.size)}")
        _delete(c.path, dry_run=dry_run)
        deleted += 1
        data_size = _dir_size_bytes(data_dir)
        ratio = (data_size / disk_total) if disk_total > 0 else 0.0

    print(f"guard done deleted={deleted} data_size={_bytes(data_size)} ratio={ratio:.3f} target<{target_ratio}")
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
