#!/usr/bin/env python3
"""Record real nvidia-smi samples until interrupted."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

QUERY = "index,name,uuid,driver_version,memory.used,memory.total,utilization.gpu"


def sample() -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "idle",
            "warm",
            "one_session",
            "two_sessions",
            "four_sessions",
            "six_sessions",
        ),
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 records until interrupted")
    args = parser.parse_args()
    if args.interval <= 0 or args.duration < 0:
        parser.error("interval must be positive and duration cannot be negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration if args.duration else None
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["timestamp", "phase", *QUERY.split(",")])
        try:
            while deadline is None or time.monotonic() < deadline:
                timestamp = datetime.now(UTC).isoformat()
                for row in sample():
                    writer.writerow([timestamp, args.phase, *row])
                destination.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
