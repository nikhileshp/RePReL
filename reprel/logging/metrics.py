"""Learning-curve metrics (CSV) and run provenance (config, git hash, seed, environment)."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import yaml

METRIC_COLUMNS = (
    "condition",
    "seed",
    "stage",
    "env_steps",
    "episodes",
    "eval_return_mean",
    "eval_return_std",
    "success_rate",
    "mean_steps_to_success",
    "epsilon",
)


class MetricsLogger:
    """Appends one CSV row per evaluation point; the header is written on open."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=METRIC_COLUMNS)
        self._writer.writeheader()

    def log(self, **row: Any) -> None:
        unknown = set(row) - set(METRIC_COLUMNS)
        if unknown:
            raise ValueError(f"unknown metric columns {sorted(unknown)}")
        self._writer.writerow({c: row.get(c, "") for c in METRIC_COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def git_info() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_hash": None, "git_dirty": None}
    return {"git_hash": commit, "git_dirty": dirty}


def write_run_metadata(out_dir: str | Path, config: Mapping[str, Any], seed: int) -> None:
    """Write ``config.yaml`` and ``meta.json`` (git hash, seed, timestamp, versions)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(config), fh, sort_keys=False)
    meta = {
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        **git_info(),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
