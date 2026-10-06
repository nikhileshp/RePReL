"""Collect exploration-only transition data inside planned subtasks (input for M7).

python -m reprel.scripts.explore --config reprel/configs/taxi_explore.yaml [--out outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reprel.config import load_config, parse_overrides
from reprel.execution import ExplorationCollector
from reprel.logging import write_run_metadata
from reprel.seeding import seed_everything

from .common import build_domain, build_planner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    out = Path(args.out or cfg.output_dir)
    for seed in cfg.seeds:
        run_dir = out / cfg.name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_run_metadata(run_dir, cfg.to_dict(), seed)
        rng = seed_everything(seed)
        domain = build_domain(cfg)
        collector = ExplorationCollector(
            domain, build_planner(cfg), max_operator_steps=cfg.explore.max_operator_steps
        )
        stats = collector.collect(
            rng,
            cfg.explore.episodes,
            run_dir / "transitions.parquet",
            run_id=f"{cfg.name}/{seed}",
            seed=seed,
        )
        (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        print(f"[{cfg.name}/seed_{seed}] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
