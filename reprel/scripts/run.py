"""Run one experiment config across its seeds.

    python -m reprel.scripts.run --config reprel/configs/taxi_reprel_dfoci.yaml [--seed K]
        [--set training.total_steps=200000 ...] [--out outputs] [--workers N]

Each seed writes ``<out>/<name>/<condition>/seed_<k>/`` with config.yaml, meta.json,
metrics.csv, agents.pkl and (optionally) transitions.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

from reprel.config import RunConfig, load_config, parse_overrides
from reprel.execution import TrainingPoint, train
from reprel.logging import MetricsLogger, TransitionLogger, write_run_metadata
from reprel.seeding import seed_everything

from .common import build_domain, build_runner, build_schedule


def run_seed(cfg: RunConfig, seed: int, out_root: Path) -> Path:
    run_dir = out_root / cfg.name / cfg.condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, cfg.to_dict(), seed)
    rng = seed_everything(seed)
    eval_rng = seed_everything(seed + 10_007)
    stages = (
        [(s.domain_params, s.total_steps) for s in cfg.transfer.stages]
        if cfg.transfer is not None
        else [({}, cfg.training.total_steps)]
    )
    schedule = build_schedule(cfg)
    pool = None
    offset = 0
    t0 = time.time()
    logger = (
        TransitionLogger(run_dir / "transitions.parquet", run_id=f"{cfg.name}/{seed}", seed=seed)
        if cfg.logging.transitions
        else None
    )
    with MetricsLogger(run_dir / "metrics.csv") as metrics:
        for stage_idx, (domain_params, total_steps) in enumerate(stages):
            domain = build_domain(cfg, domain_params)
            runner, pool = build_runner(cfg, domain, pool)

            def on_eval(p: TrainingPoint, stage: int = stage_idx) -> None:
                metrics.log(
                    condition=cfg.condition,
                    seed=seed,
                    stage=stage,
                    env_steps=p.env_steps,
                    episodes=p.episodes,
                    eval_return_mean=p.eval.return_mean,
                    eval_return_std=p.eval.return_std,
                    success_rate=p.eval.success_rate,
                    mean_steps_to_success=p.eval.mean_steps_to_success,
                    epsilon=p.epsilon,
                )
                print(
                    f"[{cfg.name}/{cfg.condition}/seed_{seed}] stage {stage} steps {p.env_steps} "
                    f"return {p.eval.return_mean:.1f} success {p.eval.success_rate:.2f} "
                    f"eps {p.epsilon:.2f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

            history = train(
                runner,
                rng,
                total_steps=total_steps,
                eval_every=cfg.training.eval_every,
                eval_episodes=cfg.training.eval_episodes,
                eval_epsilon=cfg.training.eval_epsilon,
                schedule=schedule,
                eval_rng=eval_rng,
                logger=logger,
                on_eval=on_eval,
                initial_eval=True,
                step_offset=offset,
            )
            offset = history[-1].env_steps
            pool.save(run_dir / "agents.pkl")
    if logger is not None:
        logger.close()
    return run_dir


def _worker(args: tuple[RunConfig, int, str]) -> str:
    cfg, seed, out = args
    return str(run_seed(cfg, seed, Path(out)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="run a single seed")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    parser.add_argument("--out", default=None, help="output root (default: config output_dir)")
    parser.add_argument("--workers", type=int, default=1, help="parallel seeds")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    out = Path(args.out or cfg.output_dir)
    seeds = [args.seed] if args.seed is not None else list(cfg.seeds)
    if args.workers > 1 and len(seeds) > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            for run_dir in pool.imap_unordered(_worker, [(cfg, s, str(out)) for s in seeds]):
                print(f"finished {run_dir}", flush=True)
    else:
        for seed in seeds:
            print(f"finished {run_seed(cfg, seed, out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
