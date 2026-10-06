"""Run one experiment config across its seeds.

    python -m reprel.scripts.run --config reprel/configs/taxi_reprel_dfoci.yaml [--seed K]
        [--set training.total_steps=200000 ...] [--out outputs] [--workers N] [--overwrite]

Each seed writes ``<out>/<name>/<condition>/seed_<k>/`` with config.yaml, meta.json,
metrics.csv, agents.pkl and (optionally) transitions.parquet. Transfer runs train the same
agents through successive stages; step counts and the epsilon schedule continue across stages.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from reprel.config import RunConfig, load_config, parse_overrides
from reprel.execution import TrainingPoint, train
from reprel.logging import MetricsLogger, TransitionLogger, write_run_metadata
from reprel.seeding import seed_everything

from .common import build_domain, build_runner, build_schedule


def stage_list(cfg: RunConfig) -> list[tuple[dict[str, Any], int]]:
    if cfg.transfer is not None:
        return [(dict(s.domain_params), s.total_steps) for s in cfg.transfer.stages]
    return [({}, cfg.training.total_steps)]


def validate(cfg: RunConfig) -> None:
    """Build every stage's domain, the planner and the schedule before any training starts."""
    for params, _ in stage_list(cfg):
        build_domain(cfg, params)
    build_schedule(cfg)
    build_runner(cfg, build_domain(cfg))


def run_seed(cfg: RunConfig, seed: int, out_root: Path, overwrite: bool = False) -> Path:
    run_dir = out_root / cfg.name / cfg.condition / f"seed_{seed}"
    if (run_dir / "metrics.csv").exists() and not overwrite:
        raise FileExistsError(f"{run_dir} already has results; pass --overwrite to redo it")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, cfg.to_dict(), seed)
    rng = seed_everything(seed)
    eval_rng = seed_everything(seed + 10_007)
    schedule = build_schedule(cfg)
    pool = None
    offset, episodes = 0, 0
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        metrics = stack.enter_context(MetricsLogger(run_dir / "metrics.csv"))
        logger = None
        if cfg.logging.transitions:
            logger = stack.enter_context(
                TransitionLogger(
                    run_dir / "transitions.parquet", run_id=f"{cfg.name}/{seed}", seed=seed
                )
            )
        for stage_idx, (domain_params, total_steps) in enumerate(stage_list(cfg)):
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
                episode_offset=episodes,
            )
            offset = history[-1].env_steps
            episodes = history[-1].episodes
            pool.save(run_dir / "agents.pkl")
    return run_dir


def _worker(args: tuple[RunConfig, int, str, bool]) -> tuple[int, str | None]:
    cfg, seed, out, overwrite = args
    try:
        run_seed(cfg, seed, Path(out), overwrite)
        return seed, None
    except Exception:  # noqa: BLE001 - reported to the parent, which exits non-zero
        return seed, traceback.format_exc()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="run a single seed")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    parser.add_argument("--out", default=None, help="output root (default: config output_dir)")
    parser.add_argument("--workers", type=int, default=1, help="parallel seeds")
    parser.add_argument("--overwrite", action="store_true", help="redo runs that have results")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    out = Path(args.out or cfg.output_dir)
    seeds = [args.seed] if args.seed is not None else list(cfg.seeds)
    try:
        validate(cfg)
    except Exception as exc:  # noqa: BLE001 - any config problem is reported before running
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2
    failures: list[tuple[int, str]] = []
    jobs = [(cfg, s, str(out), args.overwrite) for s in seeds]
    if args.workers > 1 and len(seeds) > 1:
        with get_context("spawn").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_worker, jobs))
    else:
        results = [_worker(job) for job in jobs]
    for seed, error in results:
        print(f"seed {seed}: {'FAILED' if error else 'finished'}", flush=True)
        if error:
            failures.append((seed, error))
    for seed, error in failures:
        print(f"--- seed {seed} failed ---\n{error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
