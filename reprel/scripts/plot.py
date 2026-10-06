"""Aggregate metrics.csv files into mean +/- std learning curves per condition.

    python -m reprel.scripts.plot --experiment outputs/taxi_task2 [--out plots]

Reads every ``<experiment>/<condition>/seed_*/metrics.csv`` (several experiment directories
may be given; conditions are labelled ``<experiment>/<condition>`` when more than one).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#4F6D7A", "#C0504D", "#5B9A8B", "#E0A458", "#7A5C9A", "#8D8D8D"]


def load_experiment(experiment: Path, label_prefix: str = "") -> pd.DataFrame:
    frames = []
    for metrics in sorted(experiment.glob("*/seed_*/metrics.csv")):
        df = pd.read_csv(metrics)
        df["condition"] = label_prefix + metrics.parent.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no metrics.csv under {experiment}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby(["condition", "env_steps"])[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    agg = aggregate(df, metric)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (condition, g) in enumerate(agg.groupby("condition", sort=False)):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(g["env_steps"], g["mean"], label=condition, color=color, linewidth=1.8)
        ax.fill_between(
            g["env_steps"], g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.18
        )
    stages = df.groupby("stage")["env_steps"].min()
    for boundary in stages.values[1:]:
        ax.axvline(boundary, color="#8D8D8D", linestyle=":", linewidth=1)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Final-point metrics per condition (mean/std over seeds) and steps to 90% success."""
    rows = []
    for condition, g in df.groupby("condition", sort=False):
        last = g[g["env_steps"] == g["env_steps"].max()]
        per_seed = g.groupby("seed")
        reached = [
            s["env_steps"][s["success_rate"] >= 0.9].min()
            if (s["success_rate"] >= 0.9).any()
            else float("nan")
            for _, s in per_seed
        ]
        rows.append(
            {
                "condition": condition,
                "seeds": last["seed"].nunique(),
                "final_steps": int(last["env_steps"].max()),
                "final_return_mean": last["eval_return_mean"].mean(),
                "final_return_std": last["eval_return_mean"].std(ddof=0),
                "final_success_mean": last["success_rate"].mean(),
                "final_success_std": last["success_rate"].std(ddof=0),
                "steps_to_90pct_success_median": pd.Series(reached).median(),
                "seeds_reaching_90pct": int(pd.Series(reached).notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", nargs="+", required=True)
    parser.add_argument("--out", default="plots")
    parser.add_argument("--name", default=None, help="output file prefix")
    args = parser.parse_args(argv)
    experiments = [Path(e) for e in args.experiment]
    multi = len(experiments) > 1
    df = pd.concat(
        [load_experiment(e, f"{e.name}/" if multi else "") for e in experiments], ignore_index=True
    )
    name = args.name or ("_".join(e.name for e in experiments) if multi else experiments[0].name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_metric(df, "eval_return_mean", "evaluation return", name, out / f"{name}_return.png")
    plot_metric(df, "success_rate", "success rate", name, out / f"{name}_success.png")
    summary(df).to_csv(out / f"{name}_summary.csv", index=False)
    print(summary(df).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
