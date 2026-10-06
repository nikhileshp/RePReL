import csv
import json
from pathlib import Path

from reprel.scripts import explore as explore_script
from reprel.scripts import plot as plot_script
from reprel.scripts import run as run_script

CFG = """
name: smoke
condition: {condition}
domain: {{name: taxi, num_passengers: 1, layout: five, max_steps: 60}}
abstraction: {{name: dfoci, spec: taxi}}
agent: {{alpha: 0.1, gamma: 0.95, terminal_reward: 10.0}}
epsilon: {{schedule: constant, epsilon: 0.3}}
training: {{total_steps: 600, eval_every: 300, eval_episodes: 2, eval_epsilon: 0.0}}
logging: {{transitions: true}}
seeds: [0, 1]
"""


def write(tmp_path: Path, condition: str, extra: str = "") -> Path:
    p = tmp_path / f"{condition}.yaml"
    p.write_text(CFG.format(condition=condition) + extra)
    return p


def test_run_script_writes_metrics_metadata_agents_and_transitions(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for condition in ("flat", "reprel_none", "reprel_dfoci"):
        code = run_script.main(["--config", str(write(tmp_path, condition)), "--out", str(out)])
        assert code == 0
        for seed in (0, 1):
            run_dir = out / "smoke" / condition / f"seed_{seed}"
            assert (run_dir / "metrics.csv").exists()
            assert (run_dir / "agents.pkl").exists()
            assert (run_dir / "transitions.parquet").exists()
            meta = json.loads((run_dir / "meta.json").read_text())
            assert meta["seed"] == seed and "git_hash" in meta
            rows = list(csv.DictReader(open(run_dir / "metrics.csv")))
            assert rows[0]["env_steps"] == "0"  # initial evaluation
            assert int(rows[-1]["env_steps"]) >= 600
            assert rows[-1]["condition"] == condition and rows[-1]["stage"] == "0"


def test_run_script_single_seed_and_override(tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = run_script.main(
        [
            "--config",
            str(write(tmp_path, "reprel_dfoci")),
            "--out",
            str(out),
            "--seed",
            "5",
            "--set",
            "training.total_steps=200",
        ]
    )
    assert code == 0
    assert (out / "smoke" / "reprel_dfoci" / "seed_5" / "metrics.csv").exists()
    cfg = (out / "smoke" / "reprel_dfoci" / "seed_5" / "config.yaml").read_text()
    assert "total_steps: 200" in cfg


def test_run_script_transfer_stages_continue_the_same_agents(tmp_path: Path) -> None:
    extra = """
transfer:
  stages:
    - {num_passengers: 1, total_steps: 300}
    - {num_passengers: 2, total_steps: 300}
"""
    out = tmp_path / "out"
    assert (
        run_script.main(
            [
                "--config",
                str(write(tmp_path, "reprel_dfoci", extra)),
                "--out",
                str(out),
                "--seed",
                "0",
            ]
        )
        == 0
    )
    rows = list(csv.DictReader(open(out / "smoke" / "reprel_dfoci" / "seed_0" / "metrics.csv")))
    stages = {r["stage"] for r in rows}
    assert stages == {"0", "1"}
    stage1 = [r for r in rows if r["stage"] == "1"]
    assert stage1[0]["env_steps"] == "300"  # zero-shot evaluation at the start of stage 1
    assert int(rows[-1]["env_steps"]) >= 600


def test_explore_script_collects_transitions(tmp_path: Path) -> None:
    cfg = tmp_path / "explore.yaml"
    cfg.write_text("""
name: explore_smoke
domain: {name: taxi, num_passengers: 2, layout: five, max_steps: 60}
explore: {episodes: 3, max_operator_steps: 10}
seeds: [0]
""")
    out = tmp_path / "out"
    assert explore_script.main(["--config", str(cfg), "--out", str(out)]) == 0
    assert (out / "explore_smoke" / "seed_0" / "transitions.parquet").exists()
    assert (
        json.loads((out / "explore_smoke" / "seed_0" / "stats.json").read_text())["episodes"] == 3
    )


def test_plot_script_aggregates_conditions(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for condition in ("flat", "reprel_dfoci"):
        run_script.main(["--config", str(write(tmp_path, condition)), "--out", str(out)])
    plots = tmp_path / "plots"
    assert plot_script.main(["--experiment", str(out / "smoke"), "--out", str(plots)]) == 0
    assert (plots / "smoke_return.png").exists() and (plots / "smoke_success.png").exists()
    assert (plots / "smoke_summary.csv").exists()
