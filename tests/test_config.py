from pathlib import Path

import pytest

from reprel.config import RunConfig, load_config, parse_overrides


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return p


MINIMAL = """
name: t
condition: reprel_dfoci
domain: {name: taxi, num_passengers: 1, layout: five, max_steps: 50}
abstraction: {name: dfoci, spec: taxi}
agent: {alpha: 0.1, gamma: 0.95, terminal_reward: 10.0}
epsilon: {schedule: linear, start: 0.5, end: 0.01, episodes: 100}
training: {total_steps: 500, eval_every: 250, eval_episodes: 2, eval_epsilon: 0.0}
seeds: [0, 1]
"""


def test_load_minimal_config_fills_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, MINIMAL))
    assert isinstance(cfg, RunConfig)
    assert cfg.name == "t" and cfg.condition == "reprel_dfoci"
    assert cfg.domain.name == "taxi" and cfg.domain.params["num_passengers"] == 1
    assert cfg.abstraction.name == "dfoci" and cfg.abstraction.params["spec"] == "taxi"
    assert cfg.executor.max_operator_steps is None and cfg.executor.replan_on_timeout
    assert cfg.logging.transitions is False
    assert cfg.transfer is None
    assert cfg.seeds == (0, 1)
    assert cfg.output_dir == "outputs"


def test_overrides_are_typed_and_nested(tmp_path: Path) -> None:
    cfg = load_config(
        write(tmp_path, MINIMAL),
        parse_overrides(["domain.num_passengers=3", "agent.alpha=0.5", "seeds=[7]", "name=x"]),
    )
    assert cfg.domain.params["num_passengers"] == 3
    assert cfg.agent.alpha == 0.5
    assert cfg.seeds == (7,)
    assert cfg.name == "x"


def test_unknown_condition_or_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="condition"):
        load_config(write(tmp_path, MINIMAL.replace("reprel_dfoci", "magic")))
    with pytest.raises(ValueError, match="bogus"):
        load_config(write(tmp_path, MINIMAL + "\nbogus: 1\n"))


def test_transfer_stages(tmp_path: Path) -> None:
    text = (
        MINIMAL
        + """
transfer:
  stages:
    - {num_passengers: 1, total_steps: 100}
    - {num_passengers: 2, total_steps: 200}
"""
    )
    cfg = load_config(write(tmp_path, text))
    assert cfg.transfer is not None and len(cfg.transfer.stages) == 2
    assert cfg.transfer.stages[1].domain_params == {"num_passengers": 2}
    assert cfg.transfer.stages[1].total_steps == 200


def test_to_dict_round_trip(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, MINIMAL))
    assert load_config(write(tmp_path, __import__("yaml").safe_dump(cfg.to_dict()))) == cfg
