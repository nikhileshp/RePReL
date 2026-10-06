import json
from pathlib import Path

import numpy as np

from reprel.core.atoms import Atom
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.logging.metrics import MetricsLogger, write_run_metadata
from reprel.logging.transitions import TransitionLogger, TransitionRecord, read_transitions


def test_transition_log_round_trips_states(tmp_path: Path) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2))
    rng = np.random.default_rng(0)
    s = dom.reset(rng)
    path = tmp_path / "transitions.parquet"
    logger = TransitionLogger(path, run_id="r0", seed=0)
    expected: list[TransitionRecord] = []
    for t in range(20):
        action = dom.actions[int(rng.integers(dom.n_actions))]
        tr = dom.step(s, action, rng)
        rec = TransitionRecord(
            episode=0,
            t=t,
            operator="pickup",
            args=("p1",),
            state=s,
            action=action,
            reward=tr.reward,
            done=tr.done,
            next_state=tr.next_state,
            terminated=False,
        )
        logger.log(rec)
        expected.append(rec)
        s = tr.next_state
    logger.close()
    back = read_transitions(path)
    assert len(back) == 20
    for a, b in zip(back, expected, strict=True):
        assert a.state == b.state and a.next_state == b.next_state
        assert a.state.objects == b.state.objects
        assert (a.operator, a.args, a.action, a.reward, a.done) == (
            b.operator,
            b.args,
            b.action,
            b.reward,
            b.done,
        )
        assert a.run_id == "r0" and a.seed == 0


def test_transition_log_filters_by_operator_and_reports_changes(tmp_path: Path) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(1))
    path = tmp_path / "t.parquet"
    with TransitionLogger(path, run_id="r", seed=1) as logger:
        tr = dom.step(s, "north", np.random.default_rng(0))
        logger.log(
            TransitionRecord(
                0, 0, "pickup", ("p1",), s, "north", tr.reward, False, tr.next_state, False
            )
        )
        logger.log(TransitionRecord(0, 1, "drop", ("p1",), s, "south", -0.1, False, s, False))
    pickup_only = read_transitions(path, operator="pickup")
    assert [r.operator for r in pickup_only] == ["pickup"]
    added, removed = pickup_only[0].effects()
    assert all(a.pred == "at" and a.args[0] == "taxi" for a in added | removed)
    assert read_transitions(path, operator="drop")[0].effects() == (frozenset(), frozenset())


def test_jsonl_fallback(tmp_path: Path) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(2))
    path = tmp_path / "t.jsonl"
    with TransitionLogger(path, run_id="r", seed=2) as logger:
        logger.log(TransitionRecord(0, 0, "pickup", ("p1",), s, "east", -0.1, False, s, False))
    assert read_transitions(path)[0].state == s
    assert path.read_text().count("\n") == 1


def test_metrics_logger_writes_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with MetricsLogger(path) as m:
        m.log(
            condition="reprel_dfoci",
            seed=0,
            env_steps=1000,
            episodes=10,
            eval_return_mean=1.5,
            eval_return_std=0.2,
            success_rate=0.5,
            mean_steps_to_success=42.0,
            epsilon=0.7,
        )
        m.log(
            condition="reprel_dfoci",
            seed=0,
            env_steps=2000,
            episodes=20,
            eval_return_mean=2.5,
            eval_return_std=0.1,
            success_rate=1.0,
            mean_steps_to_success=30.0,
            epsilon=0.6,
        )
    lines = path.read_text().splitlines()
    assert lines[0].split(",")[:3] == ["condition", "seed", "env_steps"]
    assert len(lines) == 3 and lines[2].startswith("reprel_dfoci,0,2000")


def test_run_metadata_records_config_git_and_seed(tmp_path: Path) -> None:
    write_run_metadata(tmp_path, config={"domain": {"name": "taxi"}}, seed=7)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["seed"] == 7 and "git_hash" in meta and "python" in meta and "numpy" in meta
    assert (tmp_path / "config.yaml").read_text().startswith("domain:")
    assert Atom.parse("at(taxi,l_0_0)")  # sanity: atoms import unaffected


def test_logger_streams_row_groups_and_tolerates_double_close(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    dom = TaxiDomain(TaxiConfig(num_passengers=1))
    s = dom.reset(np.random.default_rng(0))
    path = tmp_path / "t.parquet"
    logger = TransitionLogger(path, run_id="r", seed=0, chunk_size=2)
    with logger:
        for t in range(5):
            logger.log(TransitionRecord(0, t, "pickup", ("p1",), s, "north", -0.1, False, s, False))
        logger.close()
    assert pq.ParquetFile(path).metadata.num_row_groups >= 2
    assert len(read_transitions(path)) == 5


def test_object_names_with_colons_round_trip(tmp_path: Path) -> None:
    from reprel.core.atoms import Obj
    from reprel.core.state import State

    s = State(
        frozenset({Atom.parse("at(a:1,l)")}), frozenset({Obj("a:1", "thing"), Obj("l", "loc")})
    )
    path = tmp_path / "t.jsonl"
    with TransitionLogger(path, run_id="r", seed=0) as logger:
        logger.log(TransitionRecord(0, 0, "op", (), s, "x", 0.0, False, s, False))
    assert read_transitions(path)[0].state == s
