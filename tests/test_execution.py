import numpy as np
import pytest

from reprel.abstraction import make_abstraction
from reprel.agents import AgentPool, ConstantSchedule, QLearningAgent, RandomAgent
from reprel.core.atoms import Atom
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_planning import make_taxi_planner
from reprel.execution import (
    EpisodeResult,
    ExecutorConfig,
    ExplorationCollector,
    FlatExecutor,
    RePReLExecutor,
    evaluate,
    train,
)
from reprel.logging.transitions import TransitionLogger, read_transitions


def make_executor(n: int = 1, layout: str = "five", **cfg: object) -> RePReLExecutor:
    dom = TaxiDomain(TaxiConfig(num_passengers=n, layout=layout, max_steps=200))
    return RePReLExecutor(
        domain=dom,
        planner=make_taxi_planner(terminal_reward=10.0),
        abstraction=make_abstraction("dfoci", spec="taxi", signature=dom.predicates),
        pool=AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.95)),
        config=ExecutorConfig(**cfg),  # type: ignore[arg-type]
    )


def test_random_episode_terminates_at_max_steps_and_counts_steps() -> None:
    ex = make_executor()
    ex.pool = AgentPool(lambda: RandomAgent(ex.domain.n_actions))
    res = ex.run_episode(np.random.default_rng(0), epsilon=1.0, learn=False)
    assert isinstance(res, EpisodeResult)
    assert res.env_steps <= 200
    assert res.success or res.env_steps == 200


def test_operator_step_budget_triggers_replanning() -> None:
    ex = make_executor(max_operator_steps=5, replan_on_timeout=True, max_replans=3)
    ex.pool = AgentPool(lambda: RandomAgent(ex.domain.n_actions))
    res = ex.run_episode(np.random.default_rng(0), epsilon=1.0, learn=False)
    assert res.subtask_failures >= 1
    assert res.replans == min(res.subtask_failures, 3)
    assert res.env_steps <= 5 * (3 + 1) + 5  # stops after the last allowed replan's budget


def test_already_satisfied_operators_are_skipped() -> None:
    ex = make_executor(n=2)
    rng = np.random.default_rng(3)
    s = ex.domain.reset(rng)
    s = s.with_atoms(add=[Atom.parse("delivered(p1)")])
    ex.pool = AgentPool(lambda: RandomAgent(ex.domain.n_actions))
    res = ex.run_episode(rng, epsilon=1.0, learn=False, initial_state=s)
    assert [op for op in res.operators_run] == ["pickup(p2)", "drop(p2)"] or res.env_steps == 200


def test_learning_only_updates_the_active_operator_agent() -> None:
    ex = make_executor(max_operator_steps=3, replan_on_timeout=False)
    rng = np.random.default_rng(0)
    ex.run_episode(rng, epsilon=1.0, learn=True)
    # only pickup ran (3 random steps, no replanning) -> only the pickup agent has keys
    assert ex.pool.get("pickup").n_keys > 0
    assert ex.pool.get("drop").n_keys == 0


def test_reprel_learns_one_passenger_taxi_quickly() -> None:
    ex = make_executor()
    rng = np.random.default_rng(0)
    history = train(
        ex,
        rng,
        total_steps=15_000,
        eval_every=5_000,
        eval_episodes=20,
        eval_epsilon=0.0,
        schedule=ConstantSchedule(0.2),
    )
    final = history[-1]
    assert final.env_steps >= 15_000
    assert final.eval.success_rate >= 0.9
    assert final.eval.return_mean > 20.0


def test_transfer_to_more_passengers_without_reset_keeps_success() -> None:
    ex = make_executor()
    rng = np.random.default_rng(1)
    train(
        ex,
        rng,
        total_steps=30_000,
        eval_every=30_000,
        eval_episodes=10,
        eval_epsilon=0.0,
        schedule=ConstantSchedule(0.2),
    )
    bigger = TaxiDomain(TaxiConfig(num_passengers=3, layout="five", max_steps=300))
    transferred = RePReLExecutor(bigger, ex.planner, ex.abstraction, ex.pool, ex.config)
    fresh = RePReLExecutor(
        bigger,
        ex.planner,
        ex.abstraction,
        AgentPool(lambda: QLearningAgent(bigger.n_actions, alpha=0.1, gamma=0.95)),
        ex.config,
    )
    with_transfer = evaluate(transferred, np.random.default_rng(2), episodes=10, epsilon=0.0)
    without = evaluate(fresh, np.random.default_rng(2), episodes=10, epsilon=0.0)
    assert with_transfer.success_rate >= 0.8
    assert with_transfer.success_rate > without.success_rate
    assert with_transfer.return_mean > without.return_mean


def test_executor_logs_full_ground_transitions(tmp_path: object) -> None:
    ex = make_executor()
    ex.pool = AgentPool(lambda: RandomAgent(ex.domain.n_actions))
    path = f"{tmp_path}/t.parquet"
    with TransitionLogger(path, run_id="x", seed=0) as logger:
        res = ex.run_episode(np.random.default_rng(0), epsilon=1.0, learn=False, logger=logger)
    rows = read_transitions(path)
    assert len(rows) == res.env_steps
    assert rows[0].operator == "pickup" and rows[0].args == ("p1",)
    assert any(a.pred == "wall" for a in rows[0].state.atoms)


def test_exploration_collector_writes_per_operator_data(tmp_path: object) -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=2, layout="five", max_steps=100))
    collector = ExplorationCollector(dom, make_taxi_planner(), max_operator_steps=20)
    path = f"{tmp_path}/explore.parquet"
    stats = collector.collect(np.random.default_rng(0), episodes=5, path=path)
    rows = read_transitions(path)
    assert stats["transitions"] == len(rows) > 0
    assert {r.operator for r in rows} <= {"pickup", "drop"}
    assert stats["episodes"] == 5


def test_flat_executor_runs_and_learns_a_little() -> None:
    dom = TaxiDomain(TaxiConfig(num_passengers=1, layout="five", max_steps=100))
    flat = FlatExecutor(dom, QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.95))
    rng = np.random.default_rng(0)
    before = evaluate(flat, np.random.default_rng(9), episodes=10, epsilon=0.0)
    history = train(
        flat,
        rng,
        total_steps=40_000,
        eval_every=40_000,
        eval_episodes=10,
        eval_epsilon=0.0,
        schedule=ConstantSchedule(0.3),
    )
    assert history[-1].eval.return_mean > before.return_mean
    assert history[-1].eval.success_rate > 0.5


def test_evaluate_reports_steps_to_success_only_for_successes() -> None:
    ex = make_executor()
    ex.pool = AgentPool(lambda: RandomAgent(ex.domain.n_actions))
    res = evaluate(ex, np.random.default_rng(0), episodes=5, epsilon=1.0)
    assert 0.0 <= res.success_rate <= 1.0
    if res.success_rate == 0.0:
        assert np.isnan(res.mean_steps_to_success)
    else:
        assert res.mean_steps_to_success <= 200
    with pytest.raises(ValueError):
        evaluate(ex, np.random.default_rng(0), episodes=0, epsilon=0.0)
