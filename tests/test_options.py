import numpy as np
import pytest

from reprel.agents import AgentPool, ConstantSchedule, QLearningAgent
from reprel.core.atoms import Atom
from reprel.domains.taxi import TaxiConfig, TaxiDomain
from reprel.domains.taxi_planning import make_taxi_planner, taxi_operator_target, taxi_reach_options
from reprel.execution import HRLExecutor, TRLExecutor, evaluate, train
from reprel.execution.options import ReachOption


def setup(n: int = 1) -> tuple[TaxiDomain, tuple[ReachOption, ...]]:
    dom = TaxiDomain(TaxiConfig(num_passengers=n, layout="five", max_steps=200))
    return dom, taxi_reach_options(dom, terminal_reward=10.0)


def test_taxi_has_one_reach_option_per_depot() -> None:
    dom, options = setup()
    assert {o.name for o in options} == {
        "reach(l_0_0)",
        "reach(l_0_4)",
        "reach(l_4_0)",
        "reach(l_4_3)",
    }
    s = dom.reset(np.random.default_rng(0))
    s = s.with_atoms(
        add=[Atom.parse("at(taxi,l_0_0)")],
        remove=[a for a in s.atoms_with("at") if a.args[0] == "taxi"],
    )
    o = next(o for o in options if o.name == "reach(l_0_0)")
    assert o.is_terminated(s)
    assert o.reward(-0.1, s) == pytest.approx(9.9)
    other = next(o for o in options if o.name == "reach(l_4_3)")
    assert not other.is_terminated(s) and other.reward(-0.1, s) == pytest.approx(-0.1)


def test_operator_target_is_pickup_depot_or_destination() -> None:
    dom, _ = setup(2)
    s = dom.reset(np.random.default_rng(1))
    plan = make_taxi_planner().plan(s, dom.goal(s))
    at = {a.args[0]: a.args[1] for a in s.atoms_with("at")}
    dest = {a.args[0]: a.args[1] for a in s.atoms_with("dest")}
    assert taxi_operator_target(plan[0], s) == at["p1"]
    assert taxi_operator_target(plan[1], s) == dest["p1"]
    assert taxi_operator_target(plan[2], s) == at["p2"]


def test_trl_updates_every_option_each_step() -> None:
    dom, options = setup()
    pool = AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.5, gamma=0.9))
    ex = TRLExecutor(dom, make_taxi_planner(10.0), options, taxi_operator_target, pool)
    res = ex.run_episode(np.random.default_rng(0), epsilon=1.0, learn=True)
    assert res.env_steps > 0
    assert set(pool.names) == {o.name for o in options}
    assert all(pool.get(o.name).n_keys > 0 for o in options)


def test_trl_learns_one_passenger() -> None:
    dom, options = setup()
    pool = AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.95))
    ex = TRLExecutor(dom, make_taxi_planner(10.0), options, taxi_operator_target, pool)
    h = train(
        ex,
        np.random.default_rng(0),
        total_steps=40_000,
        eval_every=40_000,
        eval_episodes=20,
        eval_epsilon=0.0,
        schedule=ConstantSchedule(0.2),
    )
    assert h[-1].eval.success_rate >= 0.9


def test_hrl_meta_controller_chooses_options_and_learns_something() -> None:
    dom, options = setup()
    pool = AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.95))
    ex = HRLExecutor(dom, options, pool, meta_alpha=0.1, gamma=0.95)
    rng = np.random.default_rng(0)
    before = evaluate(ex, np.random.default_rng(5), episodes=10, epsilon=0.0)
    res = ex.run_episode(rng, epsilon=1.0, learn=True)
    assert res.env_steps > 0 and ex.meta.n_keys > 0
    assert ex.meta.n_actions == len(options) + 2  # reach options + primitive pickup and dropoff
    h = train(
        ex,
        rng,
        total_steps=60_000,
        eval_every=60_000,
        eval_episodes=20,
        eval_epsilon=0.0,
        schedule=ConstantSchedule(0.3),
    )
    assert h[-1].eval.return_mean > before.return_mean


def test_hrl_options_run_at_least_one_step_when_already_at_target() -> None:
    dom, options = setup()
    pool = AgentPool(lambda: QLearningAgent(dom.n_actions, alpha=0.1, gamma=0.95))
    ex = HRLExecutor(dom, options, pool, meta_alpha=0.1, gamma=0.95)
    res = ex.run_episode(np.random.default_rng(3), epsilon=1.0, learn=False)
    assert res.env_steps <= dom.max_steps
