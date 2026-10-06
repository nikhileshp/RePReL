import numpy as np
import pytest

from reprel.agents import AgentPool, ConstantSchedule, LinearSchedule, QLearningAgent, RandomAgent


# ------------------------------------------------------------------ schedules
def test_linear_schedule_decays_per_episode_and_clamps() -> None:
    sched = LinearSchedule(start=0.75, end=0.01, episodes=100)
    assert sched.value(0) == pytest.approx(0.75)
    assert sched.value(50) == pytest.approx(0.38)
    assert sched.value(100) == pytest.approx(0.01)
    assert sched.value(10_000) == pytest.approx(0.01)


def test_constant_schedule() -> None:
    assert ConstantSchedule(0.1).value(0) == 0.1 and ConstantSchedule(0.1).value(99) == 0.1


# ------------------------------------------------------------------ Q-learning
class Chain:
    """Deterministic 5-state chain: action 1 moves right, action 0 moves left; +1 at the end."""

    n = 5

    def step(self, s: int, a: int) -> tuple[int, float, bool]:
        nxt = min(s + 1, self.n - 1) if a == 1 else max(s - 1, 0)
        done = nxt == self.n - 1
        return nxt, (1.0 if done else -0.01), done


def test_q_learning_converges_on_chain() -> None:
    env = Chain()
    agent = QLearningAgent(n_actions=2, alpha=0.5, gamma=0.9)
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = 0
        for _ in range(50):
            a = agent.act(s, rng, epsilon=0.2)
            nxt, r, done = env.step(s, a)
            agent.update(s, a, r, nxt, terminal=done)
            s = nxt
            if done:
                break
    for s in range(4):
        assert agent.act(s, rng, epsilon=0.0) == 1
    # V(3) = 1.0 exactly; V(0) = 0.9^3 - small step costs
    assert agent.value(3) == pytest.approx(1.0, abs=1e-6)
    assert agent.value(0) == pytest.approx(0.9**3 * 1.0 - 0.01 * (1 + 0.9 + 0.81), abs=1e-3)


def test_terminal_update_does_not_bootstrap() -> None:
    agent = QLearningAgent(n_actions=2, alpha=1.0, gamma=0.9)
    agent.update("goal", 0, 0.0, "goal", terminal=False)  # creates key with zeros
    agent.q["goal"][:] = 100.0
    agent.update("s", 1, 1.0, "goal", terminal=True)
    assert agent.q["s"][1] == pytest.approx(1.0)
    agent.update("s2", 1, 1.0, "goal", terminal=False)
    assert agent.q["s2"][1] == pytest.approx(1.0 + 0.9 * 100.0)


def test_unseen_key_breaks_ties_uniformly_and_epsilon_explores() -> None:
    agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.9)
    rng = np.random.default_rng(1)
    counts = np.bincount([agent.act("new", rng, epsilon=0.0) for _ in range(400)], minlength=4)
    assert min(counts.tolist()) > 50  # all four actions chosen when values tie
    agent.q["seen"] = np.array([0.0, 5.0, 0.0, 0.0])
    assert all(agent.act("seen", rng, epsilon=0.0) == 1 for _ in range(20))
    explored = [agent.act("seen", rng, epsilon=1.0) for _ in range(200)]
    assert len(set(explored)) == 4


def test_state_dict_round_trip() -> None:
    agent = QLearningAgent(n_actions=2, alpha=0.1, gamma=0.9)
    agent.update("a", 1, 1.0, "b", terminal=True)
    clone = QLearningAgent(n_actions=2, alpha=0.1, gamma=0.9)
    clone.load_state_dict(agent.state_dict())
    assert clone.q.keys() == agent.q.keys()
    assert np.array_equal(clone.q["a"], agent.q["a"])
    assert clone.n_keys == 1  # terminal updates do not create a row for next_key


def test_random_agent_ignores_values() -> None:
    agent = RandomAgent(n_actions=3)
    rng = np.random.default_rng(0)
    agent.update("a", 0, 1.0, "b", terminal=False)
    assert {agent.act("a", rng, epsilon=0.0) for _ in range(100)} == {0, 1, 2}


# ------------------------------------------------------------------ pool
def test_pool_creates_one_agent_per_operator_name_and_round_trips(tmp_path: object) -> None:
    pool = AgentPool(lambda: QLearningAgent(n_actions=6, alpha=0.1, gamma=0.9))
    assert pool.get("pickup") is pool.get("pickup")
    assert pool.get("pickup") is not pool.get("drop")
    pool.get("pickup").update("k", 2, 1.0, "k2", terminal=True)
    path = f"{tmp_path}/agents.pkl"
    pool.save(path)
    other = AgentPool(lambda: QLearningAgent(n_actions=6, alpha=0.1, gamma=0.9))
    other.load(path)
    assert set(other.names) == {"pickup", "drop"}
    restored, original = other.get("pickup"), pool.get("pickup")
    assert isinstance(restored, QLearningAgent) and isinstance(original, QLearningAgent)
    assert np.array_equal(restored.q["k"], original.q["k"])
    assert other.total_keys == pool.total_keys == 1


def test_update_does_not_create_a_row_for_the_next_key() -> None:
    agent = QLearningAgent(n_actions=2, alpha=0.5, gamma=0.9)
    agent.update("s", 0, 1.0, "unseen", terminal=False)
    assert set(agent.q) == {"s"}
    assert agent.q["s"][0] == pytest.approx(0.5)
