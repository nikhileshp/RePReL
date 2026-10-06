"""Shared builders for the scripts: domain, planner, abstraction, runner from a RunConfig."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import reprel.domains  # noqa: F401  (registers domains)
from reprel.abstraction import make_abstraction
from reprel.agents import (
    AgentPool,
    ConstantSchedule,
    EpsilonSchedule,
    LinearSchedule,
    QLearningAgent,
)
from reprel.config import RunConfig
from reprel.core.domain import Domain, make_domain
from reprel.execution import (
    EpisodeRunner,
    ExecutorConfig,
    FlatExecutor,
    HRLExecutor,
    RePReLExecutor,
    TRLExecutor,
)
from reprel.planning.planner import Planner


def build_domain(cfg: RunConfig, overrides: Mapping[str, Any] | None = None) -> Domain:
    params = {**cfg.domain.params, **(overrides or {})}
    return make_domain(cfg.domain.name, **params)


def build_planner(cfg: RunConfig) -> Planner:
    if cfg.domain.name == "taxi":
        from reprel.domains.taxi_planning import make_taxi_planner

        return make_taxi_planner(terminal_reward=cfg.agent.terminal_reward)
    if cfg.domain.name == "office":
        from reprel.domains.office_planning import make_office_planner

        return make_office_planner(terminal_reward=cfg.agent.terminal_reward)
    if cfg.domain.name == "boxworld":
        from reprel.domains.boxworld_planning import make_boxworld_planner

        return make_boxworld_planner(terminal_reward=cfg.agent.terminal_reward)
    raise ValueError(f"no planner registered for domain {cfg.domain.name!r}")


def build_schedule(cfg: RunConfig) -> EpsilonSchedule:
    e = cfg.epsilon
    if e.schedule == "linear":
        return LinearSchedule(e.start, e.end, e.episodes)
    if e.schedule == "constant":
        return ConstantSchedule(e.epsilon)
    raise ValueError(f"unknown epsilon schedule {e.schedule!r}")


def build_pool(cfg: RunConfig, domain: Domain) -> AgentPool:
    return AgentPool(lambda: QLearningAgent(domain.n_actions, cfg.agent.alpha, cfg.agent.gamma))


def build_runner(
    cfg: RunConfig, domain: Domain, pool: AgentPool | None = None, planner: Planner | None = None
) -> tuple[EpisodeRunner, AgentPool]:
    """Runner for the configured condition. ``pool`` is reused across transfer stages."""
    if cfg.condition == "flat":
        pool = pool or build_pool(cfg, domain)
        return FlatExecutor(domain, pool.get("flat")), pool
    pool = pool or build_pool(cfg, domain)
    if cfg.condition in ("trl", "hrl"):
        if cfg.domain.name != "taxi":
            raise ValueError(f"no option baselines for domain {cfg.domain.name!r}")
        from reprel.domains.taxi import TaxiDomain
        from reprel.domains.taxi_planning import taxi_operator_target, taxi_reach_options

        assert isinstance(domain, TaxiDomain)
        options = taxi_reach_options(domain, cfg.agent.terminal_reward)
        if cfg.condition == "trl":
            return TRLExecutor(
                domain, planner or build_planner(cfg), options, taxi_operator_target, pool
            ), pool
        return HRLExecutor(
            domain, options, pool, meta_alpha=cfg.agent.alpha, gamma=cfg.agent.gamma
        ), pool
    planner = planner or build_planner(cfg)
    if cfg.condition == "reprel_none":
        none_params = {k: v for k, v in cfg.abstraction.params.items() if k == "include_binding"}
        abstraction = make_abstraction("none", **none_params)
    else:
        params = {"spec": cfg.domain.name, **cfg.abstraction.params}
        abstraction = make_abstraction("dfoci", signature=domain.predicates, **params)
    executor = RePReLExecutor(
        domain, planner, abstraction, pool, ExecutorConfig(**vars(cfg.executor))
    )
    return executor, pool
