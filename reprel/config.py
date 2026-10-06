"""Experiment configuration: frozen dataclasses loaded from YAML with dotted overrides."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

CONDITIONS = ("flat", "reprel_none", "reprel_dfoci")
TOP_KEYS = {
    "name",
    "condition",
    "domain",
    "abstraction",
    "agent",
    "epsilon",
    "executor",
    "training",
    "logging",
    "seeds",
    "transfer",
    "output_dir",
    "explore",
}


def _check(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where}")


@dataclass(frozen=True)
class DomainConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DomainConfig:
        d = dict(d)
        return cls(name=str(d.pop("name")), params=d)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}


@dataclass(frozen=True)
class AbstractionConfig:
    name: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AbstractionConfig:
        d = dict(d)
        return cls(name=str(d.pop("name", "none")), params=d)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.params}


@dataclass(frozen=True)
class AgentConfig:
    alpha: float = 0.01
    gamma: float = 0.99
    terminal_reward: float = 1.0


@dataclass(frozen=True)
class EpsilonConfig:
    schedule: str = "linear"
    start: float = 0.75
    end: float = 0.01
    episodes: int = 20_000
    epsilon: float = 0.1  # for the constant schedule


@dataclass(frozen=True)
class ExecutorSettings:
    max_operator_steps: int | None = None
    replan_on_timeout: bool = True
    max_replans: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 1_000_000
    eval_every: int = 10_000
    eval_episodes: int = 100
    eval_epsilon: float = 0.01


@dataclass(frozen=True)
class LoggingConfig:
    transitions: bool = False


@dataclass(frozen=True)
class ExploreConfig:
    episodes: int = 100
    max_operator_steps: int = 50


@dataclass(frozen=True)
class TransferStage:
    domain_params: dict[str, Any]
    total_steps: int

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], default_steps: int) -> TransferStage:
        d = dict(d)
        steps = int(d.pop("total_steps", default_steps))
        return cls(domain_params=d, total_steps=steps)

    def to_dict(self) -> dict[str, Any]:
        return {**self.domain_params, "total_steps": self.total_steps}


@dataclass(frozen=True)
class TransferConfig:
    stages: tuple[TransferStage, ...]


def _dc(cls: type[Any], d: Mapping[str, Any] | None, where: str) -> Any:
    d = dict(d or {})
    _check(d, {f.name for f in fields(cls)}, where)
    for f in fields(cls):
        if f.name in d and f.type in ("int", "int | None") and isinstance(d[f.name], float):
            if d[f.name] != int(d[f.name]):
                raise ValueError(f"{where}.{f.name} must be an integer, got {d[f.name]!r}")
            d[f.name] = int(d[f.name])
    return cls(**d)


@dataclass(frozen=True)
class RunConfig:
    name: str
    condition: str
    domain: DomainConfig
    abstraction: AbstractionConfig = field(default_factory=AbstractionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    executor: ExecutorSettings = field(default_factory=ExecutorSettings)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    seeds: tuple[int, ...] = (0,)
    transfer: TransferConfig | None = None
    output_dir: str = "outputs"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        _check(data, TOP_KEYS, "run config")
        condition = str(data.get("condition", "reprel_dfoci"))
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
        training = _dc(TrainingConfig, data.get("training"), "training")
        transfer = None
        if data.get("transfer"):
            _check(data["transfer"], {"stages"}, "transfer")
            transfer = TransferConfig(
                tuple(
                    TransferStage.from_dict(s, training.total_steps)
                    for s in data["transfer"]["stages"]
                )
            )
        return cls(
            name=str(data["name"]),
            condition=condition,
            domain=DomainConfig.from_dict(data["domain"]),
            abstraction=AbstractionConfig.from_dict(data.get("abstraction") or {}),
            agent=_dc(AgentConfig, data.get("agent"), "agent"),
            epsilon=_dc(EpsilonConfig, data.get("epsilon"), "epsilon"),
            executor=_dc(ExecutorSettings, data.get("executor"), "executor"),
            training=training,
            logging=_dc(LoggingConfig, data.get("logging"), "logging"),
            explore=_dc(ExploreConfig, data.get("explore"), "explore"),
            seeds=_seeds(data.get("seeds", [0])),
            transfer=transfer,
            output_dir=str(data.get("output_dir", "outputs")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "condition": self.condition,
            "domain": self.domain.to_dict(),
            "abstraction": self.abstraction.to_dict(),
            "agent": vars(self.agent),
            "epsilon": vars(self.epsilon),
            "executor": vars(self.executor),
            "training": vars(self.training),
            "logging": vars(self.logging),
            "explore": vars(self.explore),
            "seeds": list(self.seeds),
            "output_dir": self.output_dir,
        }
        if self.transfer is not None:
            out["transfer"] = {"stages": [s.to_dict() for s in self.transfer.stages]}
        return out


def _seeds(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(s) for s in value)


def parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``["a.b=1", "seeds=[0,1]"]`` -> nested dict with Python-literal values."""
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like key=value: {item!r}")
        key, raw = item.split("=", 1)
        try:
            value: Any = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = _merge(data, overrides)
    return RunConfig.from_dict(data)
