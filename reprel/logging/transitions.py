"""Per-operator transition logs with full ground states (input for D-FOCI structure learning).

Each row stores ``(s_t, a_t, r_t, done_t, s_{t+1}, operator, args)`` plus run/seed/episode
bookkeeping. States are serialised as sorted lists of atom strings and the typed object set
as ``name:type`` strings, so a log round-trips to identical :class:`State` objects.
Parquet is the default; a ``.jsonl`` path selects the JSON-lines fallback.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from reprel.core.atoms import Atom, Obj
from reprel.core.state import State

COLUMNS = (
    "run_id",
    "seed",
    "episode",
    "t",
    "operator",
    "args",
    "state",
    "objects",
    "action",
    "reward",
    "done",
    "next_state",
    "terminated",
)


@dataclass(frozen=True)
class TransitionRecord:
    episode: int
    t: int
    operator: str
    args: tuple[str, ...]
    state: State
    action: str
    reward: float
    done: bool
    next_state: State
    terminated: bool
    run_id: str = ""
    seed: int = -1

    def effects(self) -> tuple[frozenset[Atom], frozenset[Atom]]:
        """Atoms (added, removed) between ``state`` and ``next_state``."""
        return self.state.diff(self.next_state)

    def to_row(self, run_id: str, seed: int) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "seed": seed,
            "episode": self.episode,
            "t": self.t,
            "operator": self.operator,
            "args": list(self.args),
            "state": self.state.to_strings(),
            "objects": sorted(f"{o.name}:{o.type}" for o in self.state.objects),
            "action": self.action,
            "reward": float(self.reward),
            "done": bool(self.done),
            "next_state": self.next_state.to_strings(),
            "terminated": bool(self.terminated),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TransitionRecord:
        objects = frozenset(Obj(*item.rsplit(":", 1)) for item in row["objects"])
        return cls(
            episode=int(row["episode"]),
            t=int(row["t"]),
            operator=str(row["operator"]),
            args=tuple(row["args"]),
            state=State.from_strings(row["state"], objects),
            action=str(row["action"]),
            reward=float(row["reward"]),
            done=bool(row["done"]),
            next_state=State.from_strings(row["next_state"], objects),
            terminated=bool(row["terminated"]),
            run_id=str(row["run_id"]),
            seed=int(row["seed"]),
        )


class TransitionLogger:
    """Streams rows to disk: a Parquet row group (or JSONL flush) every ``chunk_size`` rows."""

    def __init__(self, path: str | Path, run_id: str, seed: int, chunk_size: int = 50_000) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.seed = seed
        self.chunk_size = chunk_size
        self._rows: list[dict[str, Any]] = []
        self._writer: Any = None
        self._closed = False
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.suffix == ".jsonl":
            self._fh: Any = open(self.path, "w", encoding="utf-8")
        else:
            import pyarrow.parquet as pq

            self._fh = None
            self._writer = pq.ParquetWriter(self.path, _schema(), compression="zstd")

    def log(self, record: TransitionRecord) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")
        self._rows.append(record.to_row(self.run_id, self.seed))
        self.count += 1
        if len(self._rows) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        if self._fh is not None:
            for row in self._rows:
                self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        else:
            import pyarrow as pa

            self._writer.write_table(pa.Table.from_pylist(self._rows, schema=_schema()))
        self._rows = []

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        if self._fh is not None:
            self._fh.close()
        else:
            self._writer.close()
        self._closed = True

    def __enter__(self) -> TransitionLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _schema() -> Any:
    import pyarrow as pa

    strings = pa.list_(pa.string())
    return pa.schema(
        [
            ("run_id", pa.string()),
            ("seed", pa.int64()),
            ("episode", pa.int64()),
            ("t", pa.int64()),
            ("operator", pa.string()),
            ("args", strings),
            ("state", strings),
            ("objects", strings),
            ("action", pa.string()),
            ("reward", pa.float64()),
            ("done", pa.bool_()),
            ("next_state", strings),
            ("terminated", pa.bool_()),
        ]
    )


def iter_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return
    import pyarrow.parquet as pq

    yield from pq.read_table(path).to_pylist()


def read_transitions(path: str | Path, operator: str | None = None) -> list[TransitionRecord]:
    """Read a log back into records, optionally keeping only one operator."""
    return [
        TransitionRecord.from_row(row)
        for row in iter_rows(path)
        if operator is None or row["operator"] == operator
    ]
