"""ASCII grid layouts for the Taxi domain (``w`` = wall cell, letters = depots)."""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]

DIRECTIONS: dict[str, Cell] = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}


@dataclass(frozen=True)
class Grid:
    """Parsed layout. Coordinates are (row, col) over the interior, origin top-left."""

    height: int
    width: int
    free_cells: frozenset[Cell]
    depots: dict[str, Cell]

    def is_blocked(self, cell: Cell) -> bool:
        return cell not in self.free_cells

    def neighbour(self, cell: Cell, direction: str) -> Cell:
        dr, dc = DIRECTIONS[direction]
        return cell[0] + dr, cell[1] + dc


def parse_layout(text: str) -> Grid:
    """Parse a layout whose outer ring of ``w`` characters is the border."""
    rows = [line for line in text.splitlines() if line]
    interior = [row[1:-1] for row in rows[1:-1]]
    height, width = len(interior), len(interior[0])
    free: set[Cell] = set()
    depots: dict[str, Cell] = {}
    for r, row in enumerate(interior):
        if len(row) != width:
            raise ValueError("layout rows must have equal length")
        for c, ch in enumerate(row):
            if ch == "w":
                continue
            free.add((r, c))
            if ch != " ":
                depots[ch] = (r, c)
    return Grid(height, width, frozenset(free), depots)


EIGHT = """\
wwwwwwwwww
wR  w   Gw
w   w    w
w   w    w
w        w
w        w
w w  w   w
w w  w   w
wYw  wB  w
wwwwwwwwww
"""

FIVE = """\
wwwwwww
wR w Gw
w  w  w
w     w
w ww  w
wYwwB w
wwwwwww
"""

LAYOUTS: dict[str, Grid] = {"eight": parse_layout(EIGHT), "five": parse_layout(FIVE)}
