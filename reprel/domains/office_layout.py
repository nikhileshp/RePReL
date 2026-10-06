"""Office World map (Illanes et al. 2020, as used in RePReL-domains): 12 x 9 cells, thin walls."""

from __future__ import annotations

Cell = tuple[int, int]  # (x, y), y grows upwards as in the original environment

WIDTH, HEIGHT = 12, 9
DIRECTIONS: dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

# Thin walls between horizontally adjacent cells (x, x+1) at the listed rows ...
_VERTICAL_WALLS = {2: (0, 2, 3, 4, 5, 6, 8), 5: (0, 2, 3, 4, 5, 6, 8), 8: (0, 2, 3, 4, 5, 6, 8)}
# ... and between vertically adjacent cells (y, y+1) at the listed columns.
_HORIZONTAL_WALLS = {2: (0, 2, 3, 4, 5, 6, 7, 8, 9, 11), 5: (0, 2, 3, 5, 6, 8, 9, 11)}

ITEMS: dict[str, tuple[Cell, ...]] = {
    "a": ((1, 1),),
    "b": ((10, 1),),
    "c": ((10, 7),),
    "d": ((1, 7),),
    "mail": ((7, 4),),
    "coffee": ((3, 6), (8, 2)),
    "office": ((4, 4),),
}
PLANTS: tuple[Cell, ...] = ((4, 1), (7, 1), (4, 7), (7, 7), (1, 4), (10, 4))


def blocked_pairs() -> frozenset[tuple[Cell, Cell]]:
    pairs: set[tuple[Cell, Cell]] = set()
    for x, rows in _VERTICAL_WALLS.items():
        for y in rows:
            pairs.add(((x, y), (x + 1, y)))
            pairs.add(((x + 1, y), (x, y)))
    for y, cols in _HORIZONTAL_WALLS.items():
        for x in cols:
            pairs.add(((x, y), (x, y + 1)))
            pairs.add(((x, y + 1), (x, y)))
    return frozenset(pairs)


def is_blocked(cell: Cell, direction: str, pairs: frozenset[tuple[Cell, Cell]]) -> bool:
    dx, dy = DIRECTIONS[direction]
    nxt = (cell[0] + dx, cell[1] + dy)
    if not (0 <= nxt[0] < WIDTH and 0 <= nxt[1] < HEIGHT):
        return True
    return (cell, nxt) in pairs
