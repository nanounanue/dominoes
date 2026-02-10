"""Tile representation and domino set generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Tile:
    """A domino tile as canonical unordered pair (a <= b)."""

    a: int
    b: int

    def __post_init__(self) -> None:
        if not (0 <= self.a <= self.b <= 6):
            raise ValueError(f"Invalid tile: ({self.a}, {self.b})")


def generate_full_set() -> frozenset[Tile]:
    """Generate the complete double-six domino set (28 tiles)."""
    return frozenset(Tile(a, b) for a in range(7) for b in range(a, 7))
