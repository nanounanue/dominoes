"""Tile representation and domino set generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, order=True)
class Tile:
    """A domino tile as canonical unordered pair (a <= b).

    Tiles are immutable and ordered. The canonical form ensures ``a <= b``
    to avoid duplicate representations of the same physical tile.

    Attributes:
        a: The lower (or equal) pip value, 0-6.
        b: The higher (or equal) pip value, 0-6.

    Raises:
        ValueError: If the tile values are outside [0, 6] or not in
            canonical order (a <= b).
    """

    a: int
    b: int

    def __post_init__(self) -> None:
        if not (0 <= self.a <= self.b <= 6):
            raise ValueError(f"Invalid tile: ({self.a}, {self.b})")

    def is_double(self) -> bool:
        """Return True if the tile is a double (both ends equal).

        Returns:
            True when ``a == b``.
        """
        return self.a == self.b

    def values(self) -> tuple[int, int]:
        """Return the two pip values as a tuple.

        Returns:
            A tuple ``(a, b)`` with the tile's pip values.
        """
        return (self.a, self.b)

    def contains_value(self, v: int) -> bool:
        """Return True if the tile contains the given pip value.

        Args:
            v: The pip value to check for.

        Returns:
            True if ``a == v`` or ``b == v``.
        """
        return self.a == v or self.b == v

    def other_value(self, v: int) -> int:
        """Given one end value, return the other end.

        For doubles, returns the same value.

        Args:
            v: One of the tile's pip values.

        Returns:
            The other pip value on the tile.

        Raises:
            ValueError: If ``v`` is not one of the tile's values.
        """
        if v == self.a:
            return self.b
        if v == self.b:
            return self.a
        raise ValueError(f"Value {v} not in tile {self}")

    def pip_count(self) -> int:
        """Return the total pip count (sum of both ends).

        Returns:
            ``a + b``.
        """
        return self.a + self.b

    def __str__(self) -> str:
        """Return a human-readable string like ``[a|b]``."""
        return f"[{self.a}|{self.b}]"

    def __repr__(self) -> str:
        """Return a developer-readable representation like ``Tile(a, b)``."""
        return f"Tile({self.a}, {self.b})"


def generate_full_set() -> frozenset[Tile]:
    """Generate the complete double-six domino set (28 tiles).

    Returns:
        A frozenset containing all 28 tiles where ``0 <= a <= b <= 6``.
    """
    return frozenset(Tile(a, b) for a in range(7) for b in range(a, 7))


@lru_cache(maxsize=7)
def suits(value: int) -> frozenset[Tile]:
    """Return all tiles in the full set that contain the given pip value.

    A "suit" is the set of all tiles bearing a particular value. Each suit
    contains exactly 7 tiles in a double-six set.

    Args:
        value: The pip value to filter on (0-6).

    Returns:
        A frozenset of tiles containing ``value``.

    Raises:
        ValueError: If ``value`` is not in the range [0, 6].
    """
    if not (0 <= value <= 6):
        raise ValueError(f"Invalid suit value: {value}. Must be 0-6.")
    return frozenset(tile for tile in generate_full_set() if tile.contains_value(value))
