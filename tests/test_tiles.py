"""Tests for tile representation."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from domino_oracle.core.tiles import Tile, generate_full_set, suits

# ---------------------------------------------------------------------------
# Tile construction
# ---------------------------------------------------------------------------


class TestTileConstruction:
    """Tests for Tile creation and validation."""

    def test_valid_tile(self) -> None:
        t = Tile(0, 6)
        assert t.a == 0
        assert t.b == 6

    def test_double_tile(self) -> None:
        t = Tile(3, 3)
        assert t.a == 3
        assert t.b == 3

    def test_boundary_tile_zero_zero(self) -> None:
        t = Tile(0, 0)
        assert t.a == 0 and t.b == 0

    def test_boundary_tile_six_six(self) -> None:
        t = Tile(6, 6)
        assert t.a == 6 and t.b == 6

    def test_invalid_tile_reversed(self) -> None:
        with pytest.raises(ValueError, match="Invalid tile"):
            Tile(5, 2)

    def test_invalid_tile_negative(self) -> None:
        with pytest.raises(ValueError, match="Invalid tile"):
            Tile(-1, 3)

    def test_invalid_tile_too_large(self) -> None:
        with pytest.raises(ValueError, match="Invalid tile"):
            Tile(0, 7)

    def test_invalid_tile_both_negative(self) -> None:
        with pytest.raises(ValueError, match="Invalid tile"):
            Tile(-2, -1)

    def test_tile_is_frozen(self) -> None:
        t = Tile(1, 4)
        with pytest.raises(AttributeError):
            t.a = 2  # type: ignore[misc]

    @given(a=st.integers(0, 6), b=st.integers(0, 6))
    def test_hypothesis_tile_creation(self, a: int, b: int) -> None:
        """Any pair with a <= b produces a valid tile; reversed pairs raise."""
        if a <= b:
            t = Tile(a, b)
            assert t.a == a
            assert t.b == b
            assert t.a <= t.b
        else:
            with pytest.raises(ValueError):
                Tile(a, b)

    @given(a=st.integers(-100, -1), b=st.integers(0, 6))
    def test_hypothesis_negative_a_invalid(self, a: int, b: int) -> None:
        with pytest.raises(ValueError):
            Tile(a, b)

    @given(a=st.integers(0, 6), b=st.integers(7, 100))
    def test_hypothesis_large_b_invalid(self, a: int, b: int) -> None:
        with pytest.raises(ValueError):
            Tile(a, b)


# ---------------------------------------------------------------------------
# Tile methods
# ---------------------------------------------------------------------------


class TestTileMethods:
    """Tests for Tile utility methods."""

    def test_is_double_true(self) -> None:
        assert Tile(3, 3).is_double() is True

    def test_is_double_false(self) -> None:
        assert Tile(2, 5).is_double() is False

    def test_values(self) -> None:
        assert Tile(1, 4).values() == (1, 4)

    def test_values_double(self) -> None:
        assert Tile(0, 0).values() == (0, 0)

    def test_contains_value_a(self) -> None:
        assert Tile(2, 5).contains_value(2) is True

    def test_contains_value_b(self) -> None:
        assert Tile(2, 5).contains_value(5) is True

    def test_contains_value_missing(self) -> None:
        assert Tile(2, 5).contains_value(3) is False

    def test_contains_value_double(self) -> None:
        assert Tile(4, 4).contains_value(4) is True
        assert Tile(4, 4).contains_value(3) is False

    def test_other_value_from_a(self) -> None:
        assert Tile(1, 6).other_value(1) == 6

    def test_other_value_from_b(self) -> None:
        assert Tile(1, 6).other_value(6) == 1

    def test_other_value_double(self) -> None:
        assert Tile(5, 5).other_value(5) == 5

    def test_other_value_not_present(self) -> None:
        with pytest.raises(ValueError, match="not in tile"):
            Tile(1, 6).other_value(3)

    def test_pip_count(self) -> None:
        assert Tile(2, 5).pip_count() == 7

    def test_pip_count_double(self) -> None:
        assert Tile(6, 6).pip_count() == 12

    def test_pip_count_blank(self) -> None:
        assert Tile(0, 0).pip_count() == 0

    @given(a=st.integers(0, 6), b=st.integers(0, 6))
    def test_hypothesis_pip_count(self, a: int, b: int) -> None:
        lo, hi = min(a, b), max(a, b)
        t = Tile(lo, hi)
        assert t.pip_count() == lo + hi


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------


class TestTileStringRepresentation:
    """Tests for __str__ and __repr__."""

    def test_str_format(self) -> None:
        assert str(Tile(2, 5)) == "[2|5]"

    def test_str_double(self) -> None:
        assert str(Tile(0, 0)) == "[0|0]"

    def test_repr_format(self) -> None:
        assert repr(Tile(2, 5)) == "Tile(2, 5)"

    def test_repr_double(self) -> None:
        assert repr(Tile(6, 6)) == "Tile(6, 6)"


# ---------------------------------------------------------------------------
# Tile ordering and equality
# ---------------------------------------------------------------------------


class TestTileOrdering:
    """Tests for frozen dataclass ordering."""

    def test_equality(self) -> None:
        assert Tile(1, 3) == Tile(1, 3)

    def test_inequality(self) -> None:
        assert Tile(1, 3) != Tile(1, 4)

    def test_ordering(self) -> None:
        assert Tile(0, 1) < Tile(0, 2) < Tile(1, 1)

    def test_hash_equal_tiles(self) -> None:
        assert hash(Tile(2, 4)) == hash(Tile(2, 4))

    def test_usable_in_set(self) -> None:
        s = {Tile(1, 2), Tile(1, 2), Tile(3, 4)}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# generate_full_set
# ---------------------------------------------------------------------------


class TestGenerateFullSet:
    """Tests for the full domino set generation."""

    def test_full_set_has_28_tiles(self) -> None:
        assert len(generate_full_set()) == 28

    def test_all_tiles_canonical(self) -> None:
        for tile in generate_full_set():
            assert tile.a <= tile.b

    def test_all_values_in_range(self) -> None:
        for tile in generate_full_set():
            assert 0 <= tile.a <= 6
            assert 0 <= tile.b <= 6

    def test_contains_all_doubles(self) -> None:
        full = generate_full_set()
        for v in range(7):
            assert Tile(v, v) in full

    def test_seven_doubles(self) -> None:
        doubles = [t for t in generate_full_set() if t.is_double()]
        assert len(doubles) == 7

    def test_total_pip_count(self) -> None:
        """Each value 0-6 appears in exactly 8 tiles (7 in its suit,
        but each tile has two ends). Total pips = 2 * sum(k*8 for k in 0..6)?
        Actually: sum of all pip counts = sum(a+b) for all tiles.
        """
        total = sum(t.pip_count() for t in generate_full_set())
        # Each value v appears in 7 tiles in its suit, but doubles count
        # the value twice. Value v appears as an end in exactly 8 end-slots
        # (7 tiles * 1 end + 1 double where it appears on the other end too).
        # Wait, simpler: sum(a+b) for all tiles.
        # Value v appears as 'a' in tiles (v, v), (v, v+1), ..., (v, 6) = 7-v tiles
        # Value v appears as 'b' in tiles (0, v), (1, v), ..., (v, v) = v+1 tiles
        # Total appearances of v = (7-v) + (v+1) = 8
        # Total pips = sum(v * 8 for v in 0..6) = 8 * 21 = 168
        assert total == 168

    def test_set_is_frozenset(self) -> None:
        full = generate_full_set()
        assert isinstance(full, frozenset)


# ---------------------------------------------------------------------------
# suits
# ---------------------------------------------------------------------------


class TestSuits:
    """Tests for the suits function."""

    @pytest.mark.parametrize("value", range(7))
    def test_suit_has_seven_tiles(self, value: int) -> None:
        assert len(suits(value)) == 7

    @pytest.mark.parametrize("value", range(7))
    def test_all_tiles_in_suit_contain_value(self, value: int) -> None:
        for tile in suits(value):
            assert tile.contains_value(value)

    def test_suit_zero_includes_blank_double(self) -> None:
        assert Tile(0, 0) in suits(0)

    def test_suit_six_includes_six_six(self) -> None:
        assert Tile(6, 6) in suits(6)

    def test_suits_are_frozensets(self) -> None:
        assert isinstance(suits(3), frozenset)

    def test_suit_invalid_value_negative(self) -> None:
        with pytest.raises(ValueError, match="Invalid suit value"):
            suits(-1)

    def test_suit_invalid_value_too_large(self) -> None:
        with pytest.raises(ValueError, match="Invalid suit value"):
            suits(7)

    def test_suit_overlap(self) -> None:
        """Tile [2|5] should be in both suit(2) and suit(5)."""
        t = Tile(2, 5)
        assert t in suits(2)
        assert t in suits(5)

    def test_double_in_single_suit(self) -> None:
        """The double [3|3] is only in suit(3), not in any other."""
        t = Tile(3, 3)
        assert t in suits(3)
        for v in range(7):
            if v != 3:
                assert t not in suits(v)

    def test_union_of_all_suits_equals_full_set(self) -> None:
        """Every tile belongs to at least one suit."""
        union = frozenset().union(*(suits(v) for v in range(7)))
        assert union == generate_full_set()
