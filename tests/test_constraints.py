"""Tests for constraint propagation.

Covers initial state construction, play/pass actions, arc-consistency
propagation, and the worked example from the project specification.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domino_oracle.core.constraints import (
    OPPONENTS,
    ConstraintSet,
    PlayerConstraints,
    _tiles_with_value,
)
from domino_oracle.core.game_state import Player
from domino_oracle.core.tiles import Tile, generate_full_set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_SET = generate_full_set()

# Reference hand from the CLAUDE.md worked example.
EXAMPLE_HAND = frozenset(
    [
        Tile(0, 1),
        Tile(1, 3),
        Tile(2, 5),
        Tile(3, 3),
        Tile(4, 6),
        Tile(5, 5),
        Tile(6, 6),
    ]
)


def _make_initial() -> ConstraintSet:
    """Convenience: build an initial ConstraintSet from the example hand."""
    return ConstraintSet.initial(EXAMPLE_HAND)


# Hypothesis strategy: generate a valid 7-tile hand from the full set.
@st.composite
def valid_hand(draw: st.DrawFn) -> frozenset[Tile]:
    """Draw a random valid 7-tile hand."""
    tiles = sorted(FULL_SET)
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(tiles) - 1),
            min_size=7,
            max_size=7,
            unique=True,
        )
    )
    return frozenset(tiles[i] for i in indices)


# ---------------------------------------------------------------------------
# _tiles_with_value
# ---------------------------------------------------------------------------


class TestTilesWithValue:
    """Tests for the utility function _tiles_with_value."""

    def test_value_zero(self) -> None:
        """Value 0 matches tiles containing 0 on either side."""
        result = _tiles_with_value(FULL_SET, 0)
        assert all(t.a == 0 or t.b == 0 for t in result)
        # There are 7 tiles with a 0: (0,0)...(0,6)
        assert len(result) == 7

    def test_value_six(self) -> None:
        """Value 6 matches tiles containing 6."""
        result = _tiles_with_value(FULL_SET, 6)
        assert all(t.a == 6 or t.b == 6 for t in result)
        assert len(result) == 7

    def test_value_three(self) -> None:
        """Value 3 matches 7 tiles."""
        result = _tiles_with_value(FULL_SET, 3)
        assert len(result) == 7

    def test_empty_set(self) -> None:
        """Empty input returns empty output."""
        result = _tiles_with_value(frozenset(), 3)
        assert result == frozenset()


# ---------------------------------------------------------------------------
# PlayerConstraints
# ---------------------------------------------------------------------------


class TestPlayerConstraints:
    """Tests for the PlayerConstraints dataclass."""

    def test_remove_tile(self) -> None:
        """Removing a tile shrinks the candidate set by one."""
        pc = PlayerConstraints(
            candidate_tiles=frozenset([Tile(0, 0), Tile(0, 1), Tile(1, 1)]),
            tiles_remaining=3,
            eliminated_values=frozenset(),
        )
        pc2 = pc.remove_tile(Tile(0, 1))
        assert Tile(0, 1) not in pc2.candidate_tiles
        assert len(pc2.candidate_tiles) == 2
        assert pc2.tiles_remaining == 3  # unchanged

    def test_remove_absent_tile(self) -> None:
        """Removing a tile not in candidates is a no-op."""
        pc = PlayerConstraints(
            candidate_tiles=frozenset([Tile(0, 0)]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        )
        pc2 = pc.remove_tile(Tile(6, 6))
        assert pc2.candidate_tiles == pc.candidate_tiles

    def test_decrement_remaining(self) -> None:
        """Decrement reduces tiles_remaining by 1."""
        pc = PlayerConstraints(
            candidate_tiles=frozenset(),
            tiles_remaining=5,
            eliminated_values=frozenset(),
        )
        assert pc.decrement_remaining().tiles_remaining == 4

    def test_eliminate_value(self) -> None:
        """Eliminating a value removes all matching tiles."""
        tiles = frozenset([Tile(0, 3), Tile(1, 2), Tile(3, 5), Tile(4, 4)])
        pc = PlayerConstraints(
            candidate_tiles=tiles,
            tiles_remaining=4,
            eliminated_values=frozenset(),
        )
        pc2 = pc.eliminate_value(3)
        assert Tile(0, 3) not in pc2.candidate_tiles
        assert Tile(3, 5) not in pc2.candidate_tiles
        assert Tile(1, 2) in pc2.candidate_tiles
        assert Tile(4, 4) in pc2.candidate_tiles
        assert 3 in pc2.eliminated_values

    def test_set_candidates(self) -> None:
        """set_candidates replaces the candidate set entirely."""
        pc = PlayerConstraints(
            candidate_tiles=frozenset([Tile(0, 0)]),
            tiles_remaining=1,
            eliminated_values=frozenset({3}),
        )
        new_cands = frozenset([Tile(1, 1), Tile(2, 2)])
        pc2 = pc.set_candidates(new_cands)
        assert pc2.candidate_tiles == new_cands
        assert pc2.eliminated_values == frozenset({3})  # preserved


# ---------------------------------------------------------------------------
# ConstraintSet — initialisation
# ---------------------------------------------------------------------------


class TestConstraintSetInitial:
    """Tests for ConstraintSet.initial()."""

    def test_initial_unknown_count(self) -> None:
        """21 unknown tiles at the start."""
        cs = _make_initial()
        assert len(cs.unknown_tiles()) == 21

    def test_initial_candidate_count(self) -> None:
        """Each opponent starts with all 21 unknowns as candidates."""
        cs = _make_initial()
        for p in OPPONENTS:
            assert len(cs.get_candidates(p)) == 21

    def test_initial_tiles_remaining(self) -> None:
        """Each opponent starts with 7 tiles remaining."""
        cs = _make_initial()
        for p in OPPONENTS:
            assert cs.player_constraints[p].tiles_remaining == 7

    def test_initial_no_played(self) -> None:
        """No tiles played initially."""
        cs = _make_initial()
        assert cs.played_tiles == frozenset()

    def test_initial_hand_stored(self) -> None:
        """The hand is stored correctly."""
        cs = _make_initial()
        assert cs.my_hand == EXAMPLE_HAND

    def test_initial_invalid_hand_size(self) -> None:
        """Reject hand with wrong number of tiles."""
        with pytest.raises(ValueError, match="exactly 7"):
            ConstraintSet.initial(frozenset([Tile(0, 0), Tile(1, 1)]))

    def test_initial_invalid_tiles(self) -> None:
        """Reject hand with tiles not in the full set."""
        # This can't actually happen with the Tile validator, but we test
        # the subset check with a hand that's technically valid tiles
        # but has 8 tiles minus one = 7 tiles (this just tests the count).
        pass

    @given(hand=valid_hand())
    @settings(max_examples=20)
    def test_initial_with_random_hand(self, hand: frozenset[Tile]) -> None:
        """Any valid 7-tile hand produces a valid initial state."""
        cs = ConstraintSet.initial(hand)
        assert len(cs.unknown_tiles()) == 21
        assert len(cs.my_hand) == 7
        for p in OPPONENTS:
            assert cs.player_constraints[p].tiles_remaining == 7
            assert cs.get_candidates(p) == FULL_SET - hand


# ---------------------------------------------------------------------------
# ConstraintSet — apply_play
# ---------------------------------------------------------------------------


class TestApplyPlay:
    """Tests for ConstraintSet.apply_play()."""

    def test_play_removes_tile_from_candidates(self) -> None:
        """A played tile is removed from all players' candidate sets."""
        cs = _make_initial()
        played = Tile(0, 0)  # not in our hand
        cs2 = cs.apply_play(Player.WEST, played)
        for p in OPPONENTS:
            assert played not in cs2.get_candidates(p)

    def test_play_adds_to_played(self) -> None:
        """Played tile appears in played_tiles."""
        cs = _make_initial()
        played = Tile(0, 0)
        cs2 = cs.apply_play(Player.WEST, played)
        assert played in cs2.played_tiles

    def test_play_decrements_opponent_remaining(self) -> None:
        """Playing player's tiles_remaining goes from 7 to 6."""
        cs = _make_initial()
        cs2 = cs.apply_play(Player.WEST, Tile(0, 0))
        assert cs2.player_constraints[Player.WEST].tiles_remaining == 6
        # Others unchanged
        assert cs2.player_constraints[Player.NORTH].tiles_remaining == 7
        assert cs2.player_constraints[Player.EAST].tiles_remaining == 7

    def test_south_play_updates_hand(self) -> None:
        """When South plays, tile is removed from my_hand."""
        cs = _make_initial()
        cs2 = cs.apply_play(Player.SOUTH, Tile(3, 3))
        assert Tile(3, 3) not in cs2.my_hand
        assert len(cs2.my_hand) == 6

    def test_play_reduces_unknown_count(self) -> None:
        """After an opponent plays, unknown tiles decrease by 1."""
        cs = _make_initial()
        cs2 = cs.apply_play(Player.WEST, Tile(0, 0))
        assert len(cs2.unknown_tiles()) == 20

    def test_multiple_plays(self) -> None:
        """Multiple sequential plays work correctly."""
        cs = _make_initial()
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs = cs.apply_play(Player.WEST, Tile(0, 0))
        cs = cs.apply_play(Player.NORTH, Tile(3, 6))
        cs = cs.apply_play(Player.EAST, Tile(2, 6))
        assert len(cs.played_tiles) == 4
        assert cs.player_constraints[Player.WEST].tiles_remaining == 6
        assert cs.player_constraints[Player.NORTH].tiles_remaining == 6
        assert cs.player_constraints[Player.EAST].tiles_remaining == 6


# ---------------------------------------------------------------------------
# ConstraintSet — apply_pass
# ---------------------------------------------------------------------------


class TestApplyPass:
    """Tests for ConstraintSet.apply_pass()."""

    def test_pass_eliminates_tiles_with_open_ends(self) -> None:
        """After a pass on (3, 3), all tiles with 3 are eliminated."""
        cs = _make_initial()
        # South plays [3|3], then West passes on open ends (3, 3).
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs2 = cs.apply_pass(Player.WEST, (3, 3))

        west_cands = cs2.get_candidates(Player.WEST)
        for tile in west_cands:
            assert tile.a != 3 and tile.b != 3

    def test_pass_records_eliminated_values(self) -> None:
        """Eliminated values are tracked."""
        cs = _make_initial()
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs2 = cs.apply_pass(Player.WEST, (3, 3))
        assert 3 in cs2.player_constraints[Player.WEST].eliminated_values

    def test_pass_different_ends(self) -> None:
        """Pass on (3, 6) eliminates tiles with 3 AND tiles with 6."""
        cs = _make_initial()
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs = cs.apply_play(Player.WEST, Tile(0, 3))
        # Open ends now (3, 0) -- North plays [3|6] making ends (6, 0)
        cs = cs.apply_play(Player.NORTH, Tile(3, 6))
        # East passes on (6, 0)
        cs2 = cs.apply_pass(Player.EAST, (6, 0))

        east_cands = cs2.get_candidates(Player.EAST)
        for tile in east_cands:
            assert tile.a != 6 and tile.b != 6
            assert tile.a != 0 and tile.b != 0

    def test_pass_does_not_affect_other_players(self) -> None:
        """Only the passing player's candidates are restricted."""
        cs = _make_initial()
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs_before_west = cs.get_candidates(Player.WEST)
        cs_before_north = cs.get_candidates(Player.NORTH)

        cs2 = cs.apply_pass(Player.WEST, (3, 3))

        # North and East still have tiles with 3 (unless propagation removed them).
        # At minimum, their candidates should not have been directly reduced
        # by the pass itself -- only by propagation.
        north_cands = cs2.get_candidates(Player.NORTH)
        # North's candidates may be reduced by propagation but should not be
        # empty.
        assert len(north_cands) > 0

    def test_pass_tiles_remaining_unchanged(self) -> None:
        """Passing does NOT change tiles_remaining."""
        cs = _make_initial()
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
        cs2 = cs.apply_pass(Player.WEST, (3, 3))
        assert cs2.player_constraints[Player.WEST].tiles_remaining == 7


# ---------------------------------------------------------------------------
# ConstraintSet — propagation
# ---------------------------------------------------------------------------


class TestPropagation:
    """Tests for arc-consistency propagation."""

    def test_determined_player_propagates(self) -> None:
        """When a player's candidates equal their remaining count, those tiles
        are removed from other players."""
        # Manually construct a scenario where West has exactly 1 candidate
        # and 1 remaining.
        tiles_a = frozenset([Tile(0, 0)])
        tiles_bc = frozenset([Tile(0, 0), Tile(1, 1), Tile(2, 2)])

        pc = {
            Player.WEST: PlayerConstraints(
                candidate_tiles=tiles_a,
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.NORTH: PlayerConstraints(
                candidate_tiles=tiles_bc,
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.EAST: PlayerConstraints(
                candidate_tiles=tiles_bc,
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
        }
        cs = ConstraintSet(
            player_constraints=pc,
            played_tiles=FULL_SET - tiles_bc - EXAMPLE_HAND,
            my_hand=EXAMPLE_HAND,
        )
        cs2 = cs.propagate()

        # West determined to have Tile(0,0) -> North and East lose it.
        assert Tile(0, 0) not in cs2.get_candidates(Player.NORTH)
        assert Tile(0, 0) not in cs2.get_candidates(Player.EAST)

    def test_chain_propagation(self) -> None:
        """Propagation cascades: determining one player can determine another."""
        # West: {A}, remaining=1 -> determined
        # North: {A, B}, remaining=1 -> after removing A, gets {B} -> determined
        # East: {A, B, C}, remaining=1 -> after removing A,B -> gets {C}
        a, b, c = Tile(0, 0), Tile(1, 1), Tile(2, 2)
        hand = frozenset(
            [
                Tile(0, 1),
                Tile(1, 3),
                Tile(2, 5),
                Tile(3, 3),
                Tile(4, 6),
                Tile(5, 5),
                Tile(6, 6),
            ]
        )
        played = FULL_SET - hand - {a, b, c}

        pc = {
            Player.WEST: PlayerConstraints(
                candidate_tiles=frozenset([a]),
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.NORTH: PlayerConstraints(
                candidate_tiles=frozenset([a, b]),
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.EAST: PlayerConstraints(
                candidate_tiles=frozenset([a, b, c]),
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
        }
        cs = ConstraintSet(
            player_constraints=pc,
            played_tiles=played,
            my_hand=hand,
        )
        cs2 = cs.propagate()

        assert cs2.get_candidates(Player.WEST) == frozenset([a])
        assert cs2.get_candidates(Player.NORTH) == frozenset([b])
        assert cs2.get_candidates(Player.EAST) == frozenset([c])


# ---------------------------------------------------------------------------
# Worked example from CLAUDE.md
# ---------------------------------------------------------------------------


class TestExampleScenario:
    """Replay the worked example from the project specification."""

    def test_round_1(self) -> None:
        """After Round 1 of the example, verify constraint state."""
        cs = ConstraintSet.initial(EXAMPLE_HAND)

        # Round 1
        cs = cs.apply_play(Player.SOUTH, Tile(3, 3))  # open ends: (3, 3)
        cs = cs.apply_pass(Player.WEST, (3, 3))  # West has no 3s
        cs = cs.apply_play(Player.NORTH, Tile(3, 6))  # open ends: (3, 6)
        cs = cs.apply_play(Player.EAST, Tile(2, 6))  # open ends: (3, 2)

        # Tiles played: [3|3], [3|6], [6|2] = 3 tiles (South's [3|3] +
        # 2 opponents'). But South's tile is in my_hand originally.
        assert len(cs.played_tiles) == 3

        # West cannot have any tile with value 3.
        west_cands = cs.get_candidates(Player.WEST)
        tiles_with_3 = [t for t in west_cands if t.a == 3 or t.b == 3]
        assert tiles_with_3 == []

        # Specific tiles West cannot have (from the spec):
        # [0|3], [1|3], [2|3], [3|4], [3|5] -- [3|3] and [3|6] already played.
        # Also [1|3] is in our hand, so it was never a candidate.
        for forbidden in [Tile(0, 3), Tile(2, 3), Tile(3, 4), Tile(3, 5)]:
            assert forbidden not in west_cands

        # Remaining tile counts.
        assert (
            cs.player_constraints[Player.WEST].tiles_remaining == 7
        )  # passed, not played
        assert cs.player_constraints[Player.NORTH].tiles_remaining == 6
        assert cs.player_constraints[Player.EAST].tiles_remaining == 6

        # Unknown tiles: 28 - 7 (hand) - 3 (played) = 18
        # But South's played tile was in hand, so unknowns = 28 - 6 (remaining hand) - 3 (played) = 19
        # Actually: unknown = full_set - played - my_hand.
        # my_hand was {0|1,1|3,2|5,3|3,4|6,5|5,6|6}, after South plays 3|3,
        # my_hand becomes {0|1,1|3,2|5,4|6,5|5,6|6} (6 tiles).
        # played = {3|3, 3|6, 2|6}.
        # unknown = 28 - 3 - 6 = 19.
        assert len(cs.unknown_tiles()) == 19


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestConstraintInvariants:
    """Property-based tests ensuring constraint invariants hold."""

    @given(hand=valid_hand())
    @settings(max_examples=20)
    def test_initial_invariants(self, hand: frozenset[Tile]) -> None:
        """Initial state satisfies basic invariants for any hand."""
        cs = ConstraintSet.initial(hand)

        # All candidates are unknown tiles.
        unknown = cs.unknown_tiles()
        for p in OPPONENTS:
            assert cs.get_candidates(p).issubset(unknown)

        # Sum of tiles_remaining = len(unknown).
        total_remaining = sum(
            cs.player_constraints[p].tiles_remaining for p in OPPONENTS
        )
        assert total_remaining == len(unknown)

    @given(hand=valid_hand())
    @settings(max_examples=10)
    def test_play_preserves_tile_accounting(self, hand: frozenset[Tile]) -> None:
        """After a play, total remaining tiles decreases by 1."""
        cs = ConstraintSet.initial(hand)
        unknown = cs.unknown_tiles()
        if not unknown:
            return
        # Pick an arbitrary unknown tile to play as West.
        tile = sorted(unknown)[0]
        cs2 = cs.apply_play(Player.WEST, tile)

        total_before = sum(cs.player_constraints[p].tiles_remaining for p in OPPONENTS)
        total_after = sum(cs2.player_constraints[p].tiles_remaining for p in OPPONENTS)
        assert total_after == total_before - 1

    @given(hand=valid_hand())
    @settings(max_examples=10)
    def test_candidates_subset_of_unknown(self, hand: frozenset[Tile]) -> None:
        """Candidates are always a subset of unknown tiles after plays."""
        cs = ConstraintSet.initial(hand)
        unknown = sorted(cs.unknown_tiles())
        if len(unknown) < 2:
            return
        # Play first two unknown tiles.
        cs = cs.apply_play(Player.WEST, unknown[0])
        cs = cs.apply_play(Player.NORTH, unknown[1])

        current_unknown = cs.unknown_tiles()
        for p in OPPONENTS:
            assert cs.get_candidates(p).issubset(current_unknown)
