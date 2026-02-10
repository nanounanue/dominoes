"""Tests for game state machine."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domino_oracle.core.game_state import (
    Action,
    GameState,
    Pass,
    Play,
    Player,
    Team,
)
from domino_oracle.core.tiles import Tile, generate_full_set

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A canonical hand for South used in most tests.
SOUTH_HAND = frozenset(
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


def _make_initial() -> GameState:
    """Create a standard initial game state."""
    return GameState.initial(SOUTH_HAND)


def _tiles_list() -> list[Tile]:
    """Return a sorted list of all 28 tiles for deterministic sampling."""
    return sorted(generate_full_set())


# Hypothesis strategy: pick 7 distinct tiles from the full set for a hand.
@st.composite
def random_hand(draw: st.DrawFn) -> frozenset[Tile]:
    """Strategy that draws a valid 7-tile hand from the domino set."""
    tiles = _tiles_list()
    indices = draw(st.lists(st.integers(0, 27), min_size=7, max_size=7, unique=True))
    return frozenset(tiles[i] for i in indices)


# ---------------------------------------------------------------------------
# Player and Team
# ---------------------------------------------------------------------------


class TestPlayer:
    """Tests for the Player enum."""

    def test_player_values(self) -> None:
        assert Player.SOUTH.value == 0
        assert Player.WEST.value == 1
        assert Player.NORTH.value == 2
        assert Player.EAST.value == 3

    def test_four_players(self) -> None:
        assert len(Player) == 4


class TestTeam:
    """Tests for the Team enum."""

    def test_team_values(self) -> None:
        assert Team.NS.value == 0
        assert Team.WE.value == 1


# ---------------------------------------------------------------------------
# GameState.initial
# ---------------------------------------------------------------------------


class TestGameStateInitial:
    """Tests for initial game state creation."""

    def test_initial_state_basic(self) -> None:
        state = _make_initial()
        assert state.current_player == Player.SOUTH
        assert state.open_ends is None
        assert state.played_tiles == frozenset()
        assert state.history == ()
        assert state.tiles_remaining == (7, 7, 7, 7)
        assert len(state.all_tiles) == 28
        assert state.my_hand == SOUTH_HAND

    def test_initial_state_hand_too_small(self) -> None:
        with pytest.raises(ValueError, match="exactly 7 tiles"):
            GameState.initial(frozenset([Tile(0, 0), Tile(1, 1)]))

    def test_initial_state_hand_too_large(self) -> None:
        tiles = sorted(generate_full_set())[:8]
        with pytest.raises(ValueError, match="exactly 7 tiles"):
            GameState.initial(frozenset(tiles))

    def test_initial_state_empty_hand(self) -> None:
        with pytest.raises(ValueError, match="exactly 7 tiles"):
            GameState.initial(frozenset())

    @given(hand=random_hand())
    @settings(max_examples=50)
    def test_hypothesis_initial_state_valid(self, hand: frozenset[Tile]) -> None:
        state = GameState.initial(hand)
        assert state.my_hand == hand
        assert len(state.my_hand) == 7
        assert state.tiles_remaining == (7, 7, 7, 7)
        assert state.current_player == Player.SOUTH


# ---------------------------------------------------------------------------
# GameState.next_player
# ---------------------------------------------------------------------------


class TestNextPlayer:
    """Tests for clockwise turn rotation."""

    def test_south_to_west(self) -> None:
        assert GameState.next_player(Player.SOUTH) == Player.WEST

    def test_west_to_north(self) -> None:
        assert GameState.next_player(Player.WEST) == Player.NORTH

    def test_north_to_east(self) -> None:
        assert GameState.next_player(Player.NORTH) == Player.EAST

    def test_east_to_south(self) -> None:
        assert GameState.next_player(Player.EAST) == Player.SOUTH

    def test_full_cycle(self) -> None:
        player = Player.SOUTH
        for expected in [Player.WEST, Player.NORTH, Player.EAST, Player.SOUTH]:
            player = GameState.next_player(player)
            assert player == expected


# ---------------------------------------------------------------------------
# GameState.player_team
# ---------------------------------------------------------------------------


class TestPlayerTeam:
    """Tests for team membership."""

    def test_south_is_ns(self) -> None:
        assert GameState.player_team(Player.SOUTH) == Team.NS

    def test_north_is_ns(self) -> None:
        assert GameState.player_team(Player.NORTH) == Team.NS

    def test_west_is_we(self) -> None:
        assert GameState.player_team(Player.WEST) == Team.WE

    def test_east_is_we(self) -> None:
        assert GameState.player_team(Player.EAST) == Team.WE


# ---------------------------------------------------------------------------
# GameState.apply_action — Play
# ---------------------------------------------------------------------------


class TestApplyPlay:
    """Tests for applying Play actions."""

    def test_first_play_sets_open_ends(self) -> None:
        state = _make_initial()
        # South plays [3|3] — a double.
        play = Play(player=Player.SOUTH, tile=Tile(3, 3), end=3)
        new_state = state.apply_action(play)

        assert new_state.open_ends == (3, 3)
        assert Tile(3, 3) in new_state.played_tiles
        assert new_state.current_player == Player.WEST
        assert new_state.tiles_remaining == (6, 7, 7, 7)

    def test_first_play_non_double(self) -> None:
        state = _make_initial()
        play = Play(player=Player.SOUTH, tile=Tile(2, 5), end=2)
        new_state = state.apply_action(play)

        assert new_state.open_ends == (2, 5)
        assert Tile(2, 5) in new_state.played_tiles

    def test_play_updates_open_end_left(self) -> None:
        """Play on the left open end updates the left side."""
        state = _make_initial()
        # First play: [2|5] with end=2 → open ends (2, 5)
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(2, 5), end=2))
        # West plays [1|2] matching end=2 (left) → new left end = 1
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(1, 2), end=2))
        assert state.open_ends == (1, 5)

    def test_play_updates_open_end_right(self) -> None:
        """Play on the right open end updates the right side."""
        state = _make_initial()
        # First play: [2|5] with end=2 → open ends (2, 5)
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(2, 5), end=2))
        # West plays [5|6] matching end=5 (right) → new right end = 6
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(5, 6), end=5))
        assert state.open_ends == (2, 6)

    def test_double_play_sets_equal_open_ends(self) -> None:
        state = _make_initial()
        play = Play(player=Player.SOUTH, tile=Tile(3, 3), end=3)
        new_state = state.apply_action(play)
        assert new_state.open_ends == (3, 3)
        assert new_state.get_open_end_values() == frozenset({3})

    def test_south_play_removes_from_hand(self) -> None:
        state = _make_initial()
        tile = Tile(3, 3)
        assert tile in state.my_hand
        play = Play(player=Player.SOUTH, tile=tile, end=3)
        new_state = state.apply_action(play)
        assert tile not in new_state.my_hand

    def test_other_player_play_does_not_change_hand(self) -> None:
        state = _make_initial()
        # South plays first
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        original_hand = state.my_hand
        # West plays
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(0, 3), end=3))
        assert state.my_hand == original_hand

    def test_play_wrong_player_raises(self) -> None:
        state = _make_initial()
        with pytest.raises(ValueError, match="SOUTH's turn"):
            state.apply_action(Play(player=Player.WEST, tile=Tile(0, 3), end=0))

    def test_play_tile_not_matching_end_raises(self) -> None:
        state = _make_initial()
        with pytest.raises(ValueError, match="does not contain"):
            state.apply_action(Play(player=Player.SOUTH, tile=Tile(2, 5), end=3))

    def test_play_end_not_matching_open_end_raises(self) -> None:
        state = _make_initial()
        # First play sets open ends to (3, 3)
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        # West tries to play on end=5 which is not open
        with pytest.raises(ValueError, match="does not match either open end"):
            state.apply_action(Play(player=Player.WEST, tile=Tile(5, 6), end=5))

    def test_play_already_played_tile_raises(self) -> None:
        state = _make_initial()
        tile = Tile(3, 3)
        state = state.apply_action(Play(player=Player.SOUTH, tile=tile, end=3))
        with pytest.raises(ValueError, match="already been played"):
            state.apply_action(Play(player=Player.WEST, tile=tile, end=3))

    def test_history_records_actions(self) -> None:
        state = _make_initial()
        play1 = Play(player=Player.SOUTH, tile=Tile(3, 3), end=3)
        state = state.apply_action(play1)
        assert state.history == (play1,)

        play2 = Play(player=Player.WEST, tile=Tile(0, 3), end=3)
        state = state.apply_action(play2)
        assert state.history == (play1, play2)


# ---------------------------------------------------------------------------
# GameState.apply_action — Pass
# ---------------------------------------------------------------------------


class TestApplyPass:
    """Tests for applying Pass actions."""

    def test_pass_advances_turn(self) -> None:
        state = _make_initial()
        # Need at least one play before anyone can pass.
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        pass_action = Pass(player=Player.WEST)
        new_state = state.apply_action(pass_action)
        assert new_state.current_player == Player.NORTH

    def test_pass_does_not_change_board(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        before_open_ends = state.open_ends
        before_played = state.played_tiles
        before_remaining = state.tiles_remaining
        state = state.apply_action(Pass(player=Player.WEST))
        assert state.open_ends == before_open_ends
        assert state.played_tiles == before_played
        assert state.tiles_remaining == before_remaining

    def test_pass_on_first_turn_raises(self) -> None:
        state = _make_initial()
        with pytest.raises(ValueError, match="Cannot pass on the first turn"):
            state.apply_action(Pass(player=Player.SOUTH))

    def test_pass_wrong_player_raises(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        with pytest.raises(ValueError, match="WEST's turn"):
            state.apply_action(Pass(player=Player.NORTH))

    def test_pass_recorded_in_history(self) -> None:
        state = _make_initial()
        play = Play(player=Player.SOUTH, tile=Tile(3, 3), end=3)
        state = state.apply_action(play)
        pass_act = Pass(player=Player.WEST)
        state = state.apply_action(pass_act)
        assert state.history == (play, pass_act)


# ---------------------------------------------------------------------------
# Turn order
# ---------------------------------------------------------------------------


class TestTurnOrder:
    """Test that turn order cycles correctly through a full round."""

    def test_full_round_of_plays(self) -> None:
        state = _make_initial()
        # South plays [3|3] → open ends (3,3)
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        assert state.current_player == Player.WEST

        # West plays [0|3] matching end=3 → open ends (0, 3)
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(0, 3), end=3))
        assert state.current_player == Player.NORTH

        # North plays [0, 4] matching end=0 → open ends (4, 3)
        state = state.apply_action(Play(player=Player.NORTH, tile=Tile(0, 4), end=0))
        assert state.current_player == Player.EAST

        # East plays [3, 5] matching end=3 → open ends (4, 5)
        state = state.apply_action(Play(player=Player.EAST, tile=Tile(3, 5), end=3))
        assert state.current_player == Player.SOUTH


# ---------------------------------------------------------------------------
# unknown_tiles
# ---------------------------------------------------------------------------


class TestUnknownTiles:
    """Tests for the unknown_tiles method."""

    def test_initial_unknown_tiles(self) -> None:
        state = _make_initial()
        unknown = state.unknown_tiles()
        assert len(unknown) == 21  # 28 - 7 in hand
        assert unknown.isdisjoint(state.my_hand)
        assert unknown.isdisjoint(state.played_tiles)

    def test_unknown_after_play(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        unknown = state.unknown_tiles()
        # Hand had 7, now 6. Played 1. Unknown = 28 - 6 - 1 = 21.
        assert len(unknown) == 21
        assert Tile(3, 3) not in unknown  # played
        assert Tile(3, 3) not in state.my_hand  # removed from hand

    def test_unknown_after_other_player_play(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(0, 3), end=3))
        unknown = state.unknown_tiles()
        # 28 - 6 (hand) - 2 (played) = 20
        assert len(unknown) == 20

    def test_invariant_hand_plus_played_plus_unknown_equals_all(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        state = state.apply_action(Play(player=Player.WEST, tile=Tile(0, 3), end=3))
        assert (
            state.my_hand | state.played_tiles | state.unknown_tiles()
            == state.all_tiles
        )


# ---------------------------------------------------------------------------
# is_game_over
# ---------------------------------------------------------------------------


class TestIsGameOver:
    """Tests for end-of-game detection."""

    def test_not_over_initially(self) -> None:
        assert _make_initial().is_game_over() is False

    def test_over_when_player_has_zero_tiles(self) -> None:
        """Simulate a state where a player has 0 tiles."""
        state = _make_initial()
        # Directly construct a state with 0 remaining for one player.
        over_state = GameState(
            all_tiles=state.all_tiles,
            my_hand=frozenset(),
            history=(),
            open_ends=(1, 2),
            played_tiles=frozenset(),
            current_player=Player.SOUTH,
            tiles_remaining=(0, 7, 7, 7),
        )
        assert over_state.is_game_over() is True

    def test_over_with_four_consecutive_passes(self) -> None:
        state = _make_initial()
        # Play one tile, then 4 passes.
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        state = state.apply_action(Pass(player=Player.WEST))
        state = state.apply_action(Pass(player=Player.NORTH))
        state = state.apply_action(Pass(player=Player.EAST))
        state = state.apply_action(Pass(player=Player.SOUTH))
        assert state.is_game_over() is True

    def test_not_over_with_three_passes(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        state = state.apply_action(Pass(player=Player.WEST))
        state = state.apply_action(Pass(player=Player.NORTH))
        state = state.apply_action(Pass(player=Player.EAST))
        assert state.is_game_over() is False

    def test_not_over_with_play_between_passes(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        state = state.apply_action(Pass(player=Player.WEST))
        state = state.apply_action(Pass(player=Player.NORTH))
        # East plays instead of passing — breaks the streak.
        state = state.apply_action(Play(player=Player.EAST, tile=Tile(3, 6), end=3))
        state = state.apply_action(Pass(player=Player.SOUTH))
        assert state.is_game_over() is False


# ---------------------------------------------------------------------------
# get_open_end_values
# ---------------------------------------------------------------------------


class TestGetOpenEndValues:
    """Tests for open end value extraction."""

    def test_no_open_ends_initially(self) -> None:
        state = _make_initial()
        assert state.get_open_end_values() == frozenset()

    def test_double_gives_single_value(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        assert state.get_open_end_values() == frozenset({3})

    def test_non_double_gives_two_values(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(2, 5), end=2))
        assert state.get_open_end_values() == frozenset({2, 5})


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """Verify that applying actions does not mutate the original state."""

    def test_apply_play_does_not_mutate_original(self) -> None:
        state = _make_initial()
        original_hand = state.my_hand
        original_played = state.played_tiles
        original_open = state.open_ends
        original_remaining = state.tiles_remaining

        _new = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))

        assert state.my_hand == original_hand
        assert state.played_tiles == original_played
        assert state.open_ends == original_open
        assert state.tiles_remaining == original_remaining

    def test_apply_pass_does_not_mutate_original(self) -> None:
        state = _make_initial()
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        original_player = state.current_player
        _new = state.apply_action(Pass(player=Player.WEST))
        assert state.current_player == original_player


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------


class TestHypothesisInvariants:
    """Property-based tests for game state invariants."""

    @given(hand=random_hand())
    @settings(max_examples=50)
    def test_tiles_partition_invariant(self, hand: frozenset[Tile]) -> None:
        """my_hand + played_tiles + unknown_tiles == all_tiles."""
        state = GameState.initial(hand)
        assert (
            state.my_hand | state.played_tiles | state.unknown_tiles()
            == state.all_tiles
        )

    @given(hand=random_hand())
    @settings(max_examples=50)
    def test_initial_tiles_remaining_sum(self, hand: frozenset[Tile]) -> None:
        """All four players start with 7 tiles (total 28)."""
        state = GameState.initial(hand)
        assert sum(state.tiles_remaining) == 28

    @given(hand=random_hand())
    @settings(max_examples=50)
    def test_unknown_tiles_count(self, hand: frozenset[Tile]) -> None:
        """Initially, 21 tiles are unknown (28 - 7 in hand)."""
        state = GameState.initial(hand)
        assert len(state.unknown_tiles()) == 21


# ---------------------------------------------------------------------------
# Scenario from CLAUDE.md
# ---------------------------------------------------------------------------


class TestExampleScenario:
    """Replay the example scenario from the project docs.

    You (South) hold: [0|1], [1|3], [2|5], [3|3], [4|6], [5|5], [6|6]
    Round 1:
        South plays [3|3] -> open ends: (3, 3)
        West passes -> West has NO tile with a 3
        North plays [3|6] -> open ends: (3, 6)
        East plays [2|6] -> open ends: (3, 2)
    """

    def test_full_round_one(self) -> None:
        state = GameState.initial(SOUTH_HAND)

        # South plays [3|3]
        state = state.apply_action(Play(player=Player.SOUTH, tile=Tile(3, 3), end=3))
        assert state.open_ends == (3, 3)
        assert state.current_player == Player.WEST
        assert state.tiles_remaining == (6, 7, 7, 7)
        assert Tile(3, 3) not in state.my_hand

        # West passes
        state = state.apply_action(Pass(player=Player.WEST))
        assert state.current_player == Player.NORTH
        assert state.tiles_remaining == (6, 7, 7, 7)  # unchanged

        # North plays [3|6] matching end=3 (left), new left end = 6
        state = state.apply_action(Play(player=Player.NORTH, tile=Tile(3, 6), end=3))
        assert state.open_ends == (6, 3)
        assert state.current_player == Player.EAST
        assert state.tiles_remaining == (6, 7, 6, 7)

        # East plays [2|6] matching end=6 (left), new left end = 2
        state = state.apply_action(Play(player=Player.EAST, tile=Tile(2, 6), end=6))
        assert state.open_ends == (2, 3)
        assert state.current_player == Player.SOUTH
        assert state.tiles_remaining == (6, 7, 6, 6)

        # Verify partition invariant.
        assert (
            state.my_hand | state.played_tiles | state.unknown_tiles()
            == state.all_tiles
        )

        # 3 tiles played + 6 in hand = 9 accounted for. Unknown = 19.
        assert len(state.unknown_tiles()) == 19
        assert len(state.played_tiles) == 3
        assert len(state.my_hand) == 6

        # Game is not over.
        assert state.is_game_over() is False

        # History has 4 actions.
        assert len(state.history) == 4
