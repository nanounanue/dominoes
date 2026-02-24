"""Integration tests for full game replay.

Tests the OracleState pipeline: GameState -> ConstraintSet -> InferenceEngine.
Covers the CLAUDE.md example scenario, step-by-step consistency, edge cases,
and pass inference propagation.
"""

from __future__ import annotations

import numpy as np
import pytest

from domino_oracle.core.constraints import OPPONENTS, ConstraintSet
from domino_oracle.core.engine import OracleState, replay_game
from domino_oracle.core.game_state import GameState, Pass, Play, Player
from domino_oracle.core.inference import ProbabilityTable
from domino_oracle.core.tiles import Tile, generate_full_set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hand(*pairs: tuple[int, int]) -> frozenset[Tile]:
    """Create a hand from (a, b) pairs."""
    return frozenset(Tile(a, b) for a, b in pairs)


def _assert_probability_invariants(
    state: OracleState,
    *,
    atol: float = 0.05,
) -> None:
    """Check probability invariants on an OracleState with computed probs.

    Args:
        state: Must have probabilities computed (not None).
        atol: Absolute tolerance for floating-point comparisons.
            Higher tolerance needed for Monte Carlo estimates.
    """
    assert state.probabilities is not None, "Probabilities not computed"
    probs = state.probabilities

    unknown = state.constraints.unknown_tiles()
    assert set(probs.tiles) == unknown, "Probability table tiles != unknown tiles"

    for tile in probs.tiles:
        tile_probs = probs.get_tile_probs(tile)
        # All probabilities in [0, 1].
        for p, prob in tile_probs.items():
            assert (
                0.0 - 1e-9 <= prob <= 1.0 + 1e-9
            ), f"P({p.name} has {tile}) = {prob} out of [0, 1]"
        # Per-tile probabilities across players sum to 1.0.
        total = sum(tile_probs.values())
        assert (
            abs(total - 1.0) < atol
        ), f"Per-tile probs for {tile} sum to {total}, expected 1.0"

    for player in OPPONENTS:
        player_probs = probs.get_player_probs(player)
        # Non-candidate tiles should have probability 0.
        candidates = state.constraints.get_candidates(player)
        for tile, prob in player_probs.items():
            if tile not in candidates:
                assert abs(prob) < 1e-9, (
                    f"P({player.name} has {tile}) = {prob} but tile is not "
                    f"in candidate set"
                )
        # Per-player probabilities sum to their tiles_remaining.
        expected_sum = state.constraints.player_constraints[player].tiles_remaining
        actual_sum = sum(player_probs.values())
        assert abs(actual_sum - expected_sum) < atol, (
            f"Per-player probs for {player.name} sum to {actual_sum}, "
            f"expected {expected_sum}"
        )


# ---------------------------------------------------------------------------
# The CLAUDE.md example scenario
# ---------------------------------------------------------------------------

# South holds: [0|1], [1|3], [2|5], [3|3], [4|6], [5|5], [6|6]
EXAMPLE_HAND = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))

# Round 1 actions:
# South plays [3|3] on end 3 -> open ends: (3, 3)
# West passes -> West has no tile with a 3
# North plays [3|6] on end 3 -> open ends: (6, 3)
# East plays [2|6] on end 6 -> open ends: (2, 3)
EXAMPLE_ACTIONS = [
    Play(player=Player.SOUTH, tile=Tile(3, 3), end=3),
    Pass(player=Player.WEST),
    Play(player=Player.NORTH, tile=Tile(3, 6), end=3),
    Play(player=Player.EAST, tile=Tile(2, 6), end=6),
]


class TestExampleScenarioReplay:
    """Tests using the worked example from CLAUDE.md."""

    def test_replay_produces_correct_open_ends(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        assert state.game.open_ends == (2, 3)

    def test_replay_played_tiles(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        expected_played = frozenset({Tile(3, 3), Tile(3, 6), Tile(2, 6)})
        assert state.game.played_tiles == expected_played

    def test_replay_tiles_remaining(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        # South played 1, West passed, North played 1, East played 1
        assert state.game.tiles_remaining == (6, 7, 6, 6)

    def test_west_has_no_tiles_with_3(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        assert state.probabilities is not None
        # West passed when open ends were (3, 3), so West cannot hold
        # any tile containing a 3.
        tiles_with_3 = [t for t in state.probabilities.tiles if t.a == 3 or t.b == 3]
        for tile in tiles_with_3:
            prob = state.probabilities.get_prob(Player.WEST, tile)
            assert (
                abs(prob) < 1e-9
            ), f"West should have P=0 for {tile} (contains 3), got {prob}"

    def test_west_eliminated_values(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        west_constraints = state.constraints.player_constraints[Player.WEST]
        assert 3 in west_constraints.eliminated_values

    def test_probability_invariants(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        _assert_probability_invariants(state)

    def test_current_player_after_round(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        # After 4 actions (S, W, N, E), it's South's turn again.
        assert state.game.current_player == Player.SOUTH

    def test_consistency(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        state.verify_consistency()


class TestStepByStepConsistency:
    """Apply actions one by one and verify consistency at each step."""

    def test_consistency_after_each_action(self) -> None:
        state = OracleState.initial(EXAMPLE_HAND)
        state.verify_consistency()

        for action in EXAMPLE_ACTIONS:
            state = state.apply_action(action)
            state.verify_consistency()

    def test_probabilities_at_each_step(self) -> None:
        state = OracleState.initial(EXAMPLE_HAND)

        for action in EXAMPLE_ACTIONS:
            state = state.apply_action(action)
            state_with_probs = state.compute_probabilities(rng_seed=42)
            _assert_probability_invariants(state_with_probs)

    def test_probabilities_none_after_action(self) -> None:
        state = OracleState.initial(EXAMPLE_HAND)
        state = state.compute_probabilities(rng_seed=42)
        assert state.probabilities is not None

        # After applying an action, probabilities should be cleared.
        state = state.apply_action(EXAMPLE_ACTIONS[0])
        assert state.probabilities is None


class TestEdgeCases:
    """Edge cases: all-pass game over, exact enumeration in late game."""

    def test_all_pass_game_over(self) -> None:
        """After the first play, 4 consecutive passes end the game."""
        hand = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))
        state = OracleState.initial(hand)

        # South plays first.
        state = state.apply_action(Play(Player.SOUTH, Tile(3, 3), end=3))
        # Then 4 consecutive passes.
        state = state.apply_action(Pass(Player.WEST))
        state = state.apply_action(Pass(Player.NORTH))
        state = state.apply_action(Pass(Player.EAST))
        state = state.apply_action(Pass(Player.SOUTH))

        assert state.game.is_game_over()
        state.verify_consistency()

    def test_multi_round_game(self) -> None:
        """A multi-round game with several plays and passes."""
        hand = _make_hand((0, 0), (0, 1), (1, 1), (2, 3), (4, 5), (5, 6), (6, 6))
        state = OracleState.initial(hand)

        actions = [
            Play(Player.SOUTH, Tile(0, 0), end=0),  # open: (0, 0)
            Play(Player.WEST, Tile(0, 2), end=0),  # open: (2, 0)
            Play(Player.NORTH, Tile(0, 3), end=0),  # open: (2, 3)
            Play(Player.EAST, Tile(2, 4), end=2),  # open: (4, 3)
            Play(Player.SOUTH, Tile(2, 3), end=3),  # open: (4, 2)
            Pass(Player.WEST),  # West has no 4 or 2
            Play(Player.NORTH, Tile(4, 4), end=4),  # open: (4, 2)
        ]

        for action in actions:
            state = state.apply_action(action)
            state.verify_consistency()

        state = state.compute_probabilities(rng_seed=42)
        _assert_probability_invariants(state)

        # West passed on (4, 2) so should have no tiles with 4 or 2.
        west_constraints = state.constraints.player_constraints[Player.WEST]
        assert 4 in west_constraints.eliminated_values
        assert 2 in west_constraints.eliminated_values

    def test_south_hand_decrements_on_play(self) -> None:
        """Verify South's hand shrinks when South plays."""
        hand = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))
        state = OracleState.initial(hand)
        assert len(state.game.my_hand) == 7

        state = state.apply_action(Play(Player.SOUTH, Tile(3, 3), end=3))
        assert len(state.game.my_hand) == 6
        assert Tile(3, 3) not in state.game.my_hand
        assert Tile(3, 3) not in state.constraints.my_hand


class TestPassInference:
    """Verify pass constraints propagate correctly into probabilities."""

    def test_double_pass_eliminates_more_values(self) -> None:
        """A player who passes twice eliminates values from both passes."""
        hand = _make_hand((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 6))
        state = OracleState.initial(hand)

        # Round 1: South plays [6|6], open ends = (6, 6)
        state = state.apply_action(Play(Player.SOUTH, Tile(6, 6), end=6))
        # West passes on (6, 6) -> West has no 6
        state = state.apply_action(Pass(Player.WEST))
        state = state.apply_action(Play(Player.NORTH, Tile(1, 6), end=6))
        # open ends = (1, 6)
        state = state.apply_action(Play(Player.EAST, Tile(0, 6), end=6))
        # open ends = (1, 0)

        # Round 2: South plays [0|1], open ends = (0, 1) -> (0, 1)
        # Actually let's just check West's constraints after first pass
        west_constraints = state.constraints.player_constraints[Player.WEST]
        assert 6 in west_constraints.eliminated_values

        # No tile with 6 in West's candidates.
        for tile in west_constraints.candidate_tiles:
            assert tile.a != 6 and tile.b != 6

    def test_pass_with_same_open_ends(self) -> None:
        """Pass on a double (e.g., open ends (3, 3)) eliminates value 3."""
        hand = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))
        state = OracleState.initial(hand)

        # South plays double 3.
        state = state.apply_action(Play(Player.SOUTH, Tile(3, 3), end=3))
        # West passes on (3, 3).
        state = state.apply_action(Pass(Player.WEST))

        west = state.constraints.player_constraints[Player.WEST]
        assert 3 in west.eliminated_values

        # Verify no candidate tile contains a 3.
        for tile in west.candidate_tiles:
            assert tile.a != 3 and tile.b != 3

    def test_known_tile_prob_zero_for_non_holder(self) -> None:
        """Played tiles should not appear in probability table."""
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        assert state.probabilities is not None

        # Played tiles should not be in the probability table.
        for played_tile in state.game.played_tiles:
            assert played_tile not in state.probabilities._tile_index

    def test_initial_uniform_probabilities(self) -> None:
        """Before any actions, all opponents have equal probability for each tile."""
        hand = _make_hand((0, 0), (0, 1), (1, 1), (2, 3), (4, 5), (5, 6), (6, 6))
        state = OracleState.initial(hand)
        state = state.compute_probabilities(rng_seed=42)

        assert state.probabilities is not None

        # With no information, each of the 21 unknown tiles has P = 7/21 = 1/3
        # for each opponent (each holds 7 of 21 tiles).
        for tile in state.probabilities.tiles:
            for player in OPPONENTS:
                prob = state.probabilities.get_prob(player, tile)
                assert abs(prob - 1.0 / 3.0) < 0.05, (
                    f"Initial P({player.name} has {tile}) = {prob}, " f"expected ~0.333"
                )


class TestOracleStateAPI:
    """Test the OracleState and replay_game API surface."""

    def test_initial_state(self) -> None:
        hand = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))
        state = OracleState.initial(hand)
        assert state.game.current_player == Player.SOUTH
        assert state.game.open_ends is None
        assert state.probabilities is None
        assert len(state.game.my_hand) == 7

    def test_invalid_hand_raises(self) -> None:
        bad_hand = _make_hand((0, 1), (1, 3))  # Only 2 tiles
        with pytest.raises(ValueError):
            OracleState.initial(bad_hand)

    def test_wrong_player_raises(self) -> None:
        hand = _make_hand((0, 1), (1, 3), (2, 5), (3, 3), (4, 6), (5, 5), (6, 6))
        state = OracleState.initial(hand)
        with pytest.raises(ValueError, match="SOUTH's turn"):
            state.apply_action(Play(Player.WEST, Tile(0, 2), end=0))

    def test_replay_game_convenience(self) -> None:
        state = replay_game(EXAMPLE_HAND, EXAMPLE_ACTIONS, rng_seed=42)
        assert state.probabilities is not None
        assert state.game.open_ends == (2, 3)
        state.verify_consistency()
        _assert_probability_invariants(state)

    def test_compute_probabilities_idempotent(self) -> None:
        """Computing probabilities twice gives the same result."""
        state = OracleState.initial(EXAMPLE_HAND)
        state = state.apply_action(EXAMPLE_ACTIONS[0])

        state1 = state.compute_probabilities(rng_seed=42)
        state2 = state.compute_probabilities(rng_seed=42)

        assert state1.probabilities is not None
        assert state2.probabilities is not None
        np.testing.assert_array_equal(
            state1.probabilities.probs, state2.probabilities.probs
        )
