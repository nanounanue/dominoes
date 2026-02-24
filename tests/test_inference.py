"""Tests for inference engine.

Covers Monte Carlo sampling, exact enumeration, probability invariants,
agreement between methods, and auto-dispatch logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domino_oracle.core.constraints import (
    OPPONENTS,
    ConstraintSet,
    PlayerConstraints,
)
from domino_oracle.core.game_state import Player
from domino_oracle.core.inference import (
    _EXACT_THRESHOLD,
    ProbabilityTable,
    auto_marginals,
    exact_marginals,
    monte_carlo_marginals,
)
from domino_oracle.core.tiles import Tile, generate_full_set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_SET = generate_full_set()

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


def _make_late_game() -> ConstraintSet:
    """Build a late-game state with few unknown tiles for exact enumeration.

    Plays enough tiles to bring unknown count under the exact threshold.
    """
    cs = ConstraintSet.initial(EXAMPLE_HAND)

    # Round 1: South plays [3|3], West passes, North plays [3|6], East plays [2|6]
    cs = cs.apply_play(Player.SOUTH, Tile(3, 3))
    cs = cs.apply_pass(Player.WEST, (3, 3))
    cs = cs.apply_play(Player.NORTH, Tile(3, 6))
    cs = cs.apply_play(Player.EAST, Tile(2, 6))

    # Round 2
    cs = cs.apply_play(Player.SOUTH, Tile(1, 3))
    cs = cs.apply_play(Player.WEST, Tile(0, 2))
    cs = cs.apply_play(Player.NORTH, Tile(2, 2))
    cs = cs.apply_play(Player.EAST, Tile(2, 4))

    # Round 3
    cs = cs.apply_play(Player.SOUTH, Tile(4, 6))
    cs = cs.apply_play(Player.WEST, Tile(4, 4))
    cs = cs.apply_play(Player.NORTH, Tile(0, 4))
    cs = cs.apply_play(Player.EAST, Tile(1, 4))

    return cs


def _make_tiny_game() -> ConstraintSet:
    """Build a very small game state with only 3 unknown tiles.

    Each player has exactly 1 remaining tile, for easy verification.
    """
    hand = EXAMPLE_HAND
    a, b, c = Tile(0, 0), Tile(1, 1), Tile(2, 2)
    # All tiles except hand and {a, b, c} are played.
    played = FULL_SET - hand - {a, b, c}

    # No constraints: all three are candidates for everyone.
    pc = {
        Player.WEST: PlayerConstraints(
            candidate_tiles=frozenset([a, b, c]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
        Player.NORTH: PlayerConstraints(
            candidate_tiles=frozenset([a, b, c]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
        Player.EAST: PlayerConstraints(
            candidate_tiles=frozenset([a, b, c]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
    }
    return ConstraintSet(
        player_constraints=pc,
        played_tiles=played,
        my_hand=hand,
    )


def _make_determined_game() -> ConstraintSet:
    """Build a game where each player's hand is fully determined.

    West can only hold {A}, North only {B}, East only {C}.
    """
    hand = EXAMPLE_HAND
    a, b, c = Tile(0, 0), Tile(1, 1), Tile(2, 2)
    played = FULL_SET - hand - {a, b, c}

    pc = {
        Player.WEST: PlayerConstraints(
            candidate_tiles=frozenset([a]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
        Player.NORTH: PlayerConstraints(
            candidate_tiles=frozenset([b]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
        Player.EAST: PlayerConstraints(
            candidate_tiles=frozenset([c]),
            tiles_remaining=1,
            eliminated_values=frozenset(),
        ),
    }
    return ConstraintSet(
        player_constraints=pc,
        played_tiles=played,
        my_hand=hand,
    )


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
# ProbabilityTable
# ---------------------------------------------------------------------------


class TestProbabilityTable:
    """Tests for the ProbabilityTable dataclass."""

    def test_get_prob(self) -> None:
        """get_prob returns the correct element."""
        tiles = [Tile(0, 0), Tile(1, 1)]
        players = [Player.WEST, Player.NORTH]
        probs = np.array([[0.6, 0.3], [0.4, 0.7]])
        pt = ProbabilityTable(tiles=tiles, players=players, probs=probs)

        assert pt.get_prob(Player.WEST, Tile(0, 0)) == pytest.approx(0.6)
        assert pt.get_prob(Player.WEST, Tile(1, 1)) == pytest.approx(0.3)
        assert pt.get_prob(Player.NORTH, Tile(0, 0)) == pytest.approx(0.4)
        assert pt.get_prob(Player.NORTH, Tile(1, 1)) == pytest.approx(0.7)

    def test_get_player_probs(self) -> None:
        """get_player_probs returns all tile probs for a player."""
        tiles = [Tile(0, 0), Tile(1, 1)]
        players = [Player.WEST]
        probs = np.array([[0.6, 0.4]])
        pt = ProbabilityTable(tiles=tiles, players=players, probs=probs)

        result = pt.get_player_probs(Player.WEST)
        assert result[Tile(0, 0)] == pytest.approx(0.6)
        assert result[Tile(1, 1)] == pytest.approx(0.4)

    def test_get_tile_probs(self) -> None:
        """get_tile_probs returns all player probs for a tile."""
        tiles = [Tile(0, 0)]
        players = [Player.WEST, Player.NORTH, Player.EAST]
        probs = np.array([[0.3], [0.5], [0.2]])
        pt = ProbabilityTable(tiles=tiles, players=players, probs=probs)

        result = pt.get_tile_probs(Tile(0, 0))
        assert result[Player.WEST] == pytest.approx(0.3)
        assert result[Player.NORTH] == pytest.approx(0.5)
        assert result[Player.EAST] == pytest.approx(0.2)

    def test_missing_tile_raises(self) -> None:
        """Accessing a tile not in the table raises KeyError."""
        pt = ProbabilityTable(
            tiles=[Tile(0, 0)],
            players=[Player.WEST],
            probs=np.array([[1.0]]),
        )
        with pytest.raises(KeyError):
            pt.get_prob(Player.WEST, Tile(6, 6))

    def test_missing_player_raises(self) -> None:
        """Accessing a player not in the table raises KeyError."""
        pt = ProbabilityTable(
            tiles=[Tile(0, 0)],
            players=[Player.WEST],
            probs=np.array([[1.0]]),
        )
        with pytest.raises(KeyError):
            pt.get_prob(Player.EAST, Tile(0, 0))


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------


class TestExactMarginals:
    """Tests for exact_marginals()."""

    def test_tiny_uniform(self) -> None:
        """With 3 tiles and 3 players (1 each), uniform over permutations."""
        cs = _make_tiny_game()
        pt = exact_marginals(cs)

        # Each tile has P = 1/3 for each player (by symmetry).
        for tile in pt.tiles:
            for player in pt.players:
                assert pt.get_prob(player, tile) == pytest.approx(1.0 / 3.0, abs=1e-10)

    def test_determined(self) -> None:
        """Fully determined hands yield P=1 / P=0."""
        cs = _make_determined_game()
        pt = exact_marginals(cs)

        assert pt.get_prob(Player.WEST, Tile(0, 0)) == pytest.approx(1.0)
        assert pt.get_prob(Player.NORTH, Tile(1, 1)) == pytest.approx(1.0)
        assert pt.get_prob(Player.EAST, Tile(2, 2)) == pytest.approx(1.0)

        # Cross-player probabilities are 0.
        assert pt.get_prob(Player.WEST, Tile(1, 1)) == pytest.approx(0.0)
        assert pt.get_prob(Player.NORTH, Tile(0, 0)) == pytest.approx(0.0)
        assert pt.get_prob(Player.EAST, Tile(0, 0)) == pytest.approx(0.0)

    def test_tile_prob_sums_to_one(self) -> None:
        """For each unknown tile, probabilities across players sum to 1."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        for tile in pt.tiles:
            total = sum(pt.get_prob(p, tile) for p in pt.players)
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_player_prob_sums_to_remaining(self) -> None:
        """For each player, probabilities across tiles sum to tiles_remaining."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        for player in pt.players:
            total = sum(pt.get_player_probs(player).values())
            expected = cs.player_constraints[player].tiles_remaining
            assert total == pytest.approx(expected, abs=1e-10)

    def test_probs_in_range(self) -> None:
        """All probabilities are in [0, 1]."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        assert np.all(pt.probs >= 0.0)
        assert np.all(pt.probs <= 1.0)

    def test_non_candidate_prob_is_zero(self) -> None:
        """If a tile is not in a player's candidate set, P = 0."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        for player in pt.players:
            cands = cs.get_candidates(player)
            for tile in pt.tiles:
                if tile not in cands:
                    assert pt.get_prob(player, tile) == pytest.approx(0.0)

    def test_too_many_unknowns_raises(self) -> None:
        """Exact enumeration rejects states with > 18 unknown tiles."""
        cs = ConstraintSet.initial(EXAMPLE_HAND)
        # 21 unknowns at start.
        with pytest.raises(ValueError, match="Too many unknown tiles"):
            exact_marginals(cs)

    def test_late_game_scenario(self) -> None:
        """Verify exact marginals on the late-game scenario are internally
        consistent and non-trivial."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        # Should have some tiles with probabilities strictly between 0 and 1.
        has_uncertain = False
        for tile in pt.tiles:
            for player in pt.players:
                p = pt.get_prob(player, tile)
                if 0.01 < p < 0.99:
                    has_uncertain = True
                    break
            if has_uncertain:
                break
        assert has_uncertain, "Expected at least one uncertain tile probability"


# ---------------------------------------------------------------------------
# Monte Carlo sampling
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    """Tests for monte_carlo_marginals()."""

    def test_tiny_uniform_mc(self) -> None:
        """MC on tiny symmetric case should be close to 1/3."""
        cs = _make_tiny_game()
        pt = monte_carlo_marginals(cs, n_samples=100_000, rng_seed=42)

        for tile in pt.tiles:
            for player in pt.players:
                assert pt.get_prob(player, tile) == pytest.approx(1.0 / 3.0, abs=0.03)

    def test_determined_mc(self) -> None:
        """MC on fully determined case yields P=1 / P=0."""
        cs = _make_determined_game()
        pt = monte_carlo_marginals(cs, n_samples=10_000, rng_seed=42)

        assert pt.get_prob(Player.WEST, Tile(0, 0)) == pytest.approx(1.0)
        assert pt.get_prob(Player.NORTH, Tile(1, 1)) == pytest.approx(1.0)
        assert pt.get_prob(Player.EAST, Tile(2, 2)) == pytest.approx(1.0)

    def test_tile_prob_sums_to_one_mc(self) -> None:
        """MC: per-tile sums = 1."""
        cs = _make_late_game()
        pt = monte_carlo_marginals(cs, n_samples=50_000, rng_seed=123)

        for tile in pt.tiles:
            total = sum(pt.get_prob(p, tile) for p in pt.players)
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_player_prob_sums_to_remaining_mc(self) -> None:
        """MC: per-player sums = tiles_remaining."""
        cs = _make_late_game()
        pt = monte_carlo_marginals(cs, n_samples=50_000, rng_seed=123)

        for player in pt.players:
            total = sum(pt.get_player_probs(player).values())
            expected = cs.player_constraints[player].tiles_remaining
            assert total == pytest.approx(expected, abs=0.1)

    def test_probs_in_range_mc(self) -> None:
        """MC: all probabilities in [0, 1]."""
        cs = _make_late_game()
        pt = monte_carlo_marginals(cs, n_samples=10_000, rng_seed=42)

        assert np.all(pt.probs >= 0.0)
        assert np.all(pt.probs <= 1.0)

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical results."""
        cs = _make_late_game()
        pt1 = monte_carlo_marginals(cs, n_samples=5_000, rng_seed=99)
        pt2 = monte_carlo_marginals(cs, n_samples=5_000, rng_seed=99)

        np.testing.assert_array_equal(pt1.probs, pt2.probs)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different results (with high probability)."""
        cs = _make_late_game()
        pt1 = monte_carlo_marginals(cs, n_samples=5_000, rng_seed=1)
        pt2 = monte_carlo_marginals(cs, n_samples=5_000, rng_seed=2)

        # Extremely unlikely to be identical.
        assert not np.array_equal(pt1.probs, pt2.probs)

    def test_no_valid_samples_raises(self) -> None:
        """If constraints are effectively impossible, raise ValueError."""
        hand = EXAMPLE_HAND
        a = Tile(0, 0)
        played = FULL_SET - hand - {a}

        # West must have tile A (1 remaining, 1 candidate).
        # North must have tile A (1 remaining, only A candidate) -- impossible.
        # East has no candidates and 0 remaining (but total unknowns = 1 != 1+1+0).
        # This creates an inconsistent state on purpose.
        pc = {
            Player.WEST: PlayerConstraints(
                candidate_tiles=frozenset([a]),
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.NORTH: PlayerConstraints(
                candidate_tiles=frozenset([a]),
                tiles_remaining=1,
                eliminated_values=frozenset(),
            ),
            Player.EAST: PlayerConstraints(
                candidate_tiles=frozenset(),
                tiles_remaining=0,
                eliminated_values=frozenset(),
            ),
        }
        cs = ConstraintSet(
            player_constraints=pc,
            played_tiles=played,
            my_hand=hand,
        )
        # The unknown tile is just {a}, but remaining = 1+1+0 = 2 != 1.
        # Sampling will never produce a valid config because we can't
        # assign 1 tile to fill both West (1) and North (1).
        with pytest.raises(ValueError, match="No valid configurations"):
            monte_carlo_marginals(cs, n_samples=1_000, rng_seed=42)


# ---------------------------------------------------------------------------
# MC vs Exact agreement
# ---------------------------------------------------------------------------


class TestMCExactAgreement:
    """Monte Carlo and exact should agree on small cases."""

    def test_agreement_tiny(self) -> None:
        """On the tiny symmetric case, both methods agree closely."""
        cs = _make_tiny_game()
        pt_exact = exact_marginals(cs)
        pt_mc = monte_carlo_marginals(cs, n_samples=200_000, rng_seed=42)

        for tile in pt_exact.tiles:
            for player in pt_exact.players:
                exact_p = pt_exact.get_prob(player, tile)
                mc_p = pt_mc.get_prob(player, tile)
                assert mc_p == pytest.approx(exact_p, abs=0.02)

    def test_agreement_late_game(self) -> None:
        """On the late-game scenario, MC and exact agree within tolerance."""
        cs = _make_late_game()
        pt_exact = exact_marginals(cs)
        pt_mc = monte_carlo_marginals(cs, n_samples=200_000, rng_seed=42)

        for tile in pt_exact.tiles:
            for player in pt_exact.players:
                exact_p = pt_exact.get_prob(player, tile)
                mc_p = pt_mc.get_prob(player, tile)
                assert mc_p == pytest.approx(exact_p, abs=0.05), (
                    f"Disagreement for {player} / {tile}: "
                    f"exact={exact_p:.4f}, mc={mc_p:.4f}"
                )

    def test_agreement_determined(self) -> None:
        """Both methods agree on fully determined case."""
        cs = _make_determined_game()
        pt_exact = exact_marginals(cs)
        pt_mc = monte_carlo_marginals(cs, n_samples=10_000, rng_seed=42)

        for tile in pt_exact.tiles:
            for player in pt_exact.players:
                exact_p = pt_exact.get_prob(player, tile)
                mc_p = pt_mc.get_prob(player, tile)
                assert mc_p == pytest.approx(exact_p, abs=1e-10)


# ---------------------------------------------------------------------------
# Auto-dispatch
# ---------------------------------------------------------------------------


class TestAutoMarginals:
    """Tests for auto_marginals() dispatch logic."""

    def test_dispatches_exact_for_small(self) -> None:
        """auto_marginals uses exact when unknowns <= threshold."""
        cs = _make_tiny_game()
        assert len(cs.unknown_tiles()) <= _EXACT_THRESHOLD
        pt = auto_marginals(cs)
        # Verify it produced the same result as exact.
        pt_exact = exact_marginals(cs)
        np.testing.assert_array_almost_equal(pt.probs, pt_exact.probs)

    def test_dispatches_mc_for_large(self) -> None:
        """auto_marginals uses MC when unknowns > threshold."""
        cs = ConstraintSet.initial(EXAMPLE_HAND)
        assert len(cs.unknown_tiles()) > _EXACT_THRESHOLD
        # Should not raise (exact would raise for >18 unknowns).
        pt = auto_marginals(cs, mc_samples=10_000, rng_seed=42)
        assert pt.probs.shape[0] == 3  # 3 players
        assert pt.probs.shape[1] == 21  # 21 unknown tiles

    def test_late_game_auto(self) -> None:
        """auto_marginals on late game matches exact."""
        cs = _make_late_game()
        pt_auto = auto_marginals(cs)
        pt_exact = exact_marginals(cs)
        np.testing.assert_array_almost_equal(pt_auto.probs, pt_exact.probs)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests for invariants
# ---------------------------------------------------------------------------


class TestInferenceInvariants:
    """Property-based tests on probability invariants."""

    @given(hand=valid_hand())
    @settings(max_examples=5, deadline=30_000)
    def test_mc_invariants_initial(self, hand: frozenset[Tile]) -> None:
        """MC on any initial state satisfies probability invariants."""
        cs = ConstraintSet.initial(hand)
        pt = monte_carlo_marginals(cs, n_samples=5_000, rng_seed=42)

        # All probs in [0, 1].
        assert np.all(pt.probs >= -1e-10)
        assert np.all(pt.probs <= 1.0 + 1e-10)

        # Per-tile sum = 1.
        for tile in pt.tiles:
            total = sum(pt.get_prob(p, tile) for p in pt.players)
            assert total == pytest.approx(1.0, abs=0.05)

        # Per-player sum = tiles_remaining.
        for player in pt.players:
            total = sum(pt.get_player_probs(player).values())
            expected = cs.player_constraints[player].tiles_remaining
            assert total == pytest.approx(expected, abs=0.5)

    def test_exact_invariants_late_game(self) -> None:
        """Exact on late game satisfies all strict invariants."""
        cs = _make_late_game()
        pt = exact_marginals(cs)

        # All probs in [0, 1].
        assert np.all(pt.probs >= 0.0)
        assert np.all(pt.probs <= 1.0)

        # Per-tile sum = 1.
        for tile in pt.tiles:
            total = sum(pt.get_prob(p, tile) for p in pt.players)
            assert total == pytest.approx(1.0, abs=1e-10)

        # Per-player sum = tiles_remaining.
        for player in pt.players:
            total = sum(pt.get_player_probs(player).values())
            expected = cs.player_constraints[player].tiles_remaining
            assert total == pytest.approx(expected, abs=1e-10)

        # Non-candidate tiles have P = 0.
        for player in pt.players:
            cands = cs.get_candidates(player)
            for tile in pt.tiles:
                if tile not in cands:
                    assert pt.get_prob(player, tile) == pytest.approx(0.0)
