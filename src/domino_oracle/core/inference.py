"""Monte Carlo sampler and exact enumeration for probability inference.

Given a ConstraintSet describing what is known about each player's hand,
this module computes marginal probabilities P(player j has tile t) for
every unknown tile and every opponent player.

Two estimation modes are provided:
- **Monte Carlo**: Fast rejection sampling for early-game states with many
  unknowns.
- **Exact enumeration**: Brute-force enumeration of all valid
  configurations for late-game states with few unknowns.

A convenience function ``auto_marginals`` selects the appropriate method
based on the number of unknown tiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from domino_oracle.core.constraints import (
    OPPONENTS,
    ConstraintSet,
    Player,
)
from domino_oracle.core.tiles import Tile

# Threshold: use exact enumeration when unknown tiles <= this value.
_EXACT_THRESHOLD = 15


@dataclass
class ProbabilityTable:
    """Marginal probabilities P(player has tile) for all player-tile pairs.

    Attributes:
        tiles: Ordered list of unknown tiles (row axis).
        players: List of opponent players [WEST, NORTH, EAST] (column axis).
        probs: 2-D array of shape ``(len(players), len(tiles))``.
    """

    tiles: list[Tile]
    players: list[Player]
    probs: NDArray[np.float64]

    # Lookup helpers built on first access.
    _tile_index: dict[Tile, int] = field(
        default_factory=lambda: dict[Tile, int](), init=False, repr=False
    )
    _player_index: dict[Player, int] = field(
        default_factory=lambda: dict[Player, int](), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._tile_index = {t: i for i, t in enumerate(self.tiles)}
        self._player_index = {p: i for i, p in enumerate(self.players)}

    def get_prob(self, player: Player, tile: Tile) -> float:
        """Return P(*player* has *tile*).

        Args:
            player: One of West, North, East.
            tile: An unknown tile.

        Returns:
            The marginal probability as a float in [0, 1].

        Raises:
            KeyError: If player or tile is not in this table.
        """
        pi = self._player_index[player]
        ti = self._tile_index[tile]
        return float(self.probs[pi, ti])

    def get_player_probs(self, player: Player) -> dict[Tile, float]:
        """Return all tile probabilities for *player*.

        Args:
            player: One of West, North, East.

        Returns:
            Dict mapping each unknown tile to its probability for this player.
        """
        pi = self._player_index[player]
        return {t: float(self.probs[pi, i]) for i, t in enumerate(self.tiles)}

    def get_tile_probs(self, tile: Tile) -> dict[Player, float]:
        """Return all player probabilities for *tile*.

        Args:
            tile: An unknown tile.

        Returns:
            Dict mapping each opponent player to the probability they hold
            this tile.
        """
        ti = self._tile_index[tile]
        return {p: float(self.probs[i, ti]) for i, p in enumerate(self.players)}


def _build_probability_table(
    tiles: list[Tile],
    players: list[Player],
    counts: NDArray[np.float64],
    total: int,
) -> ProbabilityTable:
    """Normalise raw counts into a ProbabilityTable.

    Args:
        tiles: Ordered unknown tiles.
        players: Ordered opponent players.
        counts: Array of shape ``(len(players), len(tiles))`` holding the
            number of valid configurations where player *i* holds tile *j*.
        total: Total number of valid configurations counted.

    Returns:
        A ProbabilityTable with normalised probabilities.

    Raises:
        ValueError: If no valid configurations were found.
    """
    if total == 0:
        raise ValueError(
            "No valid configurations found; constraints may be inconsistent"
        )
    probs = counts / total
    return ProbabilityTable(tiles=tiles, players=players, probs=probs)


# ------------------------------------------------------------------
# Monte Carlo sampling
# ------------------------------------------------------------------


def monte_carlo_marginals(
    constraints: ConstraintSet,
    n_samples: int = 10_000,
    rng_seed: int | None = None,
) -> ProbabilityTable:
    """Estimate marginal probabilities via rejection sampling.

    Randomly assigns unknown tiles to opponents respecting their
    ``tiles_remaining`` counts, then keeps only those assignments that
    are consistent with each player's candidate set.

    Args:
        constraints: Current game constraint state.
        n_samples: Number of *attempts* (not accepted samples).
        rng_seed: Optional seed for reproducibility.

    Returns:
        A ProbabilityTable of estimated marginal probabilities.

    Raises:
        ValueError: If no valid sample is accepted after all attempts.
    """
    rng = np.random.default_rng(rng_seed)

    players = list(OPPONENTS)
    unknown_set = constraints.unknown_tiles()
    unknown_list = sorted(unknown_set)  # deterministic ordering
    n_tiles = len(unknown_list)
    n_players = len(players)

    tile_index = {t: i for i, t in enumerate(unknown_list)}

    # Pre-compute per-player candidate masks (boolean arrays).
    remaining = [constraints.player_constraints[p].tiles_remaining for p in players]
    candidate_masks: list[set[int]] = []
    for p in players:
        cands = constraints.get_candidates(p)
        candidate_masks.append({tile_index[t] for t in cands if t in tile_index})

    counts = np.zeros((n_players, n_tiles), dtype=np.float64)
    accepted = 0

    # Build an index array for shuffling.
    indices = np.arange(n_tiles)

    for _ in range(n_samples):
        rng.shuffle(indices)

        # Split shuffled indices according to remaining counts.
        assignment: list[list[int]] = []
        offset = 0
        valid = True
        for pi, r in enumerate(remaining):
            player_tiles = indices[offset : offset + r].tolist()
            # Check all assigned tiles are in the player's candidate set.
            if not all(ti in candidate_masks[pi] for ti in player_tiles):
                valid = False
                break
            assignment.append(player_tiles)
            offset += r

        if not valid:
            continue

        # Check total assignment covers all tiles.
        if offset != n_tiles:
            continue

        accepted += 1
        for pi, player_tiles in enumerate(assignment):
            for ti in player_tiles:
                counts[pi, ti] += 1.0

    return _build_probability_table(unknown_list, players, counts, accepted)


# ------------------------------------------------------------------
# Exact enumeration
# ------------------------------------------------------------------


def exact_marginals(constraints: ConstraintSet) -> ProbabilityTable:
    """Compute exact marginal probabilities by full enumeration.

    Iterates over all valid assignments of unknown tiles to opponents and
    computes exact marginals.

    Args:
        constraints: Current game constraint state.

    Returns:
        A ProbabilityTable of exact marginal probabilities.

    Raises:
        ValueError: If the number of unknown tiles exceeds 18 (use Monte
            Carlo instead) or if no valid configuration exists.
    """
    unknown_set = constraints.unknown_tiles()
    if len(unknown_set) > 18:
        raise ValueError(
            f"Too many unknown tiles ({len(unknown_set)}) for exact "
            f"enumeration. Use monte_carlo_marginals instead."
        )

    players = list(OPPONENTS)
    unknown_list = sorted(unknown_set)
    n_tiles = len(unknown_list)
    n_players = len(players)

    tile_index = {t: i for i, t in enumerate(unknown_list)}

    # Per-player: set of tile indices that are valid candidates.
    remaining = [constraints.player_constraints[p].tiles_remaining for p in players]
    candidate_indices: list[list[int]] = []
    for p in players:
        cands = constraints.get_candidates(p)
        candidate_indices.append(
            sorted(tile_index[t] for t in cands if t in tile_index)
        )

    counts = np.zeros((n_players, n_tiles), dtype=np.float64)
    total = 0

    # Enumerate: choose tiles for player 0, then player 1, then player 2.
    # Player 0 picks `remaining[0]` tiles from their candidates.
    for combo_0 in combinations(candidate_indices[0], remaining[0]):
        set_0 = set(combo_0)
        # Player 1 picks from their candidates minus what player 0 took.
        avail_1 = [ti for ti in candidate_indices[1] if ti not in set_0]
        if len(avail_1) < remaining[1]:
            continue
        for combo_1 in combinations(avail_1, remaining[1]):
            set_1 = set(combo_1)
            # Player 2 gets whatever is left; check they are all candidates.
            leftover = [
                ti for ti in range(n_tiles) if ti not in set_0 and ti not in set_1
            ]
            if len(leftover) != remaining[2]:
                continue
            # Verify all leftover tiles are in player 2's candidate set.
            cand_2_set = set(candidate_indices[2])
            if not all(ti in cand_2_set for ti in leftover):
                continue

            # Valid configuration found.
            total += 1
            for ti in combo_0:
                counts[0, ti] += 1.0
            for ti in combo_1:
                counts[1, ti] += 1.0
            for ti in leftover:
                counts[2, ti] += 1.0

    return _build_probability_table(unknown_list, players, counts, total)


# ------------------------------------------------------------------
# Auto-dispatch
# ------------------------------------------------------------------


def auto_marginals(
    constraints: ConstraintSet,
    mc_samples: int = 10_000,
    rng_seed: int | None = None,
) -> ProbabilityTable:
    """Compute marginals using the best available method.

    Dispatches to exact enumeration when the number of unknown tiles is
    small enough, otherwise falls back to Monte Carlo sampling.

    Args:
        constraints: Current game constraint state.
        mc_samples: Number of Monte Carlo attempts if sampling is used.
        rng_seed: Optional seed for reproducibility.

    Returns:
        A ProbabilityTable of marginal probabilities.
    """
    n_unknown = len(constraints.unknown_tiles())
    if n_unknown <= _EXACT_THRESHOLD:
        return exact_marginals(constraints)
    return monte_carlo_marginals(constraints, n_samples=mc_samples, rng_seed=rng_seed)
