"""Integration engine bridging GameState and ConstraintSet.

Provides ``OracleState``, a unified frozen dataclass that combines the game
state machine, constraint propagation, and probability inference into a
single pipeline. Each action produces a new OracleState with updated
constraints and optionally recomputed probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from domino_oracle.core.constraints import OPPONENTS, ConstraintSet
from domino_oracle.core.game_state import Action, GameState, Pass, Play, Player
from domino_oracle.core.inference import ProbabilityTable, auto_marginals
from domino_oracle.core.tiles import Tile


@dataclass(frozen=True)
class OracleState:
    """Unified game + inference state.

    Combines a ``GameState`` (turn tracking, open ends, history) with a
    ``ConstraintSet`` (per-player candidate tiles) and an optional
    ``ProbabilityTable`` (marginal probabilities for each player-tile pair).

    Immutable — every mutation returns a new instance.

    Attributes:
        game: The current game state machine snapshot.
        constraints: The current constraint propagation state.
        probabilities: Computed marginal probabilities, or None if not
            yet computed for this state.
    """

    game: GameState
    constraints: ConstraintSet
    probabilities: ProbabilityTable | None

    @classmethod
    def initial(cls, my_hand: frozenset[Tile]) -> OracleState:
        """Create the initial oracle state at the start of a game.

        Args:
            my_hand: The 7 tiles dealt to South (you).

        Returns:
            A fresh OracleState with no plays, uniform priors.

        Raises:
            ValueError: If my_hand is invalid (wrong size or invalid tiles).
        """
        return cls(
            game=GameState.initial(my_hand),
            constraints=ConstraintSet.initial(my_hand),
            probabilities=None,
        )

    def apply_action(self, action: Action) -> OracleState:
        """Apply a play or pass and return the updated state.

        Translates the action into both a GameState transition and a
        ConstraintSet update. Probabilities are cleared (set to None)
        since the state has changed.

        Args:
            action: A Play or Pass action.

        Returns:
            A new OracleState with updated game and constraints,
            probabilities set to None.

        Raises:
            ValueError: If the action is illegal in the current game state.
            AssertionError: If a Pass occurs before any tiles are played
                (open_ends would be None).
        """
        new_game = self.game.apply_action(action)

        if isinstance(action, Play):
            new_constraints = self.constraints.apply_play(action.player, action.tile)
        elif action.player == Player.SOUTH:
            # South's hand is known; no constraint update needed for a pass.
            new_constraints = self.constraints
        else:
            # Pass: need the open ends from the *pre-action* game state.
            assert (
                self.game.open_ends is not None
            ), "Pass requires open ends on the board"
            new_constraints = self.constraints.apply_pass(
                action.player, self.game.open_ends
            )

        return OracleState(
            game=new_game,
            constraints=new_constraints,
            probabilities=None,
        )

    def compute_probabilities(
        self,
        *,
        mc_samples: int = 10_000,
        rng_seed: int | None = None,
    ) -> OracleState:
        """Compute marginal probabilities for the current state.

        Args:
            mc_samples: Number of Monte Carlo samples if sampling is used.
            rng_seed: Optional seed for reproducibility.

        Returns:
            A new OracleState with the probabilities field populated.
        """
        probs = auto_marginals(
            self.constraints,
            mc_samples=mc_samples,
            rng_seed=rng_seed,
        )
        return OracleState(
            game=self.game,
            constraints=self.constraints,
            probabilities=probs,
        )

    def verify_consistency(self) -> None:
        """Assert that game state and constraints are consistent.

        Checks:
        - Both agree on played tiles.
        - Both agree on South's hand.
        - Both agree on the set of unknown tiles.
        - Per-player tiles_remaining match between game and constraints.
        - Each player's candidate set is a subset of the unknown tiles.

        Raises:
            AssertionError: If any consistency check fails.
        """
        # Played tiles must match.
        assert self.game.played_tiles == self.constraints.played_tiles, (
            f"Played tiles mismatch: game={self.game.played_tiles}, "
            f"constraints={self.constraints.played_tiles}"
        )

        # South's hand must match.
        assert self.game.my_hand == self.constraints.my_hand, (
            f"My hand mismatch: game={self.game.my_hand}, "
            f"constraints={self.constraints.my_hand}"
        )

        # Unknown tiles must match.
        assert (
            self.game.unknown_tiles() == self.constraints.unknown_tiles()
        ), "Unknown tiles mismatch"

        # Per-opponent tiles_remaining must match.
        for player in OPPONENTS:
            game_remaining = self.game.tiles_remaining[player.value]
            constraint_remaining = self.constraints.player_constraints[
                player
            ].tiles_remaining
            assert game_remaining == constraint_remaining, (
                f"tiles_remaining mismatch for {player.name}: "
                f"game={game_remaining}, constraints={constraint_remaining}"
            )

        # Each player's candidates must be a subset of unknown tiles.
        unknown = self.constraints.unknown_tiles()
        for player in OPPONENTS:
            candidates = self.constraints.get_candidates(player)
            assert candidates.issubset(unknown), (
                f"{player.name} has candidates not in unknown tiles: "
                f"{candidates - unknown}"
            )


def replay_game(
    my_hand: frozenset[Tile],
    actions: list[Action],
    *,
    mc_samples: int = 10_000,
    rng_seed: int | None = None,
) -> OracleState:
    """Replay a sequence of actions and return the final state with probabilities.

    Args:
        my_hand: The 7 tiles dealt to South.
        actions: Ordered list of Play/Pass actions.
        mc_samples: Number of Monte Carlo samples for probability computation.
        rng_seed: Optional seed for reproducibility.

    Returns:
        The final OracleState with probabilities computed.

    Raises:
        ValueError: If any action is illegal.
    """
    state = OracleState.initial(my_hand)
    for action in actions:
        state = state.apply_action(action)
    return state.compute_probabilities(mc_samples=mc_samples, rng_seed=rng_seed)
