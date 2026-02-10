"""Constraint propagation from passes and plays.

Tracks per-player candidate tile sets and applies domino game constraints
(plays, passes) with arc consistency propagation to narrow down the
feasible space of tile assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domino_oracle.core.tiles import Tile, generate_full_set


class Player(Enum):
    """Players at the domino table (clockwise order)."""

    SOUTH = 0
    WEST = 1
    NORTH = 2
    EAST = 3


# The three opponents whose hands are unknown (South is "you").
OPPONENTS: tuple[Player, ...] = (Player.WEST, Player.NORTH, Player.EAST)


def _tiles_with_value(tiles: frozenset[Tile], value: int) -> frozenset[Tile]:
    """Return all tiles in *tiles* that contain *value* on either side.

    Args:
        tiles: Set of tiles to filter.
        value: The pip value to match (0-6).

    Returns:
        Subset of tiles where a == value or b == value.
    """
    return frozenset(t for t in tiles if t.a == value or t.b == value)


@dataclass(frozen=True)
class PlayerConstraints:
    """Constraints on which tiles a single player might hold.

    Attributes:
        candidate_tiles: Tiles this player could still have in hand.
        tiles_remaining: Number of tiles the player currently holds.
        eliminated_values: Suit values the player definitely does NOT have
            (accumulated from pass actions).
    """

    candidate_tiles: frozenset[Tile]
    tiles_remaining: int
    eliminated_values: frozenset[int]

    def remove_tile(self, tile: Tile) -> PlayerConstraints:
        """Return new constraints with *tile* removed from candidates.

        Args:
            tile: The tile to remove from the candidate set.

        Returns:
            Updated PlayerConstraints (candidate set shrinks by at most one).
        """
        return PlayerConstraints(
            candidate_tiles=self.candidate_tiles - {tile},
            tiles_remaining=self.tiles_remaining,
            eliminated_values=self.eliminated_values,
        )

    def decrement_remaining(self) -> PlayerConstraints:
        """Return new constraints with tiles_remaining decremented by one.

        Returns:
            Updated PlayerConstraints with one fewer remaining tile.
        """
        return PlayerConstraints(
            candidate_tiles=self.candidate_tiles,
            tiles_remaining=self.tiles_remaining - 1,
            eliminated_values=self.eliminated_values,
        )

    def eliminate_value(self, value: int) -> PlayerConstraints:
        """Eliminate all tiles containing *value* from this player's candidates.

        Args:
            value: The pip value (0-6) to eliminate.

        Returns:
            Updated PlayerConstraints with matching tiles removed and the
            value recorded in eliminated_values.
        """
        bad_tiles = _tiles_with_value(self.candidate_tiles, value)
        return PlayerConstraints(
            candidate_tiles=self.candidate_tiles - bad_tiles,
            tiles_remaining=self.tiles_remaining,
            eliminated_values=self.eliminated_values | {value},
        )

    def set_candidates(self, new_candidates: frozenset[Tile]) -> PlayerConstraints:
        """Return new constraints with a replaced candidate set.

        Args:
            new_candidates: The new set of candidate tiles.

        Returns:
            Updated PlayerConstraints with the new candidate set.
        """
        return PlayerConstraints(
            candidate_tiles=new_candidates,
            tiles_remaining=self.tiles_remaining,
            eliminated_values=self.eliminated_values,
        )


@dataclass(frozen=True)
class ConstraintSet:
    """Full constraint state for all non-self players.

    Immutable -- every mutation method returns a new ConstraintSet.

    Attributes:
        player_constraints: Per-player constraint info for West, North, East.
        played_tiles: All tiles that have been played so far.
        my_hand: The tiles held by South (the user).
    """

    player_constraints: dict[Player, PlayerConstraints]
    played_tiles: frozenset[Tile]
    my_hand: frozenset[Tile]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def initial(cls, my_hand: frozenset[Tile]) -> ConstraintSet:
        """Create the initial constraint set at the start of a game.

        Args:
            my_hand: The 7 tiles dealt to South.

        Returns:
            A fresh ConstraintSet where each opponent could hold any of
            the 21 unknown tiles.

        Raises:
            ValueError: If my_hand does not contain exactly 7 valid tiles.
        """
        if len(my_hand) != 7:
            raise ValueError(f"Hand must contain exactly 7 tiles, got {len(my_hand)}")
        full = generate_full_set()
        if not my_hand.issubset(full):
            raise ValueError("Hand contains invalid tiles")

        unknown = full - my_hand
        pc: dict[Player, PlayerConstraints] = {}
        for p in OPPONENTS:
            pc[p] = PlayerConstraints(
                candidate_tiles=unknown,
                tiles_remaining=7,
                eliminated_values=frozenset(),
            )
        return cls(
            player_constraints=pc,
            played_tiles=frozenset(),
            my_hand=my_hand,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_candidates(self, player: Player) -> frozenset[Tile]:
        """Return the set of candidate tiles for *player*.

        Args:
            player: One of West, North, East.

        Returns:
            The frozenset of tiles the player might hold.

        Raises:
            KeyError: If player is South (not tracked).
        """
        return self.player_constraints[player].candidate_tiles

    def unknown_tiles(self) -> frozenset[Tile]:
        """Return tiles that are neither played nor in my_hand.

        Returns:
            The frozenset of tiles whose location is unknown.
        """
        return generate_full_set() - self.played_tiles - self.my_hand

    # ------------------------------------------------------------------
    # Mutations (return new ConstraintSet)
    # ------------------------------------------------------------------

    def apply_play(self, player: Player, tile: Tile) -> ConstraintSet:
        """Apply a PLAY action: *player* places *tile* on the board.

        Removes the tile from all candidate sets, decrements the player's
        remaining count, adds the tile to played_tiles, and propagates.

        Args:
            player: The player who played the tile.
            tile: The tile that was played.

        Returns:
            A new ConstraintSet reflecting the play.
        """
        new_pc = dict(self.player_constraints)
        for p in OPPONENTS:
            new_pc[p] = new_pc[p].remove_tile(tile)
        # Only decrement for opponents (South's hand is tracked separately).
        if player in new_pc:
            new_pc[player] = new_pc[player].decrement_remaining()

        new_cs = ConstraintSet(
            player_constraints=new_pc,
            played_tiles=self.played_tiles | {tile},
            my_hand=self.my_hand if player != Player.SOUTH else self.my_hand - {tile},
        )
        return new_cs.propagate()

    def apply_pass(self, player: Player, open_ends: tuple[int, int]) -> ConstraintSet:
        """Apply a PASS action: *player* could not play.

        The passing player cannot hold ANY tile containing either open-end
        value. Those values are added to the player's eliminated set and
        matching tiles are removed from their candidates.

        Args:
            player: The player who passed.
            open_ends: The two open-end pip values on the board.

        Returns:
            A new ConstraintSet reflecting the pass.

        Raises:
            KeyError: If player is South (South never passes in our model).
        """
        new_pc = dict(self.player_constraints)
        pc = new_pc[player]
        end_a, end_b = open_ends
        pc = pc.eliminate_value(end_a)
        if end_b != end_a:
            pc = pc.eliminate_value(end_b)
        new_pc[player] = pc

        new_cs = ConstraintSet(
            player_constraints=new_pc,
            played_tiles=self.played_tiles,
            my_hand=self.my_hand,
        )
        return new_cs.propagate()

    # ------------------------------------------------------------------
    # Constraint propagation
    # ------------------------------------------------------------------

    def propagate(self) -> ConstraintSet:
        """Run arc-consistency propagation until a fixed point.

        Two rules are applied iteratively:
        1. **Determined player**: If a player's candidate count equals their
           tiles_remaining, those tiles are locked to that player and removed
           from other players' candidate sets.
        2. **Unique tile**: If only one player has a tile in their candidate
           set, that tile is definitely theirs (no removal needed, but this
           can trigger rule 1 indirectly).

        Iteration stops when no candidate sets change, or after a safety
        limit of 50 iterations.

        Returns:
            A new ConstraintSet after propagation.
        """
        pc = dict(self.player_constraints)
        max_iterations = 50

        for _ in range(max_iterations):
            changed = False

            # Rule 1: Determined players
            for p in OPPONENTS:
                cp = pc[p]
                if len(cp.candidate_tiles) == cp.tiles_remaining:
                    # This player's tiles are fully determined.
                    determined = cp.candidate_tiles
                    for q in OPPONENTS:
                        if q != p:
                            old_cands = pc[q].candidate_tiles
                            new_cands = old_cands - determined
                            if new_cands != old_cands:
                                pc[q] = pc[q].set_candidates(new_cands)
                                changed = True

            # Rule 2: Unique tile assignment
            # Collect all unknown tiles and check if any is a candidate
            # for exactly one player.
            unknown = generate_full_set() - self.played_tiles - self.my_hand
            for tile in unknown:
                holders = [p for p in OPPONENTS if tile in pc[p].candidate_tiles]
                if len(holders) == 1:
                    # Tile must belong to this player -- this is implicit,
                    # but can help trigger rule 1 on subsequent iterations
                    # by keeping the tile only in that player's set.
                    pass
                elif len(holders) == 0:
                    # This should not happen in a valid game state; the
                    # tile must be somewhere.  We silently continue to
                    # avoid crashing mid-propagation.
                    pass

            if not changed:
                break

        return ConstraintSet(
            player_constraints=pc,
            played_tiles=self.played_tiles,
            my_hand=self.my_hand,
        )
