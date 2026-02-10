"""Game state machine and turn tracking.

Immutable game state for 2v2 double-six dominoes. Each action (play or pass)
produces a new GameState, enabling undo/replay functionality.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from domino_oracle.core.tiles import Tile, generate_full_set


class Player(Enum):
    """The four players in clockwise seating order.

    South is always "you" (the human). North is your partner.
    """

    SOUTH = 0
    WEST = 1
    NORTH = 2
    EAST = 3


class Team(Enum):
    """The two teams in a 2v2 game."""

    NS = 0  # North-South (you + partner)
    WE = 1  # West-East (opponents)


@dataclass(frozen=True)
class Play:
    """A player places a tile on one end of the domino chain.

    Attributes:
        player: The player making the play.
        tile: The tile being placed.
        end: The open-end value being matched. The tile must contain
            this value.
    """

    player: Player
    tile: Tile
    end: int


@dataclass(frozen=True)
class Pass:
    """A player cannot play and passes their turn.

    Attributes:
        player: The player who passes.
    """

    player: Player


Action = Play | Pass


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of a domino game in progress.

    Each action produces a new GameState via ``apply_action``. This
    enables full undo/replay and makes the state safe to share across
    threads or store in history.

    Attributes:
        all_tiles: The full set of 28 domino tiles.
        my_hand: South's (your) current hand.
        history: Ordered tuple of all actions taken so far.
        open_ends: The two open ends of the domino chain, or None
            before the first play.
        played_tiles: Set of tiles that have been placed on the chain.
        current_player: Whose turn it is to act.
        tiles_remaining: How many tiles each player still holds,
            indexed by ``Player.value``.
    """

    all_tiles: frozenset[Tile]
    my_hand: frozenset[Tile]
    history: tuple[Action, ...]
    open_ends: tuple[int, int] | None
    played_tiles: frozenset[Tile]
    current_player: Player
    tiles_remaining: tuple[int, int, int, int]

    @classmethod
    def initial(cls, my_hand: frozenset[Tile]) -> GameState:
        """Create a new game state at the start of a round.

        Args:
            my_hand: The 7 tiles dealt to South (you).

        Returns:
            A fresh GameState with all tiles unplayed, current player
            set to South, and each player holding 7 tiles.

        Raises:
            ValueError: If ``my_hand`` does not contain exactly 7 tiles
                or contains tiles not in the standard double-six set.
        """
        full_set = generate_full_set()
        if len(my_hand) != 7:
            raise ValueError(f"Hand must contain exactly 7 tiles, got {len(my_hand)}.")
        if not my_hand.issubset(full_set):
            raise ValueError("Hand contains tiles not in the standard domino set.")
        return cls(
            all_tiles=full_set,
            my_hand=my_hand,
            history=(),
            open_ends=None,
            played_tiles=frozenset(),
            current_player=Player.SOUTH,
            tiles_remaining=(7, 7, 7, 7),
        )

    def apply_action(self, action: Action) -> GameState:
        """Apply an action and return the resulting new game state.

        For a ``Play``, the tile is validated against the current open
        ends, removed from the available tiles, and the open ends are
        updated. For a ``Pass``, the board state is unchanged but the
        turn advances.

        Args:
            action: The Play or Pass to apply.

        Returns:
            A new GameState reflecting the action.

        Raises:
            ValueError: If the action is illegal (wrong player, invalid
                tile placement, pass on first turn, etc.).
        """
        if action.player != self.current_player:
            raise ValueError(
                f"It is {self.current_player.name}'s turn, "
                f"not {action.player.name}'s."
            )

        if isinstance(action, Play):
            return self._apply_play(action)
        # Type narrowing: action must be Pass at this point.
        return self._apply_pass(action)

    def _apply_play(self, play: Play) -> GameState:
        """Apply a Play action (internal).

        Args:
            play: The play action to apply.

        Returns:
            A new GameState with the play applied.

        Raises:
            ValueError: If the play is illegal.
        """
        tile = play.tile
        end = play.end

        if not tile.contains_value(end):
            raise ValueError(
                f"Tile {tile} does not contain the declared end value {end}."
            )

        if tile in self.played_tiles:
            raise ValueError(f"Tile {tile} has already been played.")

        if self.open_ends is None:
            # First play of the game: both open ends come from the tile.
            new_open_ends = tile.values()
        else:
            if end not in self.open_ends:
                raise ValueError(
                    f"End value {end} does not match either open end "
                    f"{self.open_ends}."
                )
            new_value = tile.other_value(end)
            left, right = self.open_ends
            if left == end:
                new_open_ends = (new_value, right)
            else:
                new_open_ends = (left, new_value)

        new_played = self.played_tiles | {tile}
        new_hand = self.my_hand
        if play.player == Player.SOUTH:
            new_hand = self.my_hand - {tile}

        remaining = list(self.tiles_remaining)
        remaining[play.player.value] -= 1
        new_remaining = (remaining[0], remaining[1], remaining[2], remaining[3])

        return GameState(
            all_tiles=self.all_tiles,
            my_hand=new_hand,
            history=self.history + (play,),
            open_ends=new_open_ends,
            played_tiles=new_played,
            current_player=self.next_player(self.current_player),
            tiles_remaining=new_remaining,
        )

    def _apply_pass(self, pass_action: Pass) -> GameState:
        """Apply a Pass action (internal).

        Args:
            pass_action: The pass action to apply.

        Returns:
            A new GameState with the turn advanced.

        Raises:
            ValueError: If passing on the first turn (no open ends yet).
        """
        if self.open_ends is None:
            raise ValueError("Cannot pass on the first turn (no tiles played yet).")

        return GameState(
            all_tiles=self.all_tiles,
            my_hand=self.my_hand,
            history=self.history + (pass_action,),
            open_ends=self.open_ends,
            played_tiles=self.played_tiles,
            current_player=self.next_player(self.current_player),
            tiles_remaining=self.tiles_remaining,
        )

    @staticmethod
    def next_player(player: Player) -> Player:
        """Return the next player in clockwise order.

        Turn order: South -> West -> North -> East -> South.

        Args:
            player: The current player.

        Returns:
            The next Player in the rotation.
        """
        return Player((player.value + 1) % 4)

    def unknown_tiles(self) -> frozenset[Tile]:
        """Return tiles not in your hand and not yet played.

        These are the tiles distributed among the other three players
        whose locations are uncertain.

        Returns:
            A frozenset of tiles whose holder is unknown.
        """
        return self.all_tiles - self.my_hand - self.played_tiles

    def is_game_over(self) -> bool:
        """Determine whether the game has ended.

        The game is over if any player has 0 tiles remaining, or if
        the last 4 consecutive actions were all passes (a lock).

        Returns:
            True if the game is over.
        """
        # Any player ran out of tiles.
        if any(count == 0 for count in self.tiles_remaining):
            return True

        # Four consecutive passes (locked board).
        if len(self.history) >= 4 and all(
            isinstance(action, Pass) for action in self.history[-4:]
        ):
            return True

        return False

    @staticmethod
    def player_team(player: Player) -> Team:
        """Return the team a player belongs to.

        South and North are team NS; West and East are team WE.

        Args:
            player: The player to look up.

        Returns:
            The player's Team.
        """
        if player in (Player.SOUTH, Player.NORTH):
            return Team.NS
        return Team.WE

    def get_open_end_values(self) -> frozenset[int]:
        """Return the set of unique values among the open ends.

        Returns:
            A frozenset with 0 elements (if no tiles played), 1 element
            (if both ends are the same value), or 2 elements.
        """
        if self.open_ends is None:
            return frozenset()
        return frozenset(self.open_ends)
