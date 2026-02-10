## The Core Problem

In a standard double-six domino game (2v2), you have 28 tiles, 7 per player. You know your hand, so the problem is tracking the distribution of 21 unknown tiles across 3 players. The initial state space is $\binom{21}{7} \times \binom{14}{7} \approx 3.4 \times 10^9$ possible configurations — large but not intractable, especially because every action drastically prunes it.

## What Makes It Rich

There are three types of information signals you can extract from each play:

**Positive signals** — when a player places a tile, that tile is revealed and removed from the unknown set. Straightforward.

**Negative signals (passes)** — this is where it gets really powerful. When a player passes, you learn they hold *no tile* matching either open end. If the open ends are 3 and 5, you can eliminate every tile containing a 3 or 5 from their possible hand. These constraints propagate beautifully — a single pass can eliminate 10+ tiles from a player's candidate set.

**Strategic signals** — this is the subtlest layer. If a player *could* play on either end and chooses one, that choice carries information. Did they play the double? Did they avoid opening a suit? In competitive domino circles, these inferences are exactly what experienced players do intuitively. Modeling this layer is optional but is what would take the tool from useful to exceptional.

## Algorithmic Approach

I'd think of it in terms of three possible strategies, in increasing sophistication:

1. **Constraint-based enumeration with uniform prior**: maintain the set of feasible tile assignments consistent with all observations (plays and passes), then compute marginal probabilities by counting. This is the Monte Carlo approach — sample N valid configurations uniformly and estimate $P(\text{player } j \text{ has tile } t)$ from frequencies. Very practical and probably where I'd start.

2. **Exact Bayesian enumeration**: for a reduced state (as the game progresses and the unknown set shrinks), you can enumerate all valid configurations exactly. By mid-game, when 8-10 tiles have been played and several passes have occurred, the feasible set often collapses to hundreds or thousands of configurations — perfectly enumerable.

3. **Bayesian with strategic priors**: instead of assuming uniform play, model each player's decision as a function of their hand (e.g., players tend to play doubles early, block opponents' strong suits, feed their partner). This turns it into a particle filter or SMC (Sequential Monte Carlo) problem where you weight configurations by their likelihood given observed behavior.

## Data Structure Sketch

The core state you'd maintain at each step:

- **Known tiles**: your hand + all played tiles (with who played them)
- **Open ends**: the two ends of the chain
- **Per-player constraints**: sets of values they *cannot* have (from passes), tiles they *must not* have (already played by others), and count of remaining tiles in hand
- **Unknown pool**: tiles not yet accounted for

From this you can generate valid assignments via rejection sampling or constraint propagation (think CSP/arc consistency before sampling).

## Visualization Ideas

This is where it could really shine as an app:

- A **tile matrix** (7×7 upper triangle for the 28 dominoes) color-coded by probability for each player — you'd see the uncertainty shrink in real time
- A **per-player bar** showing the probability they hold each remaining unknown tile
- A **"what should I play?"** advisor mode that, given probabilities, suggests the move that maximizes your team's expected score or win probability
- Timeline view showing how certainty evolved across rounds

## My Honest Assessment

The constraint-based Monte Carlo version (approach 1) is very buildable — probably a weekend project for the core engine, with the visualization taking more time depending on how polished you want it. The state space is small enough that you don't need anything exotic. Python with NumPy for the engine, and either a React frontend or even an org-mode + matplotlib prototype for quick iteration.

The strategic inference layer (approach 3) is the research-grade version and could genuinely be a publishable contribution if done well, especially if you frame it as inverse planning or Theory of Mind in a partially observable game.

Want to start sketching the data model and sampling engine? I'm thinking a clean Python class that tracks game state and exposes `P(player, tile)` after each action would be the right foundation.
