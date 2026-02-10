# Mathematical Formulation for Bayesian Domino Inference

This document presents the formal probability model underlying the Domino Oracle
system. It covers the state space, observation model, posterior computation,
constraint propagation, and both Monte Carlo and exact enumeration algorithms for
computing marginal tile-holding probabilities in a 2v2 double-six domino game.

**Prerequisites.** Familiarity with basic probability, combinatorics, and Bayes'
theorem at the undergraduate level.

---

## Table of Contents

1. [Problem Setup](#1-problem-setup)
2. [State Space](#2-state-space)
3. [Observations and Likelihood](#3-observations-and-likelihood)
4. [Posterior Distribution](#4-posterior-distribution)
5. [Marginal Probabilities](#5-marginal-probabilities)
6. [Constraint Propagation](#6-constraint-propagation)
7. [Monte Carlo Sampling](#7-monte-carlo-sampling)
8. [Exact Enumeration](#8-exact-enumeration)
9. [Worked Example](#9-worked-example)

---

## 1. Problem Setup

### 1.1 The Tile Set

A **double-six domino set** consists of 28 tiles. Each tile is an unordered pair
of values $(a, b)$ where $0 \le a \le b \le 6$. We adopt the **canonical form**
$(a, b)$ with $a \le b$ to avoid representing the same physical tile twice.

$$
\mathcal{T} = \{ (a, b) \mid 0 \le a \le b \le 6 \}
$$

$$
|\mathcal{T}| = \binom{7}{2} + 7 = 21 + 7 = 28
$$

The 7 tiles where $a = b$ are called **doubles**: $(0{|}0), (1{|}1), \ldots,
(6{|}6)$.

For any tile $t = (a, b)$, we define its **values** as the set of pips it
contains:

$$
\text{values}(t) = \{a, b\}
$$

We say tile $t$ **contains value** $v$ if $v \in \text{values}(t)$. A tile
belongs to **suit** $v$ if it contains value $v$. Each suit has 7 members (value
$v$ pairs with each of $0, 1, \ldots, 6$), and each non-double tile belongs to
exactly two suits.

### 1.2 Players and Teams

Four players sit in fixed positions around the table and play in clockwise order:

$$
\mathcal{P}_{\text{all}} = \{ S, W, N, E \} \quad \text{(South, West, North, East)}
$$

Teams are fixed:

- **Team 1:** $(S, N)$ -- South and North (partners)
- **Team 2:** $(W, E)$ -- West and East (partners)

The turn order is $S \to W \to N \to E \to S \to \cdots$

### 1.3 The Deal

At the start of the game, the 28 tiles are dealt uniformly at random, 7 to each
player. The observer is South ($S$), who knows their own hand. We define:

- $H \subset \mathcal{T}$ : South's hand, with $|H| = 7$
- $U = \mathcal{T} \setminus H$ : the set of **unknown tiles**, with $|U| = 21$
- $\mathcal{P} = \{W, N, E\}$ : the three players whose hands are unknown

Each unknown player $p \in \mathcal{P}$ holds exactly 7 tiles from $U$, forming
a partition of $U$ into three disjoint sets of size 7.

### 1.4 Game Mechanics

The game proceeds in rounds. On each turn, the active player either:

1. **Plays** a tile from their hand onto one end of the chain, or
2. **Passes** if they hold no tile that matches either open end of the chain.

The chain has two **open ends**, each showing a pip value. A tile can be played
on an end if it contains the value shown at that end.

---

## 2. State Space

### 2.1 Configurations

A **configuration** is an assignment of each unknown tile to one of the three
unknown players:

$$
\sigma : U \to \mathcal{P}
$$

The assignment must satisfy the **hand-size constraint**: each player holds
exactly as many tiles as they have remaining (initially 7, decremented each time
they play):

$$
|\sigma^{-1}(p)| = r(p) \quad \forall\, p \in \mathcal{P}
$$

where $r(p)$ denotes the **remaining tile count** for player $p$.

### 2.2 Initial State Space

Before any observations, the set of all valid configurations is:

$$
\Omega_0 = \left\{ \sigma : U \to \mathcal{P} \;\middle|\; |\sigma^{-1}(p)| = 7 \;\; \forall\, p \in \mathcal{P} \right\}
$$

The size of this initial state space is the multinomial coefficient:

$$
|\Omega_0| = \binom{21}{7, 7, 7} = \frac{21!}{7!\, 7!\, 7!} \approx 3.4 \times 10^9
$$

This is large but finite. As the game progresses, constraints dramatically
reduce the feasible set.

### 2.3 Prior Distribution

Tiles are dealt uniformly at random, so the prior distribution over
configurations is uniform:

$$
P(\sigma) = \frac{1}{|\Omega_0|} \quad \forall\, \sigma \in \Omega_0
$$

---

## 3. Observations and Likelihood

During the game, we observe a sequence of actions $O_1, O_2, \ldots, O_k$. Each
observation is one of two types.

### 3.1 Play Observation

$$
\text{PLAY}(p, t, e)
$$

Player $p$ places tile $t$ on end $e$ of the chain.

**Implications for the state:**

1. **Tile revealed:** $t$ is now known to belong to $p$. Remove $t$ from $U$.
2. **Count update:** $r(p) \leftarrow r(p) - 1$.
3. **Open ends update:** The end where $t$ was played now shows the other value
   of $t$.

**Constraint on configurations:** Any valid configuration must have assigned $t$
to $p$ before this action. After the action, $t$ is no longer in $U$ and need
not be tracked.

Formally, the indicator function for consistency is:

$$
\mathbb{1}[\sigma \text{ consistent with } \text{PLAY}(p, t, e)] =
\begin{cases}
1 & \text{if } \sigma(t) = p \\
0 & \text{otherwise}
\end{cases}
$$

### 3.2 Pass Observation

$$
\text{PASS}(p, (a, b))
$$

Player $p$ cannot play when the open ends show values $a$ and $b$.

**Implication:** Player $p$ holds **no tile** whose values include $a$ or $b$.
If $a = b$ (both ends show the same value), then $p$ holds no tile containing
$a$.

Formally, define the set of tiles blocked by open ends $(a, b)$:

$$
B(a, b) = \{ t \in U \mid a \in \text{values}(t) \;\lor\; b \in \text{values}(t) \}
$$

The constraint is:

$$
\sigma^{-1}(p) \cap B(a, b) = \emptyset
$$

Or equivalently:

$$
\mathbb{1}[\sigma \text{ consistent with } \text{PASS}(p, (a,b))] =
\prod_{t \in B(a,b)} \mathbb{1}[\sigma(t) \ne p]
$$

**Note on the power of passes.** A single pass can eliminate many tiles. If both
open ends are distinct, the blocked set $B(a, b)$ includes all tiles containing
$a$ plus all tiles containing $b$. Each suit has up to 7 tiles, so with two
distinct values, up to 13 tiles can be eliminated (7 + 7 - 1, since the tile
$(a, b)$ itself is counted once). After subtracting tiles already played or in
South's hand, a pass can still easily eliminate 8--10 candidates.

### 3.3 Accumulated Constraints

Let $\mathcal{C}$ denote the set of all constraints accumulated from
observations $O_1, \ldots, O_k$. The **feasible configuration set** after all
observations is:

$$
\Omega_{\mathcal{C}} = \left\{ \sigma \in \Omega_0 \;\middle|\;
\sigma \text{ is consistent with every } O_i, \; i = 1, \ldots, k \right\}
$$

---

## 4. Posterior Distribution

### 4.1 Bayesian Update

Starting from the uniform prior over $\Omega_0$, the posterior probability of a
configuration $\sigma$ given all observations is obtained via Bayes' theorem:

$$
P(\sigma \mid O_1, \ldots, O_k) \propto P(\sigma) \cdot \prod_{i=1}^{k}
\mathbb{1}[\sigma \text{ consistent with } O_i]
$$

Since the prior is uniform and the likelihood is either 0 or 1 (a configuration
is either consistent or not), the posterior is simply the **uniform distribution
over the feasible set**:

$$
\boxed{
P(\sigma \mid O_1, \ldots, O_k) =
\begin{cases}
\dfrac{1}{|\Omega_{\mathcal{C}}|} & \text{if } \sigma \in \Omega_{\mathcal{C}} \\[8pt]
0 & \text{otherwise}
\end{cases}
}
$$

### 4.2 Intuition

This result has an elegant interpretation: under a uniform deal, every
configuration that is compatible with what we have observed is equally likely.
There is no need for complex likelihood weighting -- the problem reduces to
**counting valid configurations**.

### 4.3 Extension: Strategic Priors

In an advanced version (not implemented in Phase 1), one could assign
non-uniform likelihoods to player actions based on strategic behavior. For
instance, a player might preferentially play doubles early or avoid opening a
suit their partner is blocking. In this case:

$$
P(\sigma \mid O_1, \ldots, O_k) \propto P(\sigma) \cdot \prod_{i=1}^{k}
P(O_i \mid \sigma, \text{history})
$$

where $P(O_i \mid \sigma, \text{history})$ is a **behavioral model** of player
decision-making. This transforms the problem into Sequential Monte Carlo (SMC)
or particle filtering. We defer this to a future phase.

---

## 5. Marginal Probabilities

### 5.1 Definition

The primary output of the inference engine is the **marginal probability** that
a specific player holds a specific tile:

$$
\boxed{
P(p \text{ has } t) = \sum_{\sigma \in \Omega_{\mathcal{C}}}
\frac{\mathbb{1}[\sigma(t) = p]}{|\Omega_{\mathcal{C}}|}
}
$$

In words: count how many feasible configurations assign tile $t$ to player $p$,
divided by the total number of feasible configurations.

### 5.2 Invariant Properties

The marginal probabilities must satisfy several invariants at all times. These
serve as correctness checks for the implementation.

**Invariant 1 -- Tile partition.** Each unknown tile is held by exactly one
player:

$$
\forall\, t \in U: \quad \sum_{p \in \mathcal{P}} P(p \text{ has } t) = 1
$$

**Invariant 2 -- Hand size.** The expected number of tiles each player holds
must equal their remaining count:

$$
\forall\, p \in \mathcal{P}: \quad \sum_{t \in U} P(p \text{ has } t) = r(p)
$$

**Invariant 3 -- Zero probability for excluded tiles.** If a tile has been ruled
out for a player (through pass constraints or other propagation), the
probability must be zero:

$$
t \notin C(p) \implies P(p \text{ has } t) = 0
$$

where $C(p)$ is the candidate set for player $p$ (see Section 6).

**Invariant 4 -- Determined tiles.** If a tile can only belong to one player
(it appears in exactly one player's candidate set), the probability must be
one:

$$
|\{p \in \mathcal{P} \mid t \in C(p)\}| = 1 \implies
P(p^* \text{ has } t) = 1
$$

where $p^*$ is the unique player with $t \in C(p^*)$.

**Invariant 5 -- Probability bounds.**

$$
\forall\, p \in \mathcal{P}, \; \forall\, t \in U: \quad
0 \le P(p \text{ has } t) \le 1
$$

### 5.3 Initial Marginals (Before Any Observations)

Before any play or pass is observed, every tile is equally likely to be held by
any of the three unknown players. By symmetry:

$$
P_0(p \text{ has } t) = \frac{r(p)}{|U|} = \frac{7}{21} = \frac{1}{3}
\quad \forall\, p \in \mathcal{P}, \; \forall\, t \in U
$$

One can verify Invariant 2: $\sum_{t \in U} \frac{1}{3} = \frac{21}{3} = 7 =
r(p)$.

---

## 6. Constraint Propagation

Before sampling or enumerating, we apply **constraint propagation** to reduce
the problem size. This is a deterministic preprocessing step that eliminates
provably impossible tile assignments.

### 6.1 Candidate Sets

For each player $p \in \mathcal{P}$, we maintain a **candidate set** $C(p)
\subseteq U$, which is the set of tiles that could still belong to player $p$
given all constraints observed so far.

**Initialization:**

$$
C_0(p) = U \quad \forall\, p \in \mathcal{P}
$$

### 6.2 Constraint Rules

We apply the following rules after each observation:

**Rule 1 -- Play removal.** When any player plays tile $t$:

$$
U \leftarrow U \setminus \{t\}
$$
$$
C(p) \leftarrow C(p) \setminus \{t\} \quad \forall\, p \in \mathcal{P}
$$

**Rule 2 -- Pass elimination.** When player $p$ passes with open ends $(a, b)$:

$$
C(p) \leftarrow C(p) \setminus B(a, b)
$$

where $B(a, b) = \{t \in U \mid a \in \text{values}(t) \lor b \in
\text{values}(t)\}$ is the set of tiles blocked by open ends $(a, b)$.

### 6.3 Arc Consistency (Derived Constraints)

After applying the direct rules above, we iterate the following derived
constraints until a **fixed point** is reached (no further changes occur):

**Rule 3 -- Determined hand.** If a player's candidate set has exactly as many
tiles as they need:

$$
|C(p)| = r(p) \implies
\begin{cases}
\text{All tiles in } C(p) \text{ are determined for } p \\
C(q) \leftarrow C(q) \setminus C(p) \quad \forall\, q \ne p
\end{cases}
$$

**Intuition:** If West can only possibly hold exactly 6 tiles and has 6 tiles
remaining, then West must hold all 6 of those tiles. Remove them from other
players' candidate sets.

**Rule 4 -- Unique assignment.** If a tile appears in exactly one player's
candidate set:

$$
|\{p \mid t \in C(p)\}| = 1 \implies t \text{ is determined for the unique }
p^* \text{ with } t \in C(p^*)
$$

**Rule 5 -- Insufficient capacity.** If removing a tile from a player's
candidate set would make it impossible for them to fill their hand, we can
deduce information about other players. Specifically, for a subset $S \subseteq
\mathcal{P}$:

$$
\left| \bigcup_{p \in S} C(p) \setminus \bigcup_{q \notin S} C(q) \right|
\ge \sum_{p \in S} r(p)
$$

This is a generalization (analogous to "naked pairs/triples" in Sudoku solving)
that ensures each subset of players has enough exclusive tiles to fill their
hands.

### 6.4 Fixed-Point Iteration

```
procedure PROPAGATE(C, r):
    repeat
        changed <- false
        for each p in {W, N, E}:
            if |C(p)| = r(p):
                for each q != p:
                    if C(q) intersects C(p):
                        C(q) <- C(q) \ C(p)
                        changed <- true
        for each t in U:
            holders <- {p : t in C(p)}
            if |holders| = 1:
                // t is determined for the unique holder
                // (handled by Rule 3 on next pass)
                changed <- true
            if |holders| = 0:
                ERROR: inconsistent state
    until not changed
    return C
```

The iteration terminates because each step strictly reduces the total size
$\sum_p |C(p)|$, which is bounded below by $\sum_p r(p)$.

---

## 7. Monte Carlo Sampling

When the feasible configuration space is too large for exact enumeration, we
estimate marginal probabilities by **sampling** from $\Omega_{\mathcal{C}}$.

### 7.1 Rejection Sampling

The simplest approach is **rejection sampling**:

```
procedure REJECTION_SAMPLE(U, C, r, N):
    accepted <- 0
    counts[p][t] <- 0  for all p, t

    while accepted < N:
        // Generate a random assignment respecting hand sizes
        sigma <- RANDOM_PARTITION(U, r)

        // Check all constraints
        if IS_CONSISTENT(sigma, C):
            accepted <- accepted + 1
            for each t in U:
                counts[sigma(t)][t] <- counts[sigma(t)][t] + 1

    // Compute marginals
    for each p in P, t in U:
        P_hat(p, t) <- counts[p][t] / N

    return P_hat
```

**RANDOM\_PARTITION** generates a uniformly random partition of $U$ into three
groups of sizes $r(W), r(N), r(E)$:

1. Shuffle $U$ randomly.
2. Assign the first $r(W)$ tiles to $W$, the next $r(N)$ to $N$, the rest to $E$.

**IS\_CONSISTENT** checks whether every tile assigned to player $p$ is in $C(p)$.

### 7.2 Improved Sampling: Constraint-Aware Generation

Pure rejection sampling can have a low acceptance rate when constraints are
tight. We improve it by sampling directly from candidate sets:

```
procedure CONSTRAINED_SAMPLE(U, C, r):
    // Sample in player order
    remaining <- U
    sigma <- {}

    for each p in [W, N, E] (in some order):
        eligible <- C(p) intersect remaining
        if |eligible| < r(p):
            return REJECT  // Infeasible partial assignment
        hand_p <- random sample of r(p) tiles from eligible
        sigma[p] <- hand_p
        remaining <- remaining \ hand_p

    if remaining is empty:
        return sigma
    else:
        return REJECT
```

This approach samples from candidate sets directly, which dramatically improves
the acceptance rate compared to naive rejection sampling. However, it does
**not** sample uniformly from $\Omega_{\mathcal{C}}$ because the sequential
assignment introduces ordering bias. Corrections can be applied via importance
weighting.

### 7.3 Acceptance Rate Analysis

Let $\alpha$ denote the acceptance rate (fraction of generated samples that are
valid). In the initial state with no constraints:

$$
\alpha_0 = 1
$$

As passes accumulate, candidate sets shrink and $\alpha$ decreases. In practice:

- **Early game** (0--2 passes): $\alpha > 0.5$ with constrained sampling.
- **Mid game** (3--6 passes): $\alpha$ varies, typically $0.05$--$0.5$.
- **Late game** (many constraints, few tiles): the feasible set is small enough
  for exact enumeration.

If $\alpha$ drops below a threshold (e.g., $0.01$), the system should switch to
exact enumeration or MCMC.

### 7.4 Convergence and Error Bounds

With $N$ accepted samples, the estimated marginal probability $\hat{P}(p
\text{ has } t)$ has standard error:

$$
\text{SE}\bigl(\hat{P}\bigr) = \sqrt{\frac{\hat{P}(1 - \hat{P})}{N}}
$$

For $N = 10{,}000$ samples and a true probability of $p = 0.33$:

$$
\text{SE} = \sqrt{\frac{0.33 \times 0.67}{10{,}000}} \approx 0.0047
$$

This gives roughly $\pm 1\%$ precision at a 95% confidence level, which is
more than sufficient for practical game play.

### 7.5 MCMC Alternative

For tightly constrained states where rejection sampling is inefficient, a
**Markov Chain Monte Carlo** (MCMC) approach can be used:

1. Start from any valid configuration $\sigma_0 \in \Omega_{\mathcal{C}}$.
2. At each step, propose a swap: pick two players $p_1, p_2$ and tiles $t_1 \in
   \sigma^{-1}(p_1), t_2 \in \sigma^{-1}(p_2)$, and swap their assignments.
3. Accept the swap if the new configuration is in $\Omega_{\mathcal{C}}$ (i.e.,
   both $t_1 \in C(p_2)$ and $t_2 \in C(p_1)$).
4. The stationary distribution is uniform over $\Omega_{\mathcal{C}}$.

This is a standard Metropolis-Hastings sampler with symmetric proposals, so the
acceptance probability is simply $\mathbb{1}[\sigma' \in \Omega_{\mathcal{C}}]$.

---

## 8. Exact Enumeration

When the number of unknown tiles is small enough, we can compute marginals
**exactly** by enumerating all valid configurations.

### 8.1 When to Enumerate

The number of valid configurations is bounded by:

$$
|\Omega_{\mathcal{C}}| \le \prod_{p \in \mathcal{P}}
\binom{|C(p)|}{r(p)}
$$

(This is an upper bound because it does not account for the requirement that the
assignments form a partition of $U$.)

Exact enumeration is feasible when $|U| \le 15$ or so, which typically occurs by
mid-game. With aggressive constraint propagation, the feasible set often
collapses to thousands or even hundreds of configurations.

**Decision rule:** If $\prod_p \binom{|C(p)|}{r(p)} < \tau$ for a threshold
$\tau$ (e.g., $\tau = 10^6$), use exact enumeration; otherwise, use Monte Carlo.

### 8.2 Enumeration Algorithm

```
procedure EXACT_ENUMERATE(U, C, r):
    counts[p][t] <- 0  for all p, t
    total <- 0

    // Enumerate valid hands for the first player
    for each subset H_W of size r(W) from C(W):
        remaining_1 <- U \ H_W

        // Enumerate valid hands for the second player
        for each subset H_N of size r(N) from C(N) intersect remaining_1:
            remaining_2 <- remaining_1 \ H_N

            // Check: can the third player hold the remaining tiles?
            if remaining_2 is a subset of C(E) and |remaining_2| = r(E):
                total <- total + 1
                for each t in H_W: counts[W][t] += 1
                for each t in H_N: counts[N][t] += 1
                for each t in remaining_2: counts[E][t] += 1

    // Compute exact marginals
    for each p, t:
        P(p, t) <- counts[p][t] / total

    return P
```

### 8.3 Complexity

The time complexity is:

$$
O\!\left(\binom{|C(W)|}{r(W)} \cdot \binom{|C(N) \cap \text{remaining}|}{r(N)}\right)
$$

In the worst case (no constraints), this is
$\binom{21}{7} \cdot \binom{14}{7} \approx 3.4 \times 10^9$, which is
infeasible. But with typical mid-game constraints:

- $|C(W)| \approx 10$, $r(W) = 5$: $\binom{10}{5} = 252$
- $|C(N) \cap \text{remaining}| \approx 8$, $r(N) = 5$: $\binom{8}{5} = 56$
- Total: $252 \times 56 = 14{,}112$ iterations -- trivially fast.

### 8.4 Optimizations

1. **Player ordering:** Enumerate the player with the smallest candidate set
   first to minimize branching.
2. **Early pruning:** After assigning tiles to the first player, check that the
   remaining tiles can still satisfy the other two players' constraints before
   continuing.
3. **Caching:** If a subset of tiles is determined (via constraint propagation),
   fix those assignments and enumerate only the remaining free tiles.

---

## 9. Worked Example

We trace through a concrete game scenario to illustrate how the inference engine
operates.

### 9.1 Initial Setup

**South's hand:**

$$
H = \{(0{|}1),\; (1{|}3),\; (2{|}5),\; (3{|}3),\; (4{|}6),\; (5{|}5),\; (6{|}6)\}
$$

**Unknown tiles** ($|U| = 21$):

$$
U = \{
(0{|}0),\, (0{|}2),\, (0{|}3),\, (0{|}4),\, (0{|}5),\, (0{|}6),\,
(1{|}1),\, (1{|}2),\, (1{|}4),\, (1{|}5),\, (1{|}6),\,
(2{|}2),\, (2{|}3),\, (2{|}4),\, (2{|}6),\,
(3{|}4),\, (3{|}5),\, (3{|}6),\,
(4{|}4),\, (4{|}5),\, (5{|}6)
\}
$$

**Initial state:**

| Quantity | Value |
|---|---|
| $\|U\|$ | 21 |
| $r(W) = r(N) = r(E)$ | 7 |
| $C(W) = C(N) = C(E)$ | $U$ (all 21 tiles) |
| $P_0(p \text{ has } t)$ | $1/3$ for all $p, t$ |

### 9.2 Action 1: South Plays (3|3)

$$
\text{PLAY}(S,\; (3{|}3),\; \text{start})
$$

**Open ends:** $(3, 3)$

**State update:**
- Tile $(3{|}3)$ was in South's hand, not in $U$. No change to $U$ or candidate
  sets.
- South's remaining count: $r(S) = 6$.
- $|U|$ remains 21.

The other players' candidate sets and remaining counts are unchanged. This
action only reveals a tile from South's (already known) hand.

### 9.3 Action 2: West Passes

$$
\text{PASS}(W,\; (3, 3))
$$

Since both open ends are 3, the blocked set is:

$$
B(3, 3) = \{t \in U \mid 3 \in \text{values}(t)\}
$$

From $U$, the tiles containing value 3 are:

$$
B(3, 3) \cap U = \{(0{|}3),\, (2{|}3),\, (3{|}4),\, (3{|}5),\, (3{|}6)\}
$$

Note: $(1{|}3)$ is in South's hand and $(3{|}3)$ has been played, so neither is
in $U$.

**Apply Rule 2 (pass elimination):**

$$
C(W) \leftarrow C(W) \setminus B(3, 3) = U \setminus \{(0{|}3),\, (2{|}3),\,
(3{|}4),\, (3{|}5),\, (3{|}6)\}
$$

$$
|C(W)| = 21 - 5 = 16
$$

West must hold 7 tiles from a candidate set of 16. The other players' candidate
sets remain at 21 tiles.

**Updated marginals (qualitative):** For any tile $t \in B(3,3) \cap U$:

$$
P(W \text{ has } t) = 0
$$

Since tile $t$ must be held by $N$ or $E$, and these two players are symmetric
with respect to $t$ at this point:

$$
P(N \text{ has } t) = P(E \text{ has } t) = \frac{1}{2}
\quad \text{(approximately, before further propagation)}
$$

The exact marginals require enumeration or sampling, but the qualitative shift
is clear: West's pass concentrates the probability of suit-3 tiles onto North
and East.

### 9.4 Action 3: North Plays (3|6)

$$
\text{PLAY}(N,\; (3{|}6),\; \text{end showing 3})
$$

**Open ends:** $(3, 6)$ -- the 3-end is extended by the 6-side of $(3{|}6)$, so
one end stays 3, the other becomes 6.

Wait -- let us be more precise. The chain before this play is just $(3{|}3)$,
so both ends show 3. North plays $(3{|}6)$ on one end, matching the 3. Now one
end shows 3 (the other end of the original $(3{|}3)$) and the other end shows 6
(the exposed end of $(3{|}6)$).

**Open ends after this play:** $(3, 6)$

**State update:**
- Remove $(3{|}6)$ from $U$:

$$
U \leftarrow U \setminus \{(3{|}6)\} \qquad |U| = 20
$$

- Update all candidate sets:

$$
C(p) \leftarrow C(p) \setminus \{(3{|}6)\} \quad \forall\, p \in \mathcal{P}
$$

- Decrement: $r(N) = 6$.
- The tile $(3{|}6)$ was already excluded from $C(W)$ by the pass, so $|C(W)|$
  remains 16 (it was already excluded). Actually, let us recount: $C(W)$ had 16
  tiles (21 minus the 5 suit-3 tiles). Since $(3{|}6)$ was among those 5, it
  was already not in $C(W)$. So $|C(W)| = 16 - 0 = 16$. But we also need to
  remove $(3{|}6)$ from $U$, so effectively $C(W) = C(W) \cap U_{\text{new}}$,
  giving $|C(W)| = 15$ (one tile removed from $U$ that was in $C(W)$? No --
  $(3|6)$ was NOT in $C(W)$.) Let us be precise.

**Correction:** After removing $(3{|}6)$ from $U$:

- $|U| = 20$
- $C(W)$: had 16 tiles, none of which was $(3{|}6)$ (already excluded by pass).
  But $(3{|}6)$ is no longer in $U$, so $C(W)$ remains the same 16 tiles minus
  any that are no longer in $U$. Since $(3{|}6) \notin C(W)$, we get
  $|C(W)| = 16$.

Actually, we should re-derive. $C(W)$ was the 16 tiles in $U$ that do not
contain value 3. Now $U$ shrinks by one tile -- $(3{|}6)$ -- which contains 3
and was already not in $C(W)$. So $|C(W)| = 16$ (unchanged).

- $C(N)$: was $U$ (21 tiles), now $U_{\text{new}}$ (20 tiles), so $|C(N)| = 20$.
- $C(E)$: same reasoning, $|C(E)| = 20$.

### 9.5 Action 4: East Plays (2|6)

$$
\text{PLAY}(E,\; (2{|}6),\; \text{end showing 6})
$$

East matches the 6-end with $(2{|}6)$. Now one end shows 3, the other shows 2.

**Open ends after this play:** $(3, 2)$

**State update:**
- Remove $(2{|}6)$ from $U$:

$$
U \leftarrow U \setminus \{(2{|}6)\} \qquad |U| = 19
$$

- Update all candidate sets:

$$
C(p) \leftarrow C(p) \setminus \{(2{|}6)\} \quad \forall\, p
$$

- Decrement: $r(E) = 6$.

**Updated candidate set sizes:**

| Player | $r(p)$ | $\|C(p)\|$ | Notes |
|---|---|---|---|
| $W$ | 7 | 15 or 16? | See below |
| $N$ | 6 | 19 | $20 - 1$ |
| $E$ | 6 | 19 | $20 - 1$ |

For West: $(2{|}6)$ does not contain value 3, so it was in $C(W)$. Removing it
gives $|C(W)| = 16 - 1 = 15$.

**Corrected candidate set sizes after Round 1:**

| Player | $r(p)$ | $\|C(p)\|$ | Tiles excluded from $C(p)$ |
|---|---|---|---|
| $W$ | 7 | 15 | 5 suit-3 tiles + $(2{|}6)$ played (was in $C(W)$, now removed from $U$) but $(3{|}6)$ not in $C(W)$, so $21 - 5 - 1 = 15$ |
| $N$ | 6 | 19 | $(3{|}6)$ played + $(2{|}6)$ played |
| $E$ | 6 | 19 | Same as North (no pass constraints on $N$ or $E$) |

### 9.6 Summary After Round 1

**Tiles remaining in $U$** (19 tiles):

$$
U = \{
(0{|}0),\, (0{|}2),\, (0{|}3),\, (0{|}4),\, (0{|}5),\, (0{|}6),\,
(1{|}1),\, (1{|}2),\, (1{|}4),\, (1{|}5),\, (1{|}6),\,
(2{|}2),\, (2{|}3),\, (2{|}4),\,
(3{|}4),\, (3{|}5),\,
(4{|}4),\, (4{|}5),\, (5{|}6)
\}
$$

**Candidate sets:**

$C(W) = U \setminus \{(0{|}3),\, (2{|}3),\, (3{|}4),\, (3{|}5)\}$ (15 tiles):

$$
C(W) = \{
(0{|}0),\, (0{|}2),\, (0{|}4),\, (0{|}5),\, (0{|}6),\,
(1{|}1),\, (1{|}2),\, (1{|}4),\, (1{|}5),\, (1{|}6),\,
(2{|}2),\, (2{|}4),\,
(4{|}4),\, (4{|}5),\, (5{|}6)
\}
$$

$C(N) = U$ (all 19 remaining tiles)

$C(E) = U$ (all 19 remaining tiles)

**Open ends:** $(3, 2)$

**Remaining counts:** $r(W) = 7$, $r(N) = 6$, $r(E) = 6$

**Constraint propagation check (Rule 3):** $|C(W)| = 15 > r(W) = 7$, so West's
hand is not yet fully determined. No further propagation triggers.

**Key observation:** The four tiles excluded from $C(W)$ -- namely $(0{|}3),
(2{|}3), (3{|}4), (3{|}5)$ -- must be distributed between North and East. Since
$r(N) + r(E) = 12$ and there are 19 unknown tiles, North and East hold 12 of the
19 tiles. The 4 suit-3 tiles excluded from West are guaranteed to be among
those 12.

### 9.7 Approximate Marginals

Without full enumeration, we can reason about some marginals:

**For tiles excluded from $C(W)$** (e.g., $(0{|}3)$):

$$
P(W \text{ has } (0{|}3)) = 0
$$

$$
P(N \text{ has } (0{|}3)) + P(E \text{ has } (0{|}3)) = 1
$$

**For tiles in all three candidate sets** (e.g., $(0{|}0)$):

The exact marginals depend on the combinatorial counting, but we can observe
that West's probability is slightly higher than $1/3$ for these tiles because
West has 7 slots to fill from only 15 candidates ($7/15 \approx 0.467$), while
North has 6 slots from 19 candidates ($6/19 \approx 0.316$) and East similarly
($6/19 \approx 0.316$). These are not the exact marginals (they do not account
for the partition constraint), but they give the right intuition: West's pass
makes it more likely that West holds the non-3 tiles.

Exact marginals would be computed by the Monte Carlo sampler or exact
enumerator.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|---|---|
| $\mathcal{T}$ | Full set of 28 domino tiles |
| $H$ | South's hand (7 known tiles) |
| $U$ | Unknown tiles: $\mathcal{T} \setminus H$ |
| $\mathcal{P}$ | Unknown players: $\{W, N, E\}$ |
| $\sigma$ | Configuration: assignment $U \to \mathcal{P}$ |
| $\sigma^{-1}(p)$ | Set of tiles assigned to player $p$ in configuration $\sigma$ |
| $r(p)$ | Remaining tile count for player $p$ |
| $C(p)$ | Candidate set for player $p$ |
| $B(a, b)$ | Set of tiles blocked by open ends $(a, b)$ |
| $\Omega_0$ | Initial set of all valid configurations |
| $\Omega_{\mathcal{C}}$ | Feasible configurations after constraints $\mathcal{C}$ |
| $\text{values}(t)$ | Set of pip values on tile $t$ |

## Appendix B: Combinatorial Reference

| Quantity | Formula | Approximate Value |
|---|---|---|
| Total tiles | $\binom{7}{2} + 7$ | 28 |
| Tiles per suit | 7 | -- |
| Initial configurations | $\binom{21}{7,7,7}$ | $3.4 \times 10^9$ |
| After 4 tiles played, 1 pass | varies | $10^5$ -- $10^7$ |
| Late game (6 tiles left) | $\binom{6}{2,2,2}$ | 90 |

## Appendix C: Implementation Correspondence

The mathematical objects defined here map to the codebase as follows:

| Math concept | Code module | Key class/function |
|---|---|---|
| Tile $(a, b)$ | `core/tiles.py` | `Tile` dataclass |
| Configuration $\sigma$ | `core/inference.py` | Assignment dict |
| Game state, $r(p)$ | `core/game_state.py` | `GameState` |
| Candidate sets $C(p)$ | `core/constraints.py` | `ConstraintSet` |
| Constraint propagation | `core/constraints.py` | `propagate()` |
| Monte Carlo marginals | `core/inference.py` | `MonteCarloSampler` |
| Exact marginals | `core/inference.py` | `ExactEnumerator` |
| Marginal $P(p, t)$ | `core/inference.py` | `compute_marginals()` |
