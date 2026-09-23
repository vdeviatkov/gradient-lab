# Experiment 18: reinforcement learning on Snake

The first milestone without labels. A Snake environment written from scratch, a random policy
and a scripted food-seeker as baselines, tabular Q-learning over the observation's bit patterns,
and deep Q-learning with experience replay and a target network — every agent evaluated the same
way, by playing greedily on a fixed set of held-out seeds.

## The environment

A 10x10 board. The snake starts three cells long at the centre heading right, and one food cell
sits uniformly at random on a free square. Three actions — straight, turn right, turn left — so
the snake can never reverse into its own neck. Eating pays **+1** and grows the snake; hitting a
wall or its own body pays **−1** and ends the episode; a step pays nothing. An episode is
**truncated** after 100 steps without eating (times one plus the score, so a longer snake gets
proportionally longer to find its food), and truncation pays no penalty and is not treated as
terminal — the agent was interrupted, not killed, so its future value still counts. The best
possible score is 97.

Both agents see the same eleven-bit observation: three danger flags for the cells the three
actions would move into, four one-hot heading flags, and four flags for where the food lies
relative to the head. The food sequence is a function of the seed, so every agent is measured on
identical episodes.

## Setup

Every learner: **3000 training episodes**, discount 0.95, epsilon decaying linearly from 1.00 to
0.02 over 30000 action selections, seed `20260922`. Every evaluation: **200 greedy episodes** on
seeds far from the training ones — no exploration at all, so what is reported is the policy
rather than the policy plus its noise. Evaluations run every 150 training episodes, and a run's
*stability* is the spread of its last six.

The tabular agent uses a learning rate of 0.1. The deep agent is two 64-unit ReLU layers to three
linear outputs, Adam at 0.001, batches of 32 drawn from a 20000-transition replay buffer after a
500-transition warmup, target network refreshed every 200 updates, gradients clipped at norm 10.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_snake_rl
```

About three and a half minutes. It writes `data/snake_tabular.checkpoint` and
`data/snake_dqn.checkpoint` (`ML_SCRATCH_SNAKE_CHECKPOINTS` overrides the directory). It exits
unsuccessfully unless both learners beat the random policy tenfold, the network beats both the
scripted baseline and the table, the table given ten times the episodes beats its own 3000-episode
score, the random policy scores under 2, removing replay both doubles the spread of the last six
evaluations and doubles how far the run dips below its own mean, exactly 256 states are ever
visited, and the table trains at least twenty times faster than the network.

## Measured result

Recorded on 2026-09-22 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type
`Release`, seed `20260922`. Repeat runs reproduced every score, spread and update count exactly;
the wall-clock columns are the only ones that move. Total wall time 3 minutes 24 seconds on an
idle machine.

### Baselines and learners

| Policy | Mean score | Best | Mean return | Mean steps | Death rate |
|---|---:|---:|---:|---:|---:|
| random | 0.21 | 2 | −0.79 | 19.1 | 1.00 |
| scripted food-seeker | 17.55 | 38 | 16.55 | 171.2 | 1.00 |
| tabular Q-learning, 3000 episodes | 16.32 | 35 | 15.32 | 141.8 | 1.00 |
| **deep Q-learning, 3000 episodes** | **18.86** | **39** | 17.86 | 149.8 | 1.00 |
| tabular Q-learning, 30000 episodes | **19.71** | 38 | 18.71 | 163.1 | 1.00 |

The random policy eats 0.21 food and dies in 19 steps: both learners beat it by roughly eighty
times, which is the only claim that needs no interpretation.

**The scripted baseline is the interesting one.** It is four lines — head for the food unless
that move kills you, otherwise take any move that does not — and it scores **17.55**, above the
tabular agent's 16.32 after 3000 episodes. That is not a defect in the learning; it is what the
observation makes possible. The eleven features were chosen to contain exactly what a greedy
food-seeker needs, so a policy that uses them directly is already strong, and a learner has to
earn its keep by finding something the script does not have: the sense to take a step *away* from
the food when the alternative is trapping itself. The deep agent does find some of it (18.86);
the table, given enough episodes, finds more (19.71).

Every policy here dies rather than being truncated. That is worth stating because the opposite
failure — an agent that learns to circle forever, never eating and never dying — is the one a
score alone would not distinguish from a good policy, which is why the death rate and the episode
length are reported next to it.

### Learning curves

Greedy mean score, evaluated every 600 episodes:

| Episodes | 600 | 1200 | 1800 | 2400 | 3000 |
|---|---:|---:|---:|---:|---:|
| tabular | 9.2 | 14.7 | 15.2 | 15.3 | 16.3 |
| DQN | 18.6 | 16.7 | 17.9 | 19.4 | 18.9 |

**The network is far more episode-efficient.** After 600 episodes it is already at 18.6, a score
the table does not reach in 3000. A table has to visit each state many times before its entry is
worth anything; a network shares what it learns in one state with every similar state, and with
only eleven binary features almost every state is similar to some other.

The table is still climbing at 3000 episodes, and an episode costs it almost nothing, so the fair
question is what it reaches for a comparable budget.

### The cost of each learner

| | Episodes | Training time | Updates | Mean score |
|---|---:|---:|---:|---:|
| tabular | 3000 | **0.14 s** | 250831 single-entry | 16.32 |
| tabular | 30000 | **0.85 s** | 3558619 single-entry | **19.71** |
| DQN | 3000 | 47.7 s | 270487 batched (32 each) | 18.86 |

**The table is 56 times cheaper than the network and ends up ahead.** At 30000 episodes it takes
0.85 seconds and scores 19.71 against the network's 18.86 for 47.7 seconds. The network's
advantage is entirely in episodes, and episodes are the cheap resource here — a Snake step costs
nothing, and there is no reason to economize on them.

That is the honest shape of the comparison on *this* problem, and the reason is in part two's
one-line finding: **only 256 of the 2048 observation bit patterns can ever occur.** The eleven
bits are not independent — the heading is one-hot (4 of 16 patterns) and the food cannot be both
left and right or both up and down (8 of 16) — so the reachable state space is 8 danger patterns
× 4 headings × 8 food directions = 256, and the table visits all of them. A function
approximator earns its place when the state space is too large to enumerate; here it is not, and
the table wins on every axis but episode count. The same DQN code on an observation that included
the body's layout, where the state space is astronomically large, would not have that
competition.

### What the network needs

Each mechanism removed one at a time, everything else held fixed. `spread` is the standard
deviation of the last six evaluations and `dip` is how far the run's worst of those falls below
its own mean — the measure of a run that keeps collapsing and recovering.

| Run | Settled mean | Spread | Dip | The last six evaluations |
|---|---:|---:|---:|---|
| DQN | 19.01 | 0.82 | 1.12 | 20.4 19.4 19.2 18.2 17.9 18.9 |
| no target network | 18.66 | 0.57 | 0.98 | 18.4 18.8 19.6 17.7 18.7 18.8 |
| **no replay** | 17.14 | **2.52** | **2.68** | **16.8 20.5 15.8 14.5 20.6 14.7** |
| no exploration decay | 18.63 | 0.94 | 1.40 | 19.5 19.1 19.7 17.5 17.2 18.7 |
| tabular (for reference) | 15.84 | 0.37 | 0.55 | 15.5 15.3 16.3 15.8 15.8 16.3 |

**Replay is the one that matters.** Without it — a buffer the size of one batch, so every update
sees the last 32 transitions from one stretch of one episode — the spread of the last six
evaluations triples and the run dips 2.68 below its own mean, against the full agent's 1.12. The
snapshots tell it plainly: **16.8, 20.5, 15.8, 14.5, 20.6, 14.7**, swinging six points between
consecutive evaluations 150 episodes apart, while the full agent moves within two and a half.
Every one of those evaluations still had a death rate of 1.00, so the collapses are not the snake
learning to circle — they are a policy that keeps being found and then destroyed. Correlated
samples do not merely slow learning down; they let it undo itself.

**The target network changed nothing here**, and neither did the exploration schedule. Removing
the target network gave 18.66 against 19.01, with a *lower* spread; removing the schedule and
holding epsilon at 0.02 throughout gave 18.63. Both are within the run-to-run variation the
spread column measures, so neither is a real effect at this scale. The explanations are
different and both worth stating. A target network stabilizes a target that moves faster than
the learner can follow; on a problem this small with a 0.001 learning rate, it does not move fast
enough to matter. And an *untrained* network's greedy policy is already arbitrary — its outputs
are near-random, so its argmax wanders — which means it explores by accident for the first
several hundred episodes whether or not epsilon tells it to. The schedule is insurance against a
failure this problem does not produce.

The no-exploration run is also the slowest (64 s against 48 s) because a greedy policy survives
longer, so each episode has more steps and more updates: 361742 against 270487.

### An episode

The tabular agent's first evaluation episode, at its end — score 26 in 267 steps, about to run
out of room:

```text
############
#          #
#     ooooo#
#     ooo o#
#       o o#
#       o o#
#       o o#
#       ooo#
#       oo@#
#   *   ooo#
#     ooooo#
############
```

This is where the eleven-feature observation runs out. The snake can see whether the next cell in
each of three directions is fatal, but nothing about the shape of the space it is coiling itself
into, so it walks into pockets it cannot leave. No amount of training fixes that: the information
is not in the observation.

### Limitations

One seed per configuration, and the ablations differ by one mechanism but share everything else,
so a difference of a point is inside the noise the spread column measures — only the replay
result is large enough to read as an effect. The scripted baseline is deliberately simple; a
better script (one that checks whether a move leaves a reachable exit) would beat every learner
here. The board is small and the observation compact, which is what makes the table competitive
and is not representative of the problems function approximation is for. And the discount,
learning rates, and network shape were chosen from a short sweep rather than tuned per agent.
