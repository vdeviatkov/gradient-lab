# Q-learning, tabular and deep

Every earlier milestone was given the right answer for each input. Here there is no right answer
— only a reward that arrives after an action, sometimes long after the action that earned it.
The problem is no longer to fit a function to labelled examples but to find a policy whose
*consequences* are good, and the labels have to be manufactured from the agent's own experience.

## The setting

At each step the agent sees a state $s_t$, picks an action $a_t$, receives a reward $r_t$ and
lands in $s_{t+1}$. The return from a step is the discounted sum of everything that follows,

$$
G_t = \sum_{k \ge 0} \gamma^k r_{t+k}, \qquad 0 \le \gamma < 1,
$$

and the discount $\gamma$ is what makes the sum finite and what expresses how much a reward now
is worth against a reward later. With $\gamma = 0.95$ a food ten steps away is worth $0.60$ of
one eaten immediately, which is the pressure that makes a policy go *toward* the food rather than
merely stay alive.

The action-value function $Q^\pi(s, a)$ is the expected return from taking $a$ in $s$ and
following $\pi$ afterwards. The optimal one satisfies the Bellman equation

$$
Q^*(s, a) = \mathbb E\Big[r + \gamma \max_{a'} Q^*(s', a')\Big],
$$

and a policy that always takes $\arg\max_a Q^*(s, a)$ is optimal. Everything below is a way of
solving that equation from samples.

## Tabular Q-learning

With finitely many states, $Q$ is a table, and each observed transition is used to move one entry
toward the Bellman target:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\Big(\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{target}} - Q(s, a)\Big).
$$

The bracket is the *temporal-difference error*: how wrong the current estimate was about what
followed. Two details decide whether the update is right.

**A terminal state has no future.** When the episode ended in death, the target is $r$ alone —
bootstrapping the value of the state a dead snake would have seen is a bug that quietly inflates
every value near a wall. But a *truncated* episode, cut off by a step limit, is not terminal: the
agent was interrupted, not killed, and its future value still counts. The implementation carries
the two apart in every transition, and the environment distinguishes them.

**The update is off-policy.** The $\max$ is over the greedy action, while the action actually
taken came from an exploring policy. That is what lets an agent learn the optimal policy while
behaving suboptimally, and it is why evaluation must be done *greedily and separately*: an
$\varepsilon$-greedy score measures the policy plus its noise, not the policy.

## Exploration

A greedy agent with an empty table takes action 0 forever and learns nothing. $\varepsilon$-greedy
takes a uniformly random action with probability $\varepsilon$, and the schedule decays
$\varepsilon$ from 1 to a small floor over a fixed number of *action selections* — not episodes,
so an agent that dies early does not race through its exploration budget. The floor is not zero:
a little noise keeps the agent from locking into a policy whose alternatives it has stopped
measuring.

## Deep Q-learning

When the state space is too large to tabulate, $Q(s, a)$ becomes a network with one output per
action, trained by regressing its prediction for the taken action onto the same Bellman target:

$$
L = \big(Q_\theta(s, a) - [\,r + \gamma \max_{a'} Q_{\theta^-}(s', a')\,]\big)^2 .
$$

Only the taken action has a target, so the other outputs' gradients are zero — the transition
says nothing about actions it did not try. The implementation computes exactly that by handing
the network an output gradient that is zero everywhere but the taken action, which is what the
GAN milestone's `backpropagate` exists for.

Two mechanisms make this trainable, and the experiment removes each in turn.

**Experience replay.** Consecutive steps of one episode are nearly the same state, and a network
fitted on them in order forgets everything else — the samples are correlated in exactly the way
gradient descent assumes they are not. A fixed-capacity buffer stores recent transitions and each
update draws a uniform batch from it, which decorrelates the batch and lets each transition be
learned from more than once.

**A target network.** The target contains $Q$ itself, so regressing onto it is chasing a moving
target: every update changes the thing being fitted. $\theta^-$ is a copy of the parameters
refreshed every few hundred updates, which holds the target still between refreshes. In the
limit of refreshing every update, the target network is the learner and the stabilization is gone.

## Why evaluation is separate

A reinforcement learner's training score is not a measurement of anything: it mixes the policy
with its exploration, it changes as $\varepsilon$ decays, and it is computed on the episodes the
agent chose to visit. Every number reported in the experiment comes from a separate greedy rollout
on a fixed set of held-out seeds, identical for every agent, so that two policies are compared on
the same food sequences.

Two failure modes need distinguishing, and one number cannot do it. An agent that dies instantly
has a low score and a death rate of one; an agent that learns to circle forever, never eating and
never dying, has a low score and a death rate near zero. The experiment reports the mean score,
the mean return, the mean episode length and the death rate together, and a run's *stability* as
the spread of its last several evaluations — because a learner that oscillates between a good
policy and a collapsed one can end on either, and the snapshot alone would not say which.
