# Gated recurrent cells: LSTM and GRU

The recurrent milestone measured its own limit. The gradient of one prediction with respect to
the hidden state twenty steps earlier was 0.6% of the gradient at the last step, and training had
made the decay steeper, not shallower. The cause is structural: an elman cell rewrites its whole
state every step,

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b),
$$

so the gradient reaching back $k$ steps is a product of $k$ Jacobians, each a tanh derivative
times $W_h$. Gated cells change the recurrence so that part of the state is carried *additively*,
which gives the gradient a path that is a product of gates rather than of weight matrices.

## The LSTM

Four affine functions of the input and previous hidden vector, three of them squashed to $(0, 1)$
by a sigmoid and one to $(-1, 1)$ by a tanh:

$$
\begin{aligned}
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) &&\text{input gate}\\
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) &&\text{forget gate}\\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) &&\text{output gate}\\
g_t &= \tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g) &&\text{candidate}
\end{aligned}
$$

and then the two state updates,

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t, \qquad h_t = o_t \odot \tanh(c_t).
$$

The cell state $c_t$ is the point. It is updated by scaling the old value and adding a gated
candidate — no matrix multiplies it and no squashing function is applied to it. Between steps the
carried state is the pair $(h_t, c_t)$, twice the hidden size. The implementation initializes the
forget gate's bias to one, so that a fresh cell keeps most of its state ($\sigma(1) \approx 0.73$)
before it has learned when to clear it; with the bias at zero the state would halve every step.

## The GRU

Two gates and one state vector:

$$
\begin{aligned}
r_t &= \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r) &&\text{reset gate}\\
z_t &= \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z) &&\text{update gate}\\
n_t &= \tanh\big(W_{xn} x_t + r_t \odot (W_{hn} h_{t-1}) + b_n\big) &&\text{candidate}\\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1}.
\end{aligned}
$$

The update gate interpolates between keeping the old state and taking the candidate: the same
additive path as the LSTM's cell, with $z_t$ playing the forget gate's part and $1 - z_t$ the
input gate's, and only one vector to carry. The reset gate scales the recurrent term *inside* the
candidate's tanh, which is what lets the cell propose a state that ignores the old one. The
implementation uses a single bias per gate; some formulations keep separate biases on the input
and recurrent halves of the candidate, which changes nothing about the gradient.

## Backpropagation through the gates

The output layer, the loss, the truncation, the carried state, and the outer loop of
backpropagation through time are the recurrent milestone's, unchanged. What changes is one step's
backward pass. For the LSTM, with $\bar{x}$ denoting $\partial L / \partial x$ and $\bar h_t$
already holding the gradient from this step's logits and from step $t+1$'s recurrence:

$$
\begin{aligned}
\bar o_t &= \bar h_t \odot \tanh(c_t), &
\bar c_t &\mathrel{+}= \bar h_t \odot o_t \odot \big(1 - \tanh^2(c_t)\big),\\
\bar i_t &= \bar c_t \odot g_t, \quad \bar g_t = \bar c_t \odot i_t, \quad \bar f_t = \bar c_t \odot c_{t-1}, &
\bar c_{t-1} &= \bar c_t \odot f_t,
\end{aligned}
$$

then each gate's pre-activation gradient through its own nonlinearity ($\sigma' = s(1 - s)$,
$\tanh' = 1 - g^2$), the parameter gradients as sums of outer products with $x_t$ and $h_{t-1}$,
and $\bar h_{t-1} = W_h^\top \bar a_t$ over all four gate rows. The line that matters is
$\bar c_{t-1} = \bar c_t \odot f_t$: the gradient through the cell path is multiplied by a forget
gate and nothing else. Over $k$ steps that path contributes $\prod f$, which stays near one while
the gates stay near one, whereas the elman path contributes a product of $k$ matrices.

For the GRU, with $q_t = W_{hn} h_{t-1}$ the recurrent term before the reset gate:

$$
\begin{aligned}
\bar n_t &= \bar h_t \odot (1 - z_t), \qquad \bar z_t = \bar h_t \odot (h_{t-1} - n_t), \\
\bar a^n_t &= \bar n_t \odot (1 - n_t^2), \qquad \bar r_t = \bar a^n_t \odot q_t, \qquad
\bar q_t = \bar a^n_t \odot r_t,\\
\bar h_{t-1} &= \bar h_t \odot z_t + W_{hn}^\top \bar q_t + W_{hr}^\top \bar a^r_t + W_{hz}^\top \bar a^z_t .
\end{aligned}
$$

The first term of $\bar h_{t-1}$ is the additive path: the gradient scaled by the update gate,
with no matrix. The candidate row's recurrent weights and its carry take $\bar q_t = r_t \odot
\bar a^n_t$ rather than $\bar a^n_t$, since $W_{hn} h_{t-1}$ enters through the reset gate.

Both backward passes are verified against central differences over every parameter, from the
zero state and from a carried state, and the carried $(h, c)$ pair is verified to be a complete
state: scoring a sequence in two halves with the state carried gives the loss of scoring it whole.

## What the gates buy, and what they cost

`gradient_reach` measures the norm of one prediction's gradient with respect to the carried state
$k$ steps earlier. For the elman cell it decays geometrically; for a gated cell the additive path
keeps it alive as long as the gates stay open. The delayed-copy stream in the experiment turns
that into a task: every other character is a copy of the random bit written $d$ pairs earlier,
which entered the network $2d$ steps before the copy is predicted, so a cell that cannot carry a
bit that far scores chance on the copies. The elman cell learns the copy at six steps and not at
twelve; the gated cells learn both, and the LSTM learns it at twenty.

The cost is parameters and arithmetic. Per hidden unit an LSTM has four gate rows and a GRU three
against the elman's one, so at a fixed parameter budget a gated cell has fewer units — 58 LSTM or
70 GRU units against 128 elman units at about 32000 parameters — and at a fixed width it does
three to four times the work per character. The experiment compares the cells both ways, because
the two comparisons do not agree: what a gate buys depends on what it is charged for.

Whether the extra reach translates into a lower loss on text is the empirical question. On
Shakespeare the dependencies that matter most are short — the recurrent milestone showed a 5-step
truncation window beating a 50-step one at equal epochs — so a gated cell's advantage on this
corpus is expected to be modest and to depend on the budget, and the experiment records what it
was.
