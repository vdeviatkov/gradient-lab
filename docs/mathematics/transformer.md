# A decoder-only Transformer

The recurrent milestones carried the past in a state, and paid for it twice: a gradient that had
to pass through every intervening step, and a computation that could not be parallelized across
positions because each depended on the one before. Attention removes the state. Every position
reads every earlier position directly, with weights the model computes from the content, so the
gradient to a character twenty steps back is one product away rather than twenty, and every
position of a sequence is computed at once.

## Embedding and position

A character id $x_t$ becomes a vector $e_{x_t} \in \mathbb R^D$ from a learned table. Attention
by itself is permutation-invariant — a sum over positions weighted by content — so the model has
to be told where each token is. The implementation adds a fixed sinusoidal encoding,

$$
p_t[2i] = \sin\!\big(t \cdot 10000^{-2i/D}\big), \qquad
p_t[2i+1] = \cos\!\big(t \cdot 10000^{-2i/D}\big),
$$

whose wavelengths run geometrically from $2\pi$ to $10000 \cdot 2\pi$: every position gets a
distinct pattern and nearby positions get similar ones. The residual stream starts as
$h^{(0)}_t = e_{x_t} + p_t$.

The tests record one subtlety. With the causal mask, a *single* block is exactly
permutation-invariant over the past when the encoding is off: the last position's prediction
cannot tell two orderings of the earlier tokens apart. A second block can, because each
position's first-block output already depends on the prefix the mask lets it see. Depth leaks
order; it does not supply it usefully, which the experiment's ablation measures.

## One block

Each block is two residual sub-layers with layer normalization applied *before* each — the
pre-normalization arrangement, which keeps the residual stream an unnormalized sum and trains
without warm-up:

$$
\begin{aligned}
a &= \operatorname{LN}_1(h), & h' &= h + \operatorname{Attention}(a)\,W_o^\top,\\
b &= \operatorname{LN}_2(h'), & h'' &= h' + \operatorname{ReLU}(b W_1^\top + b_1)\,W_2^\top + b_2 .
\end{aligned}
$$

Layer normalization standardizes each position's vector across its $D$ dimensions and rescales
with a learned gain and bias — the regularization milestone's layer normalization, applied to a
row. The feed-forward layer is a two-layer network applied to each position independently, with
a hidden width of $4D$.

## Causal multi-head attention

Each of $H$ heads projects the normalized input to queries, keys, and values of width $d = D/H$:

$$
q_t = a_t W_q^\top, \quad k_t = a_t W_k^\top, \quad v_t = a_t W_v^\top,
$$

(the implementation stores one $D \times D$ matrix per projection and slices it per head), then
for every query position $t$ scores every key position $s \le t$,

$$
\alpha_{ts} = \frac{\exp\!\big(q_t \cdot k_s / \sqrt d\big)}{\sum_{s' \le t} \exp\!\big(q_t \cdot k_{s'} / \sqrt d\big)},
\qquad
c_t = \sum_{s \le t} \alpha_{ts}\, v_s .
$$

The restriction $s \le t$ is the causal mask: position $t$ predicts $x_{t+1}$ and must not see it
or anything after it. The scale $1/\sqrt d$ keeps the dot products' variance near one so the
softmax starts soft. The heads' contexts are concatenated and projected by $W_o$.

With the optional relative position bias, a learned per-head value $r_h[t - s]$ is added to the
score before the softmax. Where the sinusoidal encoding tells a head where a token *is*, this
tells it how far away the token is, directly — the quantity a head that must look a fixed number
of steps back actually needs. The experiment shows the two are not interchangeable.

## Output and loss

After the last block, a final layer normalization and a linear layer give logits over the next
character at every position, and the loss is the mean cross-entropy over the positions of a
window. Unlike the recurrent networks there is no carried state: training draws windows of
$T + 1$ characters at random offsets, and evaluation reads a held-out text in abutting windows of
$T$ predictions, so a character early in a window is predicted from only the few before it. A
strided evaluation, with windows overlapping so that each character is predicted from the window
in which it has the most context, costs proportionally more forward passes; the implementation
offers both and the experiment says which it used.

## Backpropagation through attention

Every gradient is derived by hand and verified against central differences, as in every earlier
milestone. The new pieces are the softmax and the bilinear score. With $\bar c_t$ the gradient
arriving at a head's context,

$$
\bar\alpha_{ts} = \bar c_t \cdot v_s, \qquad
\bar v_s \mathrel{+}= \alpha_{ts}\, \bar c_t, \qquad
\bar z_{ts} = \alpha_{ts}\Big(\bar\alpha_{ts} - \sum_{s'} \alpha_{ts'} \bar\alpha_{ts'}\Big),
$$

where $z$ is the pre-softmax score; then $\bar q_t \mathrel{+}= \bar z_{ts} k_s / \sqrt d$ and
$\bar k_s \mathrel{+}= \bar z_{ts} q_t / \sqrt d$, and the relative bias, if present, takes
$\bar z_{ts}$ unscaled. The three projections, the output projection, the feed-forward layers,
and the embedding are the backpropagation milestone's affine maps, and each layer normalization
is the regularization milestone's, with the reduction taken across a position's dimensions. The
residual connections pass the incoming gradient straight through, as the residual milestone
established, which is what lets a stack of blocks train from the start.

## Why the copy task needs a marker

Experiment 16's delayed-copy stream alternates a random bit with a copy of the bit written ten
pairs earlier. A recurrent cell tracks which position is which — copy or random — in its state,
for free, by alternating. A transformer has no state and must read parity off the positional
encoding, and a sinusoid has no period-two component to read it from: the highest frequency has
a wavelength of $2\pi$ positions. In an early attempt the transformer plateaued at a loss that
corresponds exactly to attending to the right place without knowing whether the current position
is a copy. The stream used for the transformer therefore writes copies as 2 and 3 rather than 0
and 1, so the current token says what comes next; the recurrent cells are rerun on the same
marked stream so the comparison is fair, and on it the transformer's advantage is what the
architecture promises — a head that puts most of its weight exactly twenty steps back.
