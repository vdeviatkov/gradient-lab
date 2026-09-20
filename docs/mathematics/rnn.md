# Recurrent networks and backpropagation through time

Every earlier network mapped one fixed-size input to one output. Text has no fixed size, and the
right prediction for the next character depends on characters arbitrarily far back. A recurrent
network handles this with one idea: a hidden state that is updated by the same function at every
step, so the same parameters serve a sequence of any length and the state is where the past lives.

## The model

With $x_t$ the one-hot encoding of the character at step $t$,

$$
h_t = \tanh\big(W_{xh} x_t + W_{hh} h_{t-1} + b_h\big), \qquad
z_t = W_{hy} h_t + b_y, \qquad
p_t = \operatorname{softmax}(z_t),
$$

and the loss is the mean over steps of $-\log p_t[x_{t+1}]$: the cross-entropy of the next
character. Because $x_t$ is one-hot, $W_{xh} x_t$ is a column of $W_{xh}$, and the implementation
reads it as one rather than multiplying. The initial state $h_0$ is zero.

Three matrices and two biases, shared across every step: the parameter count does not grow with
the sequence. For a vocabulary of $V$ and $H$ hidden units it is $HV + H^2 + H + VH + V$.

## Backpropagation through time

Unrolled over $T$ steps the network is an ordinary feed-forward graph in which the same $W_{hh}$
appears $T$ times. The gradient is the backpropagation milestone's chain rule applied to that
graph, with one addition: every step's contribution to a shared parameter is summed.

Walking backwards from step $T-1$, with $\delta^z_t = p_t - \text{onehot}(x_{t+1})$ scaled by
$1/T$,

$$
\frac{\partial L}{\partial h_t} = W_{hy}^\top \delta^z_t + W_{hh}^\top \delta^a_{t+1},
\qquad
\delta^a_t = \frac{\partial L}{\partial h_t} \odot (1 - h_t^2),
$$

where the second term is the gradient carried back from the following step through the
recurrence — zero at the last step. Then

$$
\frac{\partial L}{\partial W_{hh}} \mathrel{+}= \delta^a_t\, h_{t-1}^\top, \quad
\frac{\partial L}{\partial W_{xh}}[:, x_t] \mathrel{+}= \delta^a_t, \quad
\frac{\partial L}{\partial b_h} \mathrel{+}= \delta^a_t, \quad
\frac{\partial L}{\partial W_{hy}} \mathrel{+}= \delta^z_t\, h_t^\top, \quad
\frac{\partial L}{\partial b_y} \mathrel{+}= \delta^z_t .
$$

The tests verify this against central differences over every parameter, from the zero state and
from an arbitrary carried state.

## Truncation

Backpropagating through a million-character text would need every hidden state of the text in
memory and a gradient that is a product of a million Jacobians. Truncated BPTT cuts the text into
windows of $T$ steps. The hidden state is *carried* from one window into the next — the forward
pass is exactly the forward pass over the whole text — but no gradient flows across the boundary:
the window's initial state is treated as a constant.

The consequence is precise. A dependency that spans more than $T$ steps produces no gradient, so
nothing in the parameters is pushed to capture it. The test makes this concrete with a stream in
which every other character is a copy of a random character that entered six steps earlier: with
a window of 4 the network learns nothing beyond chance, and with a window of 16 it reaches the
best achievable loss. The experiment repeats the comparison on real text.

One subtlety the test also records: truncation does not stop the *state* from carrying
information across windows, only the gradient. On a periodic pattern a window of one step is
enough, because each step's loss already tells the recurrence which state to be in next, and
counting emerges without any long-range credit assignment. It is the need to store something that
pays off only later that truncation blocks.

## Why the gradient vanishes

The gradient of a loss at step $T$ with respect to the state $k$ steps earlier is a product,

$$
\frac{\partial L_T}{\partial h_{T-k}}
= \frac{\partial L_T}{\partial h_T}\prod_{j=1}^{k}
  \operatorname{diag}\!\big(1 - h_{T-j+1}^2\big)\, W_{hh} ,
$$

and a product of $k$ matrices behaves geometrically in $k$. Each tanh derivative is at most one
and the recurrent matrix is initialized with singular values near one, so in practice the product
shrinks: the gradient that reaches twenty steps back is a small fraction of what reaches the last
state. `CharRnn::gradient_reach` measures exactly this — the norm of $\partial L_T / \partial
h_{T-k}$ for every $k$ — and the experiment reports it before and after training. This is the
residual milestone's argument again in a different setting: there the product ran over layers,
here it runs over time, and the same geometric decay results. The gated architectures of the next
milestone exist to give this product an additive path.

## Why the gradient explodes, and clipping

The same product can grow. Nothing keeps the recurrent matrix's singular values below one during
training, and when the product grows the gradient can be orders of magnitude larger on one window
than on the next. A single such step at a normal learning rate can move the parameters far enough
to undo everything learned. Gradient clipping is the standard guard: if the whole gradient's
Euclidean norm exceeds a threshold $c$, it is scaled to $c$,

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right),
$$

which preserves the direction and bounds the step. The implementation clips the global norm over
all parameters at once, reports the fraction of updates that were clipped and the largest norm
seen before clipping, and can be switched off so the experiment can show what the norms do
without it. Clipping does nothing about a *vanishing* gradient; scaling zero is zero.

## Evaluation

Cross-entropy per character in nats is what the model minimizes; divided by $\ln 2$ it is bits per
character, and its exponential is perplexity — the number of equally likely characters the model
is effectively choosing between. A uniform guess over 62 characters scores $\ln 62 = 4.13$ nats.

Count-based $n$-gram models set the floor. A unigram model ignores context; a bigram model
conditions on one character; a trigram on two. Each is a table of counts with additive smoothing
so that an unseen continuation keeps a finite probability. Anything that claims to model text has
to beat the trigram, and the trigram is not weak: on this corpus it gets most of the way from the
uniform guess to what the recurrent network reaches. The experiment scores every model on the
same held-out splits and reports parameter counts and characters per second alongside, so the
recurrent network's advantage can be read against what it costs.
