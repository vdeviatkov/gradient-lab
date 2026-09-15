# Initialization, regularization, and normalization

Six techniques that change how a network trains without changing what it computes at inference —
except for the one that does.

## Weight initialization

Consider a layer $z = Wa$ with $n$ inputs. If the weights are drawn independently with variance
$\sigma_w^2$ and the inputs have variance $\sigma_a^2$, then

$$
\operatorname{Var}(z) = n\,\sigma_w^2\,\sigma_a^2 .
$$

Activation variance is therefore multiplied by $n\sigma_w^2$ at every layer. Anything other than
roughly one compounds geometrically with depth: too small and the signal — and with it the gradient
— decays to nothing; too large and it explodes, or saturates a squashing activation where the slope
is nearly zero. **Glorot** takes the middle course by balancing the forward and backward passes,
which have fan-in and fan-out respectively:

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}},
\qquad
w \sim \mathcal{U}\!\left[-\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}},\;
\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}\right],
$$

using $\operatorname{Var}(\mathcal{U}[-l, l]) = l^2/3$. **He** corrects for ReLU: it zeroes half its
inputs, halving the variance it passes on, so the target doubles to
$\operatorname{Var}(w) = 2/n_{\text{in}}$.

Zero initialization fails for a different reason. Every hidden unit in a layer then computes the
same function, receives the same gradient, and is updated identically — they remain indistinguishable
forever, so a layer of $h$ units has the expressive power of one. The tests assert exactly that: the
gradient rows of a zero-initialized layer are identical. Note that this is *not* a problem for
softmax or logistic regression, whose output rows are distinguished by their own labels, which is
why those milestones initialize at zero deliberately.

## L2 and L1 penalties

Add a penalty on the weights to the objective:

$$
L_{\text{total}} = L_{\text{data}} + \lambda_1 \sum_j |w_j| + \frac{\lambda_2}{2}\sum_j w_j^2 .
$$

Their gradients are $\lambda_1 \operatorname{sign}(w_j)$ and $\lambda_2 w_j$, which behave quite
differently near zero. L2's force is **proportional to the weight**, so it shrinks large weights hard
and essentially stops bothering small ones — the result is many small weights and none exactly zero.
L1's force has **constant magnitude** regardless of how small the weight already is, so it keeps
pushing until the weight hits zero and the subgradient lets it stay there. That is why L1 produces
sparsity and L2 does not.

$|w|$ has no derivative at zero; the implementation takes the subgradient $0$, which is what allows
a weight to sit exactly at zero rather than jitter across it.

Biases and normalization parameters are never penalized. A bias shifts a response rather than scaling
it, so shrinking it does not reduce model complexity — it only biases the fit toward the origin.

## Dropout

During training each unit's output is zeroed independently with probability $p$. This prevents units
from co-adapting into fragile combinations, since no unit can rely on any particular other one being
present, and it can be read as training an ensemble of exponentially many sub-networks that share
weights.

At inference nothing is dropped, so the expected scale would differ between the two regimes.
**Inverted dropout** fixes this on the training side by scaling survivors up by $1/(1-p)$:

$$
\mathbb{E}[\text{output}] = (1-p)\cdot\frac{a}{1-p} + p\cdot 0 = a .
$$

Inference is then the plain network with no rescaling. The consequence for verification is that the
training loss is *stochastic*: the same parameters give different losses on different forward passes,
so a finite-difference gradient check is meaningless. `check_batch_gradient` refuses a network with
dropout for that reason, and dropout is verified instead by its deterministic properties — that it
never fires at inference, never changes the parameter count, and still permits training to converge.

## Batch and layer normalization

Both standardize a set of pre-activations and then restore a learnable degree of freedom:

$$
\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \varepsilon}},
\qquad
y = \gamma\hat{z} + \beta .
$$

The $\gamma$ and $\beta$ matter: without them normalization would force every layer's output into a
fixed distribution, and the network could not represent a function that needs a different one. With
them it can undo the normalization entirely if that is what minimizes the loss, so the technique
constrains the *optimization path* rather than the hypothesis space.

They differ only in the axis of the reduction.

**Batch normalization** reduces over the samples of a mini-batch, one unit at a time. This makes the
loss of any one sample depend on the other samples in its batch, which has three consequences the
implementation must handle: the forward and backward passes cannot be done a sample at a time; small
batches give noisy statistics; and inference has no batch, so training accumulates a running mean and
variance to use instead. Those running statistics are state rather than parameters — no gradient
touches them — but inference is wrong without them, so checkpoints carry them.

**Layer normalization** reduces over the units of a single sample. It involves no batch statistics at
all, so training and inference compute exactly the same function, the batch size is irrelevant, and
there is nothing extra to checkpoint.

### The backward pass

Because $\mu$ and $\sigma$ are themselves functions of $z$, the derivative has three terms: the
direct one, the shift of the mean, and the shift of the variance. Collecting them gives a single
formula, where the mean runs over whichever axis was normalized and $\hat{z}$ is the normalized value:

$$
\frac{\partial L}{\partial z_i}
= \frac{1}{\sqrt{\sigma^2+\varepsilon}}
\left(
\frac{\partial L}{\partial \hat{z}_i}
- \overline{\frac{\partial L}{\partial \hat{z}}}
- \hat{z}_i \; \overline{\frac{\partial L}{\partial \hat{z}}\hat{z}}
\right).
$$

One routine implements this for both, called with the batch as the axis for one and the layer's units
for the other. Verifying it is what gradient checking is for, and batch normalization must be checked
through a *batch-level* loss: perturbing a parameter changes every sample's loss jointly, so a
per-sample check would be measuring the wrong function.

## Early stopping

Training longer reduces training loss monotonically but not validation loss, so the epoch count is
itself a hyperparameter. Early stopping measures validation loss each epoch, keeps the parameters
from the best one, and stops after `patience` epochs without improvement. Keeping the parameters is
the part that is easy to omit and impossible to do without: by the time the curve has clearly turned,
the peak is already `patience` epochs behind.

Two honest caveats. Validation loss is a *proxy* for generalization, estimated from a finite sample,
so the selected epoch is only as good as that estimate — with a small validation split the choice can
easily be worse than simply training to the end, and the experiment measures a case where it is.
And because the stopping epoch is chosen on validation data, validation loss is no longer an unbiased
estimate of generalization; only the test split is.
