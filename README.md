# ml-from-scratch-cpp-python

Machine learning algorithms implemented from scratch in pure Python, NumPy, PyTorch, JAX, C++,
and Triton, with experiments comparing training methods, accuracy, and performance.

## Purpose

This repository is an educational laboratory for understanding how machine-learning algorithms
work below the framework API. Each milestone starts from the mathematics, adds tests and a
reproducible experiment, and then implements equivalent behavior across six complementary
environments:

- **Pure Python** makes every operation visible and keeps the first principles dependency-free.
- **NumPy** introduces vectorized numerical programming without hiding the algorithm.
- **PyTorch** connects the same ideas to tensors, automatic differentiation, and common tooling.
- **JAX** explores functional transformations, automatic differentiation, vectorization, and JIT
  compilation.
- **Modern C++** explores explicit data structures, memory behavior, and compiled performance.
- **Triton** exposes GPU tiling, memory movement, kernel fusion, and hardware-aware optimization.

The goal is not to make six textually identical programs. It is to keep their datasets,
initialization, learning rules, stopping conditions, and metrics comparable enough to explain
meaningful differences.

## Implementation status

| Milestone | Pure Python | NumPy | PyTorch | JAX | C++ | Triton |
|---|---:|---:|---:|---:|---:|---:|
| Perceptron: AND / OR | ✅ Complete | Planned | Planned | Planned | ✅ Complete | — |
| Perceptron: XOR limitation | ✅ Demonstrated | Planned | Planned | Planned | ✅ Demonstrated | — |
| Two-layer MLP: XOR | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Linear regression | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Binary logistic regression | Planned | Planned | Planned | Planned | ✅ Complete | — |
| PCA and k-means | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Decision tree classification | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Backpropagation and gradient checking | Planned | Planned | Planned | Planned | ✅ Complete | — |
| Softmax regression: MNIST | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| MLP: MNIST digit recognition | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Optimizer comparisons | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Regularization, initialization, and normalization | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| CNN: MNIST digit recognition | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Residual network: image classification | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Conditional GAN: MNIST digit generation | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| RNN: character-level sequence modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| LSTM and GRU: character-level sequence modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Tiny Transformer: character-level language modeling | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Snake agent: Q-learning and DQN | Planned | Planned | Planned | Planned | ✅ Complete | Planned |
| Implementation benchmarks | Planned | Planned | Planned | Planned | ✅ Complete | Planned |

`—` means that a standalone Triton kernel would not add meaningful educational value for that
milestone.

## Quick start

Python 3.11 or newer is required. The core pure-Python implementation has no runtime
dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python -m ml_scratch.pure.perceptron_experiment
# Equivalent installed command:
ml-perceptron-logic-gates
```

Optional extras can be installed only when their milestones need them:

```bash
python -m pip install -e ".[numpy]"
python -m pip install -e ".[pytorch]"
python -m pip install -e ".[plotting]"
```

## Current experiments

### A perceptron and logic gates

The first experiment trains a single-layer binary perceptron on the complete truth tables for
AND, OR, and XOR. The implementation uses deterministic seeded initialization and the classic
perceptron update rule

```text
prediction = 1 if w·x + b >= 0 else 0
error      = target - prediction
w          = w + learning_rate * error * x
b          = b + learning_rate * error
```

AND and OR are **linearly separable**: one straight decision boundary can put their positive and
negative examples on opposite sides. The perceptron convergence theorem therefore applies, and
the experiment verifies all four predictions after training.

XOR is **not linearly separable**. Its positive examples `(0, 1)` and `(1, 0)` lie on opposite
corners from its negative examples `(0, 0)` and `(1, 1)`, so no single straight boundary can
separate them. The experiment intentionally stops after a fixed epoch budget and reports this
limitation; it does not claim that XOR was learned. See
[the mathematical explanation](docs/mathematics/perceptron.md) for a short proof.

Run the source-level experiment wrapper after installing the project with:

```bash
python experiments/01_perceptron_logic_gates/run.py
```

The equivalent C++ experiment is built as `cpp_perceptron_logic_gates`. Its dataset, update rule,
stopping condition, deterministic seed, and verification criteria mirror the pure-Python version.

### A two-layer MLP for XOR

The C++ milestone continues from the perceptron's expected XOR failure with a `2 → 4 → 1` network:

- two inputs;
- one hidden layer with four tanh units;
- one sigmoid output;
- binary cross-entropy loss; and
- full-batch gradient descent with explicitly derived gradients.

“Two-layer” counts the two trainable affine transformations (input-to-hidden and
hidden-to-output). The hidden nonlinearities allow the network to form multiple decision
boundaries, so it can represent XOR. Training is considered converged only when all four
predictions are correct and the measured loss reaches the configured threshold. See
[the MLP mathematics](docs/mathematics/xor_mlp.md) for the forward and backward equations.

The implementation contains the gradients needed for this concrete network. The
[backpropagation milestone](docs/mathematics/backpropagation.md) turns that focused derivation into
a general design and verifies its gradients numerically.

### Linear regression

The C++ linear regression milestone implements a reusable multivariate model and four fitting
strategies: the normal-equation solution, batch gradient descent, stochastic gradient descent,
and mini-batch gradient descent. The closed-form solver uses pivoted Gaussian elimination written
with standard-library containers; the iterative solvers use explicitly derived gradients and
zero initialization. Seeded shuffling makes SGD and mini-batch runs reproducible.

The experiment fits a small exact line, evaluates on a fixed held-out set, and checks each
optimizer against the closed-form coefficients. See the
[linear regression mathematics](docs/mathematics/linear_regression.md) and the
[reproducible experiment](experiments/03_linear_regression/README.md).

### Binary logistic regression

The C++ binary logistic regression model learns probabilities with explicitly implemented
mini-batch gradients and numerically stable binary cross-entropy. It supports a positive-class
weight, seeded shuffling, configurable loss-based stopping, and multivariate inputs. Separate
evaluation utilities calculate a confusion matrix, accuracy, precision, recall, and F1.

The deterministic experiment uses fixed train, validation, and test splits with an imbalanced,
overlapping dataset. It chooses a threshold by validation F1 and compares the frozen threshold on
test data against both `0.5` and an always-negative majority baseline. See the
[logistic regression mathematics](docs/mathematics/logistic_regression.md) and the
[reproducible experiment](experiments/04_logistic_regression/README.md).

### PCA and k-means

The C++ unsupervised milestone implements both algorithms without a linear-algebra dependency. PCA
builds the unbiased sample covariance matrix and diagonalizes it with cyclic Jacobi rotations,
then reports explained variance, explained-variance ratios, and held-out reconstruction error for
each rank. Deterministic conventions — descending eigenvalues with an index tie-break and a fixed
eigenvector sign — make repeated runs comparable.

k-means implements Lloyd's algorithm with both random-sample and k-means++ initialization, seeded
restarts that keep the lowest-inertia run, and explicit empty-cluster handling. Because inertia
falls monotonically with the cluster count, the experiment selects `k` from held-out inertia with
an elbow heuristic and never from labels. Purity against the known groups is computed only at
evaluation time and compared against a single-cluster baseline. See the
[PCA and k-means mathematics](docs/mathematics/pca_kmeans.md) and the
[reproducible experiment](experiments/05_pca_kmeans/README.md).

### Decision tree classification

The C++ decision tree grows axis-aligned splits by exhaustive search over features and midpoint
thresholds, scoring each candidate by impurity decrease under either Gini impurity or entropy.
Sorting with an index tie-break and keeping only strictly better candidates makes the tree a
deterministic function of the dataset. Multiclass labels, `max_depth`, `min_samples_split`,
`min_samples_leaf`, and `min_impurity_decrease` are all supported, and reduced-error post-pruning
collapses subtrees against a validation split.

Growth is greedy, which the tests record as a real limitation rather than a rough edge: on an exact
checkerboard every single-feature split leaves impurity unchanged, so the search stops at the root
even though a depth-two tree would be perfect. The experiment measures the overfitting story
directly — training accuracy climbing to 100% while validation accuracy peaks early and then
falls — instead of reporting training accuracy alone. See the
[decision tree mathematics](docs/mathematics/decision_tree.md) and the
[reproducible experiment](experiments/06_decision_tree/README.md).

### Backpropagation and gradient checking

The C++ backpropagation milestone turns the XOR network's hand-derived gradients into a general
`FeedForwardNetwork`: any stack of fully connected layers, identity, sigmoid, tanh, or ReLU
activations, and squared error, binary cross-entropy, or softmax cross-entropy. The cross-entropy
losses apply their own output transform, so the final layer stays linear, the output delta
collapses to `p - y`, and the loss can be written in the overflow-free softplus and log-sum-exp
forms.

Every parameter is exposed as one flat vector, which is what makes the verification general: any
architecture becomes a function from a parameter vector to a scalar, and the same central-difference
loop checks all of them. The verdict uses the norm ratio rather than a per-element one, and the
experiment measures where the method itself breaks down — the step-size trade-off between
truncation and round-off, and the degeneracy at a converged minimum where every gradient entry
approaches zero. See the
[backpropagation mathematics](docs/mathematics/backpropagation.md) and the
[reproducible experiment](experiments/07_backpropagation/README.md).

### Softmax regression on MNIST

The first milestone on a real dataset. It adds an IDX reader written from scratch — big-endian
headers decoded byte by byte, with explicit rejection of unsupported element types, truncated
payloads, and mismatched image and label counts — and a `SoftmaxRegression` model with stable
log-sum-exp cross-entropy, optional L2 on the weights, and macro-averaged multiclass metrics.

The model computes the same function as a single linear layer of the backpropagation milestone's
network under softmax cross-entropy, and a test requires the two implementations to agree on the
loss, the gradient, and the predicted probabilities. It exists separately because it indexes a
dataset of class indices in place instead of materializing one-hot targets and per-batch copies,
which is what makes the full 60000-image training set practical.

MNIST is not committed; `./scripts/download_mnist.sh` fetches it into the ignored `data/`
directory. The experiment measures where a linear model stops: it reaches 92.21% test accuracy and
its errors are structured — 5 read as 3, 4 as 9 — which is all a per-pixel linear vote can see. See
the [softmax regression mathematics](docs/mathematics/softmax_regression.md) and the
[reproducible experiment](experiments/08_softmax_mnist/README.md).

### An MLP on MNIST

The same `FeedForwardNetwork` from the backpropagation milestone, applied to real data. Three
things were added to make that practical and honest: training overloads that take class indices
instead of one-hot targets and index the dataset in place, so a 54000-image epoch copies nothing
and allocates nothing; text checkpoints that round-trip every parameter exactly at 17 significant
digits; and a confusion matrix on the class-index path. Tests require the class-index and one-hot
paths to produce identical losses, gradients, and training trajectories.

The experiment reuses experiment 08's data, preprocessing, splits, batch size, and seed, and
retrains the linear baseline inside the same run so the comparison is measured rather than quoted.
Validation accuracy is measured after every epoch and the best parameters are checkpointed, then
reloaded from disk to produce the evaluated model. Test accuracy rises from 92.21% to 98.14%, a 76%
reduction in error rate, and the linear model's structured confusions fall furthest: 5 read as 3
drops from 53 cases to 6. See [what a hidden layer adds](docs/mathematics/mlp_mnist.md) and the
[reproducible experiment](experiments/09_mlp_mnist/README.md).

### Optimizer comparisons

An `Optimizer` that turns a gradient into an update: plain gradient descent, momentum, Nesterov
momentum, RMSProp, and Adam. It owns only its own state and hyperparameters and receives the
learning rate per step, so a later milestone can add a schedule without touching the update rules.
`NetworkTrainingConfig` gained an `OptimizerConfig` whose default is plain gradient descent, and the
backpropagation experiment's output is byte-identical before and after the change.

The tests check each rule against its recursion by hand rather than only that training improves:
momentum's geometric series and its asymptotic step, Nesterov braking sooner on a reversed gradient,
RMSProp equalizing steps across gradients four orders of magnitude apart, and Adam's bias correction
producing a first step of about the learning rate even for a gradient of `1e-6`. One test records a
consequence that is easy to overlook — RMSProp at a fixed learning rate does not converge on
`f(x) = x²/2` but settles into a limit cycle at exactly `lr/2`, which is the concrete reason
schedules exist.

The experiment separates the batch-size axis from the update-rule axis, gives each rule a rate that
suits it, and reports update counts and wall-clock time next to the losses. Two of its findings cut
against expectation: raising the full-batch learning rate made the loss *worse* rather than
recovering the missing updates, and across a four-decade sweep plain gradient descent was the
*least* sensitive rule — Adam is flatter only inside the decade it is normally run in. See the
[optimizer mathematics](docs/mathematics/optimizers.md) and the
[reproducible experiment](experiments/10_optimizers/README.md).

### Initialization, regularization, and normalization

`FeedForwardNetwork` gained selectable weight initialization (Glorot, He, a fixed scale, or zeros),
L1 and L2 penalties on the weights, inverted dropout, early stopping with best-parameter restore,
and both batch and layer normalization. Everything is off by default, and the backpropagation
experiment's output is byte-identical before and after the change.

Batch normalization makes one sample's loss depend on every other sample in its batch, so it needs a
batch-at-a-time forward and backward pass; the sample-at-a-time path is kept for everything else,
and a test requires the two to agree where both apply. Its backward pass and layer normalization's
share one routine — the same formula with the reduction taken over a different axis — and both are
verified by gradient checking, batch normalization through a batch-level loss because a per-sample
check would be differentiating the wrong function. Dropout makes the loss stochastic, so gradient
checking refuses a network that uses it and it is verified by its deterministic properties instead.

Two of the experiment's results cut against the usual story: no weight penalty recovered much of an
8.5-point generalization gap, because that gap comes from having 2000 training images rather than
from oversized weights; and early stopping's restored epoch was marginally *worse* on test than
training to the end, a compute saving rather than an accuracy win. The clearest effect is
normalization's: at a learning rate that collapsed a plain network to 38% accuracy, the
batch-normalized one scored 89%. See the
[mathematics](docs/mathematics/regularization.md) and the
[reproducible experiment](experiments/11_regularization/README.md).

### A convolutional network on MNIST

A `ConvolutionalNetwork` built from convolution, max and average pooling, flatten, and dense layers,
with explicitly derived gradients for each. Activations flow as flat vectors with an explicit shape
rather than nested tensors, and every layer's output shape is derived at construction, so an
impossible architecture is rejected before training. It is a separate class from
`FeedForwardNetwork` — shaped activations and several layer kinds — but shares its activations,
losses, optimizers, and flat-parameter convention, so the same gradient checker verifies both.

Every layer kind's backward pass is verified against central differences, including stride, padding,
both poolings, and stacked convolutions. Weight sharing is tested as a property rather than assumed:
the same feature placed at two positions must produce the same response, translated.

The experiment holds the optimizer, schedule, and data fixed so only architecture varies. The CNN
reached 96.88% test accuracy with 5258 parameters against an MLP's 94.20% with 101770 — 19.4x fewer
— and lost half as much accuracy when the test digits were shifted two pixels. Two results are worth
the read: max and average pooling were indistinguishable on centered data yet differed by 4.3 points
once digits moved, and the CNN cost 4.3x the MLP's training time despite the parameter gap, because
weight sharing divides parameters and leaves arithmetic alone. See the
[convolution mathematics](docs/mathematics/convolution.md) and the
[reproducible experiment](experiments/12_cnn_mnist/README.md).

### Residual connections

Both networks gained an optional identity skip: a layer computes `g(Wx + b + x)` instead of
`g(Wx + b)`, with the addition placed before the activation as in the original residual network.
The two terms must have the same shape, so a residual convolution's filter count must match its
input channels and its stride and padding must preserve the spatial size. Defaults are unchanged and
the backpropagation experiment's output is byte-identical. A CIFAR-10 binary reader was added
alongside the MNIST one, tested against files the test writes itself.

The experiment isolates the skip in an MLP before introducing convolutional blocks, and the
measurement is not the expected one. A plain 20-layer stack loses most of its first-layer gradient
and degrades with depth, as advertised — but a **bare** residual stack does not fix it. Its
first-layer gradient *explodes* to `4.0e+03`, four orders of magnitude above the plain stack's, and
its accuracy at depth 20 is worse than plain. The skip trades a vanishing gradient for a growing
one, because each block adds its input back and the activations compound. Only with a normalization
inside the block does the picture invert: the gradient norm stays essentially flat from 2 blocks to
20, and it becomes the only variant that does not degrade. That is why a residual block in practice
is never just a skip. See the [residual mathematics](docs/mathematics/residual.md) and the
[reproducible experiment](experiments/13_residual/README.md).

### A conditional GAN on MNIST

The first milestone without a loss that can be written down. `ConditionalGan` pairs two
`FeedForwardNetwork`s: a generator from noise plus a one-hot label to an image, and a discriminator
from an image plus the label to one logit. The discriminator trains with ordinary binary
cross-entropy; the generator trains *through* the discriminator, which needed one extension of the
backward pass — it now accepts an externally supplied output gradient, can skip the parameter
gradient, and returns the gradient with respect to its input. Two backward passes joined end to end
are the generator's step, and the composite gradient is verified by central differences with the
discriminator held fixed. A leaky rectifier was added so no discriminator unit can stop passing
gradient to the generator, and checkpoints can now be written to a stream so both networks share a
file.

Losses are not evidence in a GAN, so the experiment scores generated digits with a separately
trained classifier and with distance-based diversity, coverage, and novelty measures, against a
class-mean and an independent-pixel baseline. The trained generator's samples are judged as the
requested digit 94.2% of the time, from exactly chance untrained, with within-class diversity at
0.88 of the real images' and samples farther from their nearest training image than a genuinely new
digit is. Three findings are worth the read: the two losses barely move while consistency climbs
from 0.19 to 0.94, which is the reason a separate judge exists; the generator learns variety long
before it learns the label, and then gives some of it back as it commits to a class; and the
minimax loss's gradient is 4.3x smaller than the non-saturating loss's at every epoch yet the
generator still learns, because Adam normalizes the size away — the saturation is real in the
gradient and mostly hidden in the parameters. See the
[GAN mathematics](docs/mathematics/gan.md) and the
[reproducible experiment](experiments/14_cgan_mnist/README.md).

### A character-level recurrent network

The first sequence model. `CharRnn` is a tanh recurrent network with explicitly derived
backpropagation through time, truncated to a window whose hidden state carries across windows,
global-norm gradient clipping that reports the fraction of updates it touched and the largest norm
it saw, seeded sampling at a temperature, and a `gradient_reach` measurement — the norm of one
prediction's gradient with respect to the hidden state each step earlier. A `TextCorpus` fixes the
vocabulary and contiguous splits that the LSTM, GRU, and Transformer milestones will share, and
count-based n-gram models with additive smoothing set the floor any sequence model has to clear.

On the first 500000 characters of Tiny Shakespeare the network reaches 1.96 nats per character on
test, below the trigram's 2.15 with an eighth of its parameters. Three measurements are the
substance. The gradient reaching twenty steps back is 0.6% of what reaches the last state — and
training made that *worse*, from 27% untrained, because a trained network saturates its units. A
5-step truncation window beats the 50-step one at equal epochs, because it makes ten times the
updates, while the 50-step window wins at equal update counts; a 1-step window is worst of all. And
clipping never fired under Adam yet decided everything under plain gradient descent, where the
unclipped run's gradient norm hit 37 and its loss ended worse than a uniform guess while clipping at
1 trained cleanly. See the [RNN mathematics](docs/mathematics/rnn.md) and the
[reproducible experiment](experiments/15_char_rnn/README.md).

### LSTM and GRU cells

`CharRnn` gained a `RecurrentCell`: the elman cell, an LSTM with its additively updated cell
state and a forget bias of one, and a GRU. The output layer, loss, truncation, carried state,
clipping, sampling, checkpoints, and outer BPTT loop are shared; only the step forward and its
backward differ, and both gated backward passes are checked against central differences. The
elman path is unchanged and experiment 15's output is byte-identical.

Three comparisons on the same corpus give three different winners, and the README explains why
that is not a contradiction. At equal parameters the GRU beats the elman cell on test (1.919
against 1.964) and the LSTM loses to it (2.051), because 58 units is the price of four gates and
Shakespeare's dependencies are mostly short. At equal width both gated cells win, at 2.5–3.4 times
the cost. On a delayed-copy stream that genuinely needs memory, the elman cell learns a six-step
copy and not a twelve-step one; both gated cells learn twelve, and only the LSTM learns twenty.
Gradient reach, measured on each trained network, is the mechanism and is *learned*, not
architectural: on text the trained LSTM carries less gradient twenty steps back than the elman
cell, while on the copy stream the same cell carries 146% of it to the source bit — the gradient
grows along the cell path. See the [LSTM and GRU mathematics](docs/mathematics/lstm_gru.md) and
the [reproducible experiment](experiments/16_lstm_gru/README.md).

### A decoder-only Transformer

`CharTransformer`: token embeddings plus a sinusoidal positional encoding, pre-normalized blocks of
causal multi-head self-attention and a feed-forward layer with residual connections, a final
normalization, and logits over the next character, with an optional learned relative position
bias on the attention scores. The backward pass — softmax and bilinear score included — is derived
by hand and checked against central differences; a test shows one block is exactly
permutation-invariant over the past without positions and that a second block leaks order through
the causal mask. Training draws random windows; evaluation reads text in abutting or strided
windows, and `attention_weights` exposes what each head looks at.

On the shared corpus the 107711-parameter Transformer is ahead of both recurrent networks at
every epoch after the first and beats the elman cell on test after 8 epochs (1.935 against 1.964
after 12), but not the 70-unit GRU (1.919), at 3.3 times the parameters and 3.8 times the time per
character; at the recurrent networks' own parameter count it is behind both. Six of its eight
heads attend within three characters and one looks across the whole window. The ablations say
what it needs: without positions the loss is worse by 0.33 nats, a 512-parameter relative
position bias is the single most valuable change, and at width 32 one head beat four and a
16-character context beat 64. On a twenty-step copy stream that the elman cell scores chance on,
the Transformer reaches the floor with a head putting 81% of its weight on the source bit — and
cannot without positional information. Getting there required marking the stream, because a
sinusoid has no period-two component from which a stateless model can read parity; the reason is
in the docs. See the [Transformer mathematics](docs/mathematics/transformer.md) and the
[reproducible experiment](experiments/17_transformer/README.md).

### Reinforcement learning on Snake

The first milestone without labels. A Snake environment written from scratch — three relative
actions, seeded food, and death distinguished from truncation so a cut-off episode is not treated
as terminal — with `TabularQLearning`, a `ReplayBuffer`, and `DeepQLearning`: the same Bellman
target, once into a table and once into a network whose untaken actions get no gradient. Every
agent is scored by greedy rollouts on fixed held-out seeds, because an epsilon-greedy score
measures the policy plus its noise.

Both learners beat a random policy eighty-fold. Three results are worth the read. A four-line
scripted food-seeker scores 17.55 and beats the tabular agent's 16.32 after 3000 episodes,
because the eleven-feature observation was built to contain what a greedy food-seeker needs — the
learners have to earn their keep elsewhere. Only **256 of the 2048** observation bit patterns can
ever occur, since the heading is one-hot and the food cannot be both left and right, and that
makes the table competitive: at 30000 episodes it scores 19.71 in 0.85 s against the network's
18.86 in 47.7 s, 56 times cheaper, while the network is far more episode-efficient (18.6 after
600 episodes, a score the table does not reach in 3000). And of the two mechanisms that are
supposed to make a network trainable here, only replay did anything measurable: without it the
run's last six evaluations swing six points between neighbours (16.8, 20.5, 15.8, 14.5, 20.6,
14.7) against the full agent's two and a half; removing the target network or the exploration
schedule changed nothing outside the noise, and the README says why. See the
[reinforcement learning mathematics](docs/mathematics/reinforcement_learning.md) and the
[reproducible experiment](experiments/18_snake_rl/README.md).

### Implementation benchmarks

A benchmarking harness — auto-tuned iteration counts, a cold first call reported separately from
the steady state, seven repetitions summarized by their median, a reset callback outside the timed
region, `do_not_optimize`, measured clock resolution, captured environment, peak resident memory,
and JSON output that keeps every repetition — applied to the repository's own implementations.
`scripts/benchmark_build.sh` times a clean build, which is the one cost a running program cannot
measure about itself.

The numbers put costs on what earlier milestones asserted. The gated cells cost 2.4x (GRU) and
3.2x (LSTM) an elman cell of the same width, close to their gate counts. One tabular Q-learning
update takes **1.9 ns** against 178 µs for one DQN batch — 93000x — and a Snake step costs 37 ns,
so a 150-step episode is 5.6 µs of environment against 26.7 ms of learning. The three fastest
benchmarks have the largest cold ratios, up to **88x**, which is the case for warm-up stated as a
measurement. Two findings cut against expectation: reusing the forward pass's scratch space
instead of allocating it saves 0.9% against a 0.1% spread, because at this size the arithmetic
dominates the allocator entirely; and the CNN forward pass, alone among the fourteen, falls into
two clusters a factor of 2.3 apart between runs of the same binary — neither code alignment nor
machine load explains it, and the README says so rather than quoting a number it cannot stand
behind. See the [reproducible experiment](experiments/19_benchmarks/README.md).

## Testing and quality checks

```bash
pytest
ruff check .
```

Python tests cover the truth-table datasets, predictions, successful AND/OR training, the expected
XOR failure, input validation, and seed reproducibility. CTest covers the same C++ perceptron
behavior, deterministic model parameters and histories, MLP convergence on every XOR example,
all four linear-regression solvers, optimizer agreement, regression metrics and edge cases,
logistic-regression convergence, numerical stability, threshold selection, classification
metrics, class weighting and imbalance behavior, PCA covariance and eigenpair identities,
projection and reconstruction properties, k-means recovery of known clusters, inertia monotonicity,
restart and seed reproducibility, empty-cluster and duplicate-point handling, decision-tree
impurity measures, split selection and tie-breaking, depth and leaf constraints, the greedy
search's checkerboard failure, reduced-error pruning and node compaction, hand-computed neural
network forward passes, losses and gradients, gradient checks across six architecture and loss
combinations, detection of deliberately corrupted gradients, softmax cross-entropy gradients
checked by hand and against central differences, agreement between the softmax model and the
general network, IDX parsing including malformed files, multiclass metrics, agreement between the
class-index and one-hot training paths, checkpoint round trips and rejection of corrupted
checkpoints, each optimizer's update recursion verified by hand, Adam's bias correction, RMSProp's
fixed-rate limit cycle, the adaptive rules on an ill-conditioned quadratic, initialization scales
and the identical gradients zero initialization produces, L1 and L2 penalty values and their
gradients, gradient checks for both normalizations including batch normalization's batch-level
training path, dropout's determinism at inference, early stopping restoring its best epoch, and the
convolution and pooling output shapes, hand-computed convolution and pooling values, gradient
checks for every convolutional layer kind including stride and padding, translation equivariance
from weight sharing, convolutional checkpoint round trips, a zeroed residual layer being exactly the identity, gradient
checks through residual dense and convolutional blocks, the first-layer gradient a deep skip
preserves, CIFAR-10 binary parsing including malformed records, the leaky rectifier's values and
gradient, input gradients and externally supplied output gradients checked against central
differences, two networks sharing one checkpoint stream, the GAN's generator gradient through the
discriminator and the discriminator's own gradient checked against central differences under both
generator losses, the minimax gradient vanishing against a confident discriminator, adversarial
training recovering a two-class toy distribution without collapsing, GAN seed reproducibility and
checkpoint round trips, text corpus vocabulary, contiguous splits and prefix loading, n-gram
probabilities and cross-entropies by hand, a hand-computed recurrent forward pass, backpropagation
through time checked against central differences from the zero state and from a carried state,
gradient reach decaying under a contractive recurrence, gradient clipping's scaling and reporting,
a recurrent network learning a periodic sequence and generating its continuation, the truncation
window blocking a dependency longer than itself on a delayed-copy stream, hand-computed LSTM and
GRU steps, both gated cells' backpropagation through time checked against central differences
from the zero state and from a carried state with the carried (h, c) pair verified as a complete
state, recurrent seed reproducibility and checkpoint round trips, the Transformer's parameter
layout, causal masking and attention rows summing to one, a single block's permutation invariance
without positions and a second block leaking order, the full Transformer gradient checked against
central differences with and without positions and with the relative bias, the loss as the mean
of each prefix's prediction, abutting and strided evaluation, training on a periodic sequence and
generating past the context, attention learning a twenty-step copy with a head on the source bit,
Transformer seed reproducibility and checkpoint round trips, the Snake environment's movement,
turning, death, truncation without a death penalty, growth on eating, the eleven-bit observation
and its packed index, seeded food sequences and rendering, the linear exploration schedule, the
tabular Q-learning update by hand including a terminal transition's missing future, seeded
exploration, the replay buffer's overwriting and uniform sampling, a network learning per-action
values, reinforcement seed reproducibility and checkpoint round trips, greedy evaluation on fixed
seeds distinguishing a dying policy from a circling one, the benchmark harness's statistics by
hand, its measurement of a known sleep, auto-tuning and exact call counts, a reset callback that
runs outside the timed region, `do_not_optimize` keeping work a compiler would otherwise delete,
clock and environment capture, source-size counting, JSON output shape and escaping, and the
project smoke test.

## C++ build

The C++20 project is standard-library-only. It builds a reusable `ml_scratch_cpp` library, nineteen
experiment executables, and CTest executables without downloading a testing framework. The tests
never need a downloaded dataset: the IDX reader is exercised against small files the test writes
itself.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

./build/cpp_perceptron_logic_gates
./build/cpp_xor_mlp
./build/cpp_linear_regression
./build/cpp_logistic_regression
./build/cpp_pca_kmeans
./build/cpp_decision_tree
./build/cpp_backpropagation

# Needs MNIST; see scripts/download_mnist.sh
./build/cpp_softmax_mnist
./build/cpp_mlp_mnist
./build/cpp_optimizers
./build/cpp_regularization
./build/cpp_cnn_mnist
./build/cpp_cgan_mnist

# Needs MNIST and CIFAR-10; see scripts/download_cifar10.sh
./build/cpp_resnet_cifar10

# Needs Tiny Shakespeare; see scripts/download_tiny_shakespeare.sh
./build/cpp_char_rnn
./build/cpp_lstm_gru
./build/cpp_transformer

# Needs no data
./build/cpp_snake_rl
ML_SCRATCH_BUILD_TYPE=Release ./build/cpp_benchmarks
```

## Planned roadmap

1. Perceptron on AND/OR and XOR failure
2. Two-layer MLP for XOR
3. Linear regression using the closed-form solution, batch gradient descent, SGD, and mini-batches
4. Binary logistic regression with threshold selection, confusion matrices, precision, recall, F1,
   and class-imbalance tests
5. PCA for dimensionality reduction and k-means for unsupervised clustering
6. Decision tree classification using Gini impurity and entropy, with depth limits and pruning
7. Manual backpropagation and numerical gradient checking
8. Softmax regression on MNIST
9. MLP for handwritten-digit recognition on MNIST
10. Optimizer comparisons: batch GD, SGD, mini-batch SGD, Momentum, RMSProp, and Adam
11. Weight initialization, L1/L2 regularization, dropout, early stopping, batch normalization, and
   layer normalization
12. Convolutional neural network for handwritten-digit recognition on MNIST
13. Residual connections in an MLP, followed by a small ResNet image classifier on CIFAR-10
14. Conditional GAN for controllable MNIST digit generation, including investigations of training
    instability and mode collapse
15. Character-level RNN with unrolled backpropagation through time and gradient clipping
16. Character-level LSTM and GRU models for longer-term dependencies
17. Tiny decoder-only Transformer with positional encoding, causal masking, and multi-head
    attention
18. Snake environment with a random baseline, tabular Q-learning, and DQN using experience replay,
    a target network, and a documented exploration schedule
19. Reproducible Python, NumPy, PyTorch, JAX, C++, and Triton benchmarks

The classical-ML milestones should use small, inspectable datasets and make preprocessing part of
the experiment. Linear regression should compare optimization against a closed-form reference.
PCA should report measured explained variance, while k-means should document initialization,
inertia, and multiple seeded restarts. Decision trees should demonstrate how depth control or
pruning changes overfitting rather than reporting training accuracy alone.

The RNN and Transformer milestones should share a small fixed text corpus, vocabulary, data split,
and next-character objective so recurrence and self-attention can be compared fairly. The residual
milestone should first isolate an identity skip connection in a small MLP before introducing
convolutional residual blocks. Sequence reports may add cross-entropy, perplexity, parameter count,
and tokens per second, but only when those values have actually been measured.

The GAN milestone should treat generated digits as an evaluation problem, not merely publish a
sample grid. It should compare against a simple baseline and report measured class consistency,
diversity, and coverage using a separately trained classifier or another documented evaluation
method. Generator and discriminator losses alone are not evidence that useful generation was
learned.

Every milestone should use these requirements where applicable:

- a shared dataset representation and behavior contract across implementations;
- fixed train, validation, and test splits;
- deterministic seeds and versioned experiment configuration;
- numerical-stability and edge-case tests;
- a simple baseline that establishes whether learning improved anything;
- checkpoint save/load tests for models that require meaningful training time;
- cross-implementation agreement tests with documented numerical tolerances;
- parameter-count and environment metadata alongside measured results;
- separate compilation, warm-up, and steady-state benchmark timings;
- CPU correctness checks in normal CI and hardware-gated accelerator tests; and
- evaluation that is separate from training, including exploration-free evaluation for
  reinforcement-learning agents.

A status-table cell becomes complete only when its mathematics are documented, the implementation
has automated tests, a reproducible command runs the experiment, and any published result was
actually measured. Serialization and accelerator benchmarks are required only when they apply to
that milestone.

Comparisons will consider **accuracy, convergence, execution time, memory usage, and
implementation complexity**. Each comparison should document its environment, dataset split,
seed, configuration, and measurement method.

## Results policy

Only results produced by an actually executed, reproducible experiment may be published. Never
invent benchmark numbers, fill a table with estimates, or present an unmeasured claim as a result.
Generated figures and benchmark reports belong under `results/` with enough metadata to reproduce
them; large datasets, trained model artifacts, and build outputs do not belong in Git.

## License

This project is available under the [MIT License](LICENSE).
