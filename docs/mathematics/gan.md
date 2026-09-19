# Conditional generative adversarial networks

Every earlier milestone minimized a loss that could be written down. A generative model has no
such loss: the goal is samples that look like the data, and "looks like the data" is not a
formula. A GAN's move is to *learn* the loss — a second network is trained to tell real samples
from generated ones, and the generator is trained to make that second network fail.

## The game

The generator $G$ maps a noise vector $z \sim \mathcal N(0, I)$ to a sample $G(z)$. The
discriminator $D$ maps a sample $x$ to a probability that $x$ came from the data rather than from
$G$. The two are trained against each other on

$$
\min_G \max_D \;
\mathbb E_{x \sim p_{\text{data}}}\big[\log D(x)\big] +
\mathbb E_{z}\big[\log\big(1 - D(G(z))\big)\big] .
$$

The discriminator's half is ordinary binary cross-entropy: real samples toward target one, fakes
toward target zero. For a fixed generator its optimum is

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)},
$$

and substituting $D^*$ back turns the generator's objective into $2\,\mathrm{JSD}(p_{\text{data}}
\,\|\, p_G) - \log 4$: the generator is driven to minimize the Jensen–Shannon divergence between
its distribution and the data's. At the equilibrium $p_G = p_{\text{data}}$, $D^* \equiv 1/2$ —
the discriminator can do no better than a coin flip. The implementation reports the mean $D(x)$
and $D(G(z))$ every epoch for exactly this reason: a healthy run keeps both away from 0 and 1.

## Conditioning

A conditional GAN gives both networks a class label $y$ as an additional input:

$$
G(z, y), \qquad D(x, y).
$$

The implementation concatenates a one-hot encoding of $y$ to the generator's noise and to the
discriminator's sample. The discriminator therefore judges whether the *pair* $(x, y)$ is
plausible, which forces the generator to produce an $x$ that matches the requested $y$ — the label
is what lets a caller ask for a particular digit. The mathematics is unchanged: $y$ is simply part
of both networks' inputs, and the objective becomes an expectation over $(x, y)$ pairs.

## Training the generator through the discriminator

The discriminator's gradient is the backpropagation milestone's, with the fakes treated as
constants. The generator's is new. Its loss is a function of the discriminator's output, so the
chain runs

$$
\frac{\partial L_G}{\partial \theta_G}
= \frac{\partial L_G}{\partial z_D}\cdot
  \frac{\partial z_D}{\partial x}\cdot
  \frac{\partial x}{\partial \theta_G},
\qquad x = G(z, y),
$$

where $z_D$ is the discriminator's logit. The middle factor is the discriminator's gradient with
respect to its *input* rather than its parameters — a quantity the earlier backward pass computed
implicitly on its way down and threw away at the first layer. Exposing it is the one extension
the network needed: `FeedForwardNetwork::backpropagate` now accepts an externally supplied
$\partial L/\partial(\text{output})$, optionally skips the parameter gradient, and returns
$\partial L/\partial(\text{input})$. The generator step is then two backward passes joined end to
end: through the discriminator with a null parameter buffer, and through the generator with the
discriminator's input gradient as the incoming output gradient. The label half of that input
gradient is discarded, since the label was given rather than generated.

The tests verify this composite gradient by central differences over the generator's parameters,
with the discriminator held fixed — the same check every other gradient in the repository
passes.

## Which generator loss

With $p = D(G(z)) = \sigma(z_D)$, the generator's half of the minimax objective is

$$
L_{\text{minimax}} = \log(1 - p) = -\operatorname{softplus}(z_D),
\qquad
\frac{\partial L_{\text{minimax}}}{\partial z_D} = -p .
$$

Early in training the discriminator wins easily, $p \to 0$, and this gradient vanishes precisely
when the generator most needs to move. The original paper's practical recommendation is to
maximize $\log p$ instead:

$$
L_{\text{non-saturating}} = -\log p = \operatorname{softplus}(-z_D),
\qquad
\frac{\partial L_{\text{non-saturating}}}{\partial z_D} = p - 1 .
$$

Both losses have the same fixed point, but their gradients differ by the factor $p / (1 - p)$: the
non-saturating gradient is largest when the fake is rejected, the minimax gradient is largest when
it is already accepted. The experiment trains the same networks under both and measures the
generator's gradient norm epoch by epoch; a test constructs a confidently rejecting discriminator
and requires the minimax gradient to have all but vanished while the other has not.

## Why the discriminator uses a leaky rectifier

The generator receives *only* the gradient the discriminator passes back. A ReLU unit whose
pre-activation is negative passes back nothing, so a discriminator with many dead units starves
the generator regardless of how good its verdicts are. The leaky rectifier
$\max(v, \alpha v)$ with $\alpha = 0.2$ keeps a small slope on the negative side, so every unit
always transmits some gradient. The activation was added for this milestone and is available to
every network.

The same concern rules out two features the discriminator might otherwise use. Batch
normalization couples the samples of a batch, and the generator's fakes would then be judged
relative to each other rather than to the data; dropout would make the gradient the generator
learns from stochastic. The implementation refuses both and trains sample at a time.

## Why losses are not evidence

A GAN's losses are a moving target: the discriminator's loss falls when it improves *and* when the
generator worsens, and vice versa, so neither number says whether generation is any good. The
milestone therefore treats generated digits as an evaluation problem in their own right. A
classifier is trained separately on the real images and never sees a generated one; the
generator is then scored on

- **class consistency** — the fraction of samples the classifier assigns to the class they were
  generated for, against chance at $1/10$;
- **diversity** — the mean pairwise distance between samples of one class, as a fraction of the
  real held-out images' figure, so that a generator producing one image per class scores zero;
- **coverage** — the mean distance from each real held-out image to its nearest generated image
  of the same class, so that a generator missing part of a class is penalized;
- **novelty** — the mean distance from each generated image to its nearest real *training*
  image, so that a generator that memorized the training set is caught; real held-out images
  give the yardstick for an image that is new but on the data manifold.

Two baselines calibrate the numbers. The class-mean image scores as high on consistency as a
single image can and zero on diversity; the independent-pixel sampler has full per-pixel variety
but no model of how pixels go together. The generator has to cover the data better than both.

## Mode collapse

The generator is rewarded for any sample the discriminator accepts, not for covering the data,
so nothing in the objective stops it from producing the same few convincing samples for every
noise vector. When it does, the discriminator eventually learns to reject those samples, the
generator moves to a different few, and the pair cycles without converging. The diversity ratio
above is the direct measurement, and the experiment records it every epoch rather than only at
the end, so a collapse and recovery during training would be visible. Conditioning helps by
construction — the label forces at least one mode per class — but says nothing about variety
within a class, which is what the ratio measures.
