# Residual connections

Deeper networks can represent strictly more than shallow ones — a deep network can always emulate a
shallow one by making its extra layers the identity. Yet plain deep stacks trained by gradient
descent often perform *worse* than shallow ones, and not because they overfit: their training loss
is worse too. The problem is optimization, not capacity, and the fix is to change what the layers
have to learn.

## Why depth makes gradients hard

For a stack of layers $x_{l+1} = f_l(x_l)$, the gradient reaching layer $l$ is a product of
Jacobians:

$$
\frac{\partial L}{\partial x_l}
= \frac{\partial L}{\partial x_L}\prod_{k=l}^{L-1} \frac{\partial f_k}{\partial x_k} .
$$

A product of $L - l$ matrices behaves geometrically. If their typical singular value is below one
the gradient decays exponentially with depth and the early layers barely move; above one it grows
exponentially and training diverges. Careful initialization targets a value near one, which is why
Glorot and He scaling help — but they control the product only at initialization, and only on
average.

## The residual form

A residual layer adds its input back:

$$
y = g\big(F(x) + x\big),
$$

where $F$ is the layer's own transformation. The implementation uses the simplest such block, a
single affine map with the skip joining before the activation, $y = g(Wx + b + x)$, which is where
He et al. place the addition. The two terms must have the same shape, so the layer's input and
output sizes have to match — and for a convolution, the filter count must equal the input channel
count and the stride and padding must preserve the spatial size.

Two things change.

**What the layer has to learn.** To make the block an identity, a plain layer must fit
$F(x) = x$ exactly; a residual layer only needs $F(x) = 0$, which it reaches by driving its weights
to zero. Extra depth therefore starts from something close to a no-op and can only add, rather than
having to relearn the identity at every level. The tests assert this directly: a residual layer with
zeroed weights *is* the identity function.

**How the gradient flows.** Differentiating gives

$$
\frac{\partial y}{\partial x} = g'\cdot\left(\frac{\partial F}{\partial x} + I\right),
$$

so the backward pass has a term that is multiplied by no weight matrix at all. The product over
layers now contains an identity path, and the early-layer gradient no longer decays geometrically
by construction.

## The skip alone is not enough

That second property cuts both ways, and the experiment measures it. Expanding the product of
$(\partial F/\partial x + I)$ terms over $D$ blocks gives $2^D$ paths, and each block also adds its
input to its output in the *forward* direction, so activations compound as well. Without something
holding the scale steady, a residual stack replaces a vanishing gradient with a growing one.

The measurement is unambiguous: over 20 blocks, the first-layer gradient norm of a plain stack fell
to $2.9\times10^{-1}$ while a bare residual stack's rose to $4.0\times10^{3}$ — and the bare
residual stack's accuracy was *worse* than the plain one's, not better. An identity skip on its own
does not rescue depth. It trades one failure mode for the other.

Adding a normalization inside the block fixes it, because normalizing the pre-activation discards
whatever scale the accumulating identity path has built up. With layer normalization the first-layer
gradient norm stayed essentially flat from 2 blocks to 20 — $2.7$, $2.4$, $2.9$, $4.9$ — and it was
the only variant whose accuracy did not degrade with depth. This is why a residual block in practice
is never just a skip: the original network pairs every one with batch normalization, and the pairing
is doing real work rather than being incidental.

## Residual blocks in a convolutional network

The convolutional case is the same algebra with a shaped tensor. A block is a convolution whose
output shape matches its input, so the sum is defined elementwise, and the backward pass scatters
each output's gradient both through the kernel and directly back to the matching input position.

Where a network changes shape — a different filter count, or a pooling step — the identity is not
defined and the skip must be dropped or projected. The implementation takes the first route: skips
live inside a stage, and the shape changes happen in the stem and in the pooling layers between
stages. That keeps the identity path exactly an identity, with no parameters of its own, which is
what makes the plain and residual networks in the experiment have *identical* parameter counts. Any
difference between them is therefore attributable to the skip and not to capacity.
