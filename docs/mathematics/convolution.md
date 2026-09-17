# Convolutional network mathematics

A fully connected layer gives every input its own weight. On a 28x28 image that is 784 weights per
unit, and the layer has no idea that two pixels are adjacent: permute the pixels consistently across
the dataset and it learns exactly as well. A convolutional layer builds the missing structure in.

## The convolution

A layer holds $F$ kernels, each spanning every input channel. With input $x$ of shape
$(C, H, W)$ and kernels $w$ of shape $(F, C, K, K)$:

$$
z_{f,i,j} = b_f + \sum_{c=1}^{C}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1}
w_{f,c,u,v}\; x_{c,\,si-p+u,\;sj-p+v},
$$

with stride $s$ and padding $p$, and terms whose index falls outside the input taken as zero. Strictly
this is a cross-correlation rather than a convolution — a true convolution flips the kernel — but the
kernel is learned, so the flip is absorbed into what is learned and the name has stuck.

The output extent follows from counting the window positions that fit:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1 .
$$

Padding of $p = (K-1)/2$ with $s = 1$ keeps the size unchanged, which is why odd kernels are usual.

Two properties follow directly from that formula, and they are the whole point:

**Local connectivity.** Each output depends on a $K \times K$ patch, not the entire image. Stacking
layers grows the *receptive field* — with two 3x3 layers each output sees a 5x5 region — so the
network builds large-scale structure out of small-scale parts instead of having to learn it at full
size in one step.

**Weight sharing.** The same kernel is applied at every position, so a feature learned in one part of
the image is detected everywhere. This makes the layer **equivariant** to translation: shift the
input and the feature map shifts with it, which the tests assert directly. It also decouples the
parameter count from the image size — a layer has $F \cdot (C K^2 + 1)$ parameters whether the image
is 28x28 or 280x280.

## Pooling

Pooling reduces resolution with no parameters at all. Over each window,

$$
\text{max: } y = \max_{u,v} x_{u,v},
\qquad
\text{average: } y = \frac{1}{K^2}\sum_{u,v} x_{u,v}.
$$

Downsampling turns the convolution's *equivariance* into approximate *invariance*: after pooling, a
small shift of the input often leaves the output unchanged rather than merely moving it. Max pooling
keeps the strongest response in a neighbourhood and discards where exactly it was, which is why it
tends to tolerate displacement better than averaging, and the experiment measures that difference.

## Backward passes

**Convolution.** The kernel weight $w_{f,c,u,v}$ is used at every output position, so by the chain
rule its gradient sums over all of them:

$$
\frac{\partial L}{\partial w_{f,c,u,v}} = \sum_{i,j} \delta_{f,i,j}\; x_{c,\,si-p+u,\;sj-p+v},
\qquad
\frac{\partial L}{\partial b_f} = \sum_{i,j}\delta_{f,i,j} .
$$

That sum *is* weight sharing expressed backwards, and it is the structural difference from a dense
layer, where each weight appears once. The gradient with respect to the input scatters each
$\delta$ back through the same kernel positions it read from:

$$
\frac{\partial L}{\partial x_{c,m,n}} = \sum_{f}\sum_{u,v}
\delta_{f,i,j}\, w_{f,c,u,v}
\quad\text{over every } (i,j,u,v) \text{ that read } (m,n),
$$

which is a full correlation with the flipped kernel — the transposed convolution. The implementation
writes it as a scatter during the same loop that accumulates the weight gradient, which avoids
building the flipped kernel explicitly and keeps the padding bookkeeping in one place.

**Max pooling** passes each output's gradient to the single input that produced it, so the forward
pass caches that index. Every other input in the window receives nothing: it had no effect on the
output, so it has no derivative. **Average pooling** splits the gradient equally, $1/K^2$ to each.

**Flatten** moves no values and has no gradient of its own; the layout is already contiguous and
channel-major, so it is a reinterpretation in both directions.

## Parameters are not computation

A convolution is small in parameters and large in arithmetic, and the two must not be confused. One
layer costs

$$
F \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C K^2
$$

multiply-accumulates, because every weight is reused at every output position. Weight sharing
*divides* the parameter count by the number of positions and leaves the FLOP count alone. A network
with twenty times fewer parameters than an MLP can therefore take several times longer to train,
which is exactly what the experiment measures — and it is why convolution became practical only with
hardware that could sustain that arithmetic.

A second consequence is worth stating because it is easy to get backwards: adding a convolutional
layer can *reduce* the total parameter count. Each pooling step shrinks the feature map, so the
flattened vector reaching the final dense layer gets smaller, and the dense layer is usually where
most parameters live. The experiment measures a two-convolution network with fewer parameters than a
one-convolution network for that reason.
