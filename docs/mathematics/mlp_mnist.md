# What a hidden layer adds

The gradients used here are the general ones derived in
[the backpropagation notes](backpropagation.md). This note is about what changes when a hidden
layer is placed in front of the softmax, and why that change shows up where it does on MNIST.

## From one template per class to learned features

Softmax regression scores a digit with $z_k = w_k^T x + b_k$. Each $w_k$ is a single
784-dimensional template, and the score is an inner product: a weighted per-pixel vote. Two images
that overlap heavily in pixel space therefore receive similar scores no matter what shape the ink
forms, which is why the linear model's errors concentrate on digit pairs whose strokes occupy the
same region.

A hidden layer replaces the raw pixels with $h = g(W^{(1)} x + b^{(1)})$ and scores
$z = W^{(2)} h + b^{(2)}$. Each hidden unit is its own template with its own threshold, and the
nonlinearity is what makes the composition more than another linear map: without $g$, the product
$W^{(2)}W^{(1)}$ would collapse back to a single linear layer of rank at most the hidden width, so
the model would gain parameters and no expressiveness at all. With $g$, the output layer scores
*combinations of detected features* rather than pixels, and a class can be recognized by the
presence of one feature together with the absence of another — something no single template can
encode.

The universal approximation theorem says that one hidden layer of sufficient width can approximate
any continuous function on a compact set to arbitrary accuracy. That result is about existence, not
about what gradient descent finds or how wide "sufficient" is, so it justifies looking for an
improvement rather than predicting its size. The measured improvement is what the experiment
reports.

## Initialization scale

Zero initialization, which is safe for softmax regression, is fatal here: every hidden unit would
compute the same function, receive the same gradient, and stay identical forever. The symmetry has
to be broken randomly.

The scale matters as much as the randomness. Signal variance is multiplied by roughly
$\text{fan\_in} \cdot \operatorname{Var}(w)$ at each layer, so weights that are too small shrink
activations and gradients toward zero through depth, and weights that are too large saturate
squashing activations or explode. The implementation draws uniformly from $[-l, l]$ with

$$
l_{\text{Glorot}} = \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}},
\qquad
l_{\text{He}} = \sqrt{\frac{6}{\text{fan\_in}}} .
$$

Glorot balances the forward and backward variance for activations that are symmetric about zero.
ReLU zeroes half its inputs and so halves the variance passed forward, and the He scale is the
factor-of-two correction that compensates — which is why the implementation selects it for ReLU
layers and Glorot for the rest.

## Capacity, early stopping, and checkpoints

A network with 101770 parameters fitted on 54000 images has more parameters than examples, and its
training loss falls essentially to zero: it can memorize the training split. Training loss therefore
stops being informative long before training ends, and the validation curve is the only signal that
says when to stop.

Early stopping treats the epoch count itself as a hyperparameter chosen on validation data. It costs
nothing extra to measure, and it needs a checkpoint: by the time the validation curve has clearly
turned, the parameters that achieved the peak are already gone. Saving the parameters whenever
validation improves, and restoring them at the end, is what makes the selected epoch usable rather
than merely identified. The checkpoint stores 17 significant digits per value, which round-trips an
IEEE-754 double exactly, so the restored model is the same function and not an approximation of it.

Because the stopping epoch is chosen on validation data, validation accuracy is no longer an
unbiased estimate of generalization — it has been optimized against. The test split is what remains
unbiased, and it is evaluated once.

## Reading the result

Accuracy alone compresses the comparison to two numbers. The confusion matrix says whether the
hidden layer fixed the specific thing that limited the linear model: the pairs whose ink overlaps
in pixel space. If those cells fall by roughly the same factor as the overall error rate, the
improvement is uniform; if they fall further, the hidden layer is doing exactly the work the linear
model could not. The experiment reports both so the claim is measured rather than asserted.
