# Milestone roadmap

The sequence below is the learning progression. Completion by implementation is tracked in the
[status matrix](status.md); topics marked complete there are already implemented.

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
