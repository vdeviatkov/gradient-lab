# Decision tree classification mathematics

A classification tree partitions feature space with axis-aligned splits. Each internal node holds a
feature $j$ and a threshold $t$ and sends a sample left when $x_j \le t$; each leaf predicts the
majority class of the training samples that reached it.

## Impurity

Let a node hold $n$ samples with class counts $n_1,\dots,n_K$ and empirical class probabilities
$p_c = n_c/n$. Two impurity measures are implemented:

$$
G = 1 - \sum_{c=1}^{K} p_c^2,
\qquad
H = -\sum_{c=1}^{K} p_c \log_2 p_c .
$$

Both are zero exactly when the node is pure and maximal when the classes are uniform, where
$G = 1 - 1/K$ and $H = \log_2 K$. The term $p_c\log_2 p_c$ is taken as zero when $p_c = 0$, its
limit. Gini is the probability that two samples drawn from the node have different labels; entropy
is the average number of bits needed to encode a label. They are different scales — a balanced
binary node has $G = 0.5$ and $H = 1$ bit — so their numeric values are not comparable, only the
rankings they induce over candidate splits.

## Choosing a split

A split $(j, t)$ divides the node's $n$ samples into $n_L$ left and $n_R$ right. Its **impurity
decrease** is measured against the node's own samples:

$$
\Delta = I(\text{node}) - \frac{n_L}{n} I(\text{left}) - \frac{n_R}{n} I(\text{right}),
$$

where $I$ is $G$ or $H$. Because $I$ is concave in the class probabilities and the node's
probability vector is the $n_L/n, n_R/n$ mixture of its children's, Jensen's inequality gives
$\Delta \ge 0$: a split never increases weighted impurity. For entropy, $\Delta$ is exactly the
mutual information between the split indicator and the label, which is why it is also called
information gain.

The implementation searches every feature exhaustively. For one feature it sorts the node's samples
by value and sweeps left to right, maintaining running class counts so each candidate threshold
costs $O(K)$ rather than a fresh pass. Candidate thresholds are midpoints between consecutive
*distinct* values — a split between two equal values is impossible — giving $O(dn\log n)$ per node.
The sort breaks ties by sample index and the sweep keeps a candidate only if it strictly beats the
incumbent, so the lowest feature index and lowest threshold win a tie and the tree is a
deterministic function of the dataset.

Growth stops at a node when it is pure, when it holds fewer than `min_samples_split` samples, when
the depth limit is reached, or when no admissible split has $\Delta$ above `min_impurity_decrease`.
A split is admissible only if both children keep at least `min_samples_leaf` samples.

## Greedy search is myopic

The search is greedy: it maximizes $\Delta$ one node at a time and never looks ahead. That is not
merely suboptimal, it can fail outright. On an exact checkerboard — $(0,0)$ and $(1,1)$ labeled one
class, $(0,1)$ and $(1,0)$ the other — every single-feature split leaves both children exactly as
impure as the parent, so $\Delta = 0$ for every candidate and growth stops at the root, even though
a depth-two tree classifies all four points correctly. Finding the optimal tree is NP-hard in
general, so practical algorithms accept this greedy heuristic. A depth limit is not the remedy
here; lookahead or a change of representation is.

## Overfitting, depth limits, and pruning

Training accuracy is monotone in tree size: given distinct feature vectors, a tree grown without
limits isolates individual training points and reaches 100% training accuracy, including on
mislabeled ones. Training accuracy therefore says nothing about generalization, and the honest
measurement is a held-out split.

Two controls are implemented.

**Pre-pruning** stops growth early through `max_depth`, `min_samples_split`, `min_samples_leaf`, or
`min_impurity_decrease`. It is cheap, but it decides before seeing what a subtree would have
achieved, so a split with low immediate gain that enables a valuable one below it is lost.

**Post-pruning** grows the full tree first, then removes subtrees. The implementation uses
**reduced-error pruning**: repeatedly collapse the internal node whose replacement by a leaf does
not lower accuracy on a validation split, preferring the collapse that removes the most nodes, and
stop when no such node remains. Every node stores the majority class of its training samples, so
any subtree can become a leaf. Because each round strictly decreases the number of internal nodes
and never accepts a validation-accuracy loss, the loop terminates and the result is at least as
accurate on validation as the unpruned tree. This sees whole subtrees rather than one split at a
time, so it can remove a branch that pre-pruning had no way to anticipate.

The validation split used for pruning must not be the training split — the pruning criterion would
otherwise be the same quantity the tree already maximized, and nothing would ever be removed. It
must also not be the test split, since accuracy on data used to choose the model is no longer an
unbiased estimate.
