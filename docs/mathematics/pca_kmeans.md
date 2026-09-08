# PCA and k-means mathematics

Both algorithms in this milestone are unsupervised: they see only the feature matrix
$X \in \mathbb{R}^{n \times d}$ and never the labels used later for evaluation.

## Principal component analysis

PCA finds an orthonormal basis ordered by how much variance each direction carries. Start by
centering with the column mean $\mu = \frac{1}{n}\sum_i x_i$ and form the unbiased sample
covariance

$$
C = \frac{1}{n-1}\sum_{i=1}^{n} (x_i-\mu)(x_i-\mu)^T .
$$

$C$ is real, symmetric, and positive semi-definite, so it has an orthonormal eigenbasis with
non-negative eigenvalues:

$$
C v_j = \lambda_j v_j, \qquad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0 .
$$

The variance of the data projected onto a unit vector $u$ is $u^T C u$, which is maximized by
$v_1$ and takes the value $\lambda_1$. Restricting to directions orthogonal to $v_1$ and repeating
the argument gives $v_2$, and so on. Therefore $v_1,\dots,v_k$ are the $k$ directions that capture
the most variance, the **explained variance** of component $j$ is $\lambda_j$, and its
**explained-variance ratio** is $\lambda_j / \sum_{m} \lambda_m$.

Projection and reconstruction use the same basis:

$$
z = V_k^T (x-\mu), \qquad \hat{x} = \mu + V_k z ,
$$

where $V_k$ has the top $k$ eigenvectors as columns. Because $V_k$ is orthonormal, this
reconstruction minimizes the squared error among all rank-$k$ affine approximations, and the
residual equals the discarded variance $\sum_{j>k}\lambda_j$. Keeping all $d$ components is only a
change of basis, so its reconstruction error is zero up to rounding. The implementation reports
reconstruction error as a mean squared error per feature, measured on data the model was not fitted
on.

### Eigen-decomposition by Jacobi rotations

Rather than calling a linear-algebra library, the implementation diagonalizes $C$ with the cyclic
Jacobi method. Each step picks an off-diagonal entry $C_{pq}$ and applies an orthogonal rotation
$J$ in the $(p,q)$ plane chosen so that $(J^T C J)_{pq} = 0$. With

$$
\theta = \frac{C_{qq}-C_{pp}}{2C_{pq}}, \qquad
t = \frac{\operatorname{sign}(\theta)}{|\theta| + \sqrt{\theta^2+1}}, \qquad
c = \frac{1}{\sqrt{t^2+1}}, \qquad s = tc,
$$

the rotation is numerically stable even when $C_{pp}\approx C_{qq}$, where the naive
$\tan(2\phi)$ formula loses precision. Each rotation is orthogonal, so it preserves the eigenvalues
and the Frobenius norm while strictly reducing the off-diagonal norm. Sweeping over all pairs until
that norm falls below a tolerance leaves the eigenvalues on the diagonal and the accumulated
product of rotations holds the eigenvectors as its columns.

Eigenvectors are only defined up to sign, and eigenvalues sort ties arbitrarily. The implementation
therefore sorts by descending eigenvalue with the original index as a tie-break, and flips each
eigenvector so its largest-magnitude entry is positive. Without those two conventions, two runs on
the same data could report mirrored or permuted components.

## k-means

k-means partitions the samples into $k$ clusters $S_1,\dots,S_k$ with centroids $m_1,\dots,m_k$ to
minimize the **inertia**, the within-cluster sum of squares:

$$
J = \sum_{j=1}^{k} \sum_{x \in S_j} \lVert x - m_j \rVert^2 .
$$

Lloyd's algorithm alternates the two steps that each minimize $J$ while holding the other variable
fixed:

1. **Assignment.** With centroids fixed, every sample goes to its nearest centroid, since that is
   the term-by-term minimum of $J$.
2. **Update.** With assignments fixed, $\partial J/\partial m_j = -2\sum_{x\in S_j}(x-m_j) = 0$
   gives $m_j = \frac{1}{|S_j|}\sum_{x \in S_j} x$, the cluster mean.

Each step is non-increasing in $J$ and there are finitely many partitions, so the algorithm
terminates. It converges to a **local** minimum only: the objective is non-convex, and the outcome
depends on initialization. The implementation stops when the total squared centroid movement of an
iteration falls below a tolerance.

### Initialization, restarts, and empty clusters

Two initializations are implemented:

- **Random samples** draws $k$ distinct samples with a partial Fisher-Yates shuffle.
- **k-means++** picks the first centroid uniformly, then draws each remaining centroid with
  probability proportional to $D(x)^2$, the squared distance to the nearest centroid chosen so far.
  Spreading the initial centroids this way makes a poor local minimum less likely.

Because a single run can land in a bad local minimum, the implementation runs several independent
restarts and keeps the one with the lowest inertia. Restart $r$ uses seed $\text{seed}+r$, so a
configuration reproduces exactly. Uniform values are drawn directly from `std::mt19937` output
instead of through a standard distribution, whose implementation is not specified bit-for-bit, so
seeded runs match across standard libraries.

A cluster can lose every member during the assignment step, leaving its mean undefined. The
implementation reseeds such a cluster with the worst-served sample — the point farthest from its
own centroid — taken from a cluster that has more than one member. This keeps exactly $k$ centroids
and cannot increase $J$, because moving that point onto its own centroid drops its contribution to
zero.

### Choosing k, and why inertia is not enough

Inertia decreases monotonically in $k$ and reaches zero when $k = n$, so it cannot select $k$ on
its own. The experiment instead uses an elbow heuristic on held-out inertia: it picks the interior
$k$ maximizing the discrete curvature

$$
J_{k-1} - 2J_k + J_{k+1},
$$

which is largest where the curve bends most sharply. This uses no labels.

### Evaluating a clustering

Labels are used only after training, as evaluation. **Purity** assigns each cluster its majority
true label and reports the fraction of samples that agree:

$$
\text{purity} = \frac{1}{n}\sum_{j=1}^{k} \max_{c} \lvert S_j \cap C_c \rvert .
$$

Purity is not a fair measure across different $k$ — it reaches one when every sample gets its own
cluster — so it is only compared at a fixed $k$ against a baseline. The single-cluster baseline
($k=1$) puts everything in one group; its purity is the majority-class frequency, and its inertia
is the total scatter about the dataset mean. A clustering that does not beat both numbers has not
found structure.
