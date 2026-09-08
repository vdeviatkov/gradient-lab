# Experiment 05: PCA and k-means

This C++ experiment builds a synthetic dataset whose structure is known in advance, then checks
whether unsupervised methods recover it.

## Dataset

Three groups of 40 samples live in a two-dimensional latent plane with centers `(0, 0)`,
`(6, 0.5)`, and `(3, 5)` and a per-group spread of `0.6`. Each latent point is mapped to four
observed features by a fixed linear map,

```text
f0 = x
f1 = y
f2 = 0.8 x - 0.6 y
f3 = 0.3 x + 0.4 y
```

with independent noise of scale `0.05` added to each observed feature. The data is therefore
intrinsically two-dimensional: two of the four features are redundant up to that noise, so PCA has
real structure to find rather than an artificial one.

Samples are split by index inside each group into fixed train (24), validation (8), and test (8)
splits, giving 72 / 24 / 24 overall. Seed `20260907` drives a self-contained splitmix64 stream with
a Box-Muller transform written in the experiment, because the C++ standard distributions are not
specified bit-for-bit and the dataset should be identical on every standard library.

## Method

1. PCA is fitted on the training split only, and its full eigenvalue spectrum, explained-variance
   ratios, and total variance are reported.
2. Reconstruction error is measured on the held-out test split for ranks 1 through 4.
3. The number of clusters is chosen without labels: k-means is fitted on the training split for
   `k = 1..6`, inertia is measured on the validation split, and the elbow heuristic picks the
   interior `k` with the largest discrete curvature.
4. At the selected `k`, k-means is fitted on training data with 10 seeded restarts and evaluated
   once on the test split, both on the raw four features and on the two-dimensional PCA
   projection, and with both initializations.
5. Purity against the true group labels is computed only at evaluation time. The `k = 1` clustering
   is the baseline: its purity is the majority-class frequency and its inertia is the total scatter.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_pca_kmeans
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
two components explain at least 99% of the training variance, the elbow heuristic selects the true
group count, k-means beats the single-cluster baseline on both test inertia and test purity, and
clustering the projection does at least as well as clustering the raw features.

## Measured result

Recorded on 2026-09-07 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seed `20260907`. Debug and Release builds printed identical output at this precision:

| Component | Explained variance | Ratio | Cumulative |
|---|---:|---:|---:|
| 1 | 11.047807 | 0.608547 | 0.608547 |
| 2 | 7.101859 | 0.391192 | 0.999739 |
| 3 | 0.002634 | 0.000145 | 0.999884 |
| 4 | 0.002105 | 0.000116 | 1.000000 |

Total variance was `18.154404`. Held-out reconstruction error per feature fell from `1.836722` at
rank 1 to `0.000975` at rank 2, confirming the two-dimensional latent structure; ranks 3 and 4 gave
`0.000375` and `0.000000`.

Validation inertia over the cluster-count grid was `423.446842`, `164.539913`, `24.781671`,
`18.805875`, `16.584257`, and `15.690816` for `k = 1..6`. The elbow heuristic selected `k = 3`,
which matches the true group count.

| Clustering (test split) | Test inertia | Test purity |
|---|---:|---:|
| single-cluster baseline (`k = 1`) | 396.979729 | 0.333333 |
| k-means++ on raw features | 20.942124 | 1.000000 |
| random initialization on raw features | 20.942124 | 1.000000 |
| k-means++ on the 2-D projection | 20.848189 | 1.000000 |

Both initializations reached the same optimum here; the groups are well separated, so 10 restarts
are enough for either one. Clustering the projection was marginally cheaper in inertia because the
two discarded components carry only noise, and it lost no purity.
