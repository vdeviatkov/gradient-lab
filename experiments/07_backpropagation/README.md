# Experiment 07: backpropagation and gradient checking

This C++ experiment verifies the general backpropagation implementation against finite differences,
measures how the step size affects that verification, and then trains the network on a task a
linear model cannot solve.

## Dataset

Three interleaved spiral arms, 150 samples per class. For position $p \in [0, 1]$ along an arm, a
point has radius $0.2 + 0.8p$ and angle $4p + 2\pi k/3$ for class $k$, plus Gaussian angular noise
of scale `0.14`. The classes are separable but not linearly separable, so a linear softmax model is
a meaningful baseline and hidden layers have to do real work.

Splits are **interleaved**, not index ranges: three of every five samples train, then one each for
validation and test. The index sweeps position along the arm, so a range split would give each
split a disjoint piece of the spiral and measure extrapolation instead of generalization. Seed
`20260910` drives `ml_scratch::DeterministicRandom`, and the networks are initialized from seed
`20260910`.

Gradient checks use whichever small dataset matches the loss under test: the spiral for softmax
cross-entropy, a fixed 3-sample two-output set for squared error, and the XOR truth table for
binary cross-entropy.

## Method

1. Seven architecture and loss combinations are gradient-checked at initialization with central
   differences at `epsilon = 1e-5`, against a tolerance of `1e-7` on the norm ratio.
2. The step size is swept over ten decades for both the central and the one-sided difference, to
   measure the truncation and round-off trade-off directly.
3. A known-good gradient is corrupted by 0.1% in one parameter to confirm the check catches it.
4. A linear softmax model and a `2 -> 16 -> 16 -> 3` tanh MLP are trained on the spiral and
   compared against a majority-class baseline.
5. The MLP's gradient is checked at three points on its trajectory: initialization, after 20
   epochs, and at convergence.

## Running it

```bash
cmake -S . -B build
cmake --build build
./build/cpp_backpropagation
```

The executable prints only values measured during the current run. It exits unsuccessfully unless
every initialization check passes, central differences beat one-sided differences at the same step,
the deliberate error is caught, the MLP beats the linear model, the linear model beats the majority
baseline, the checks at initialization and after 20 epochs pass, and the converged gradients still
agree to better than `1e-9` in absolute terms.

## Measured result

Recorded on 2026-09-10 with Apple clang 17.0.0 on macOS 26.3.1 (arm64), CMake build type `Release`,
seeds as above.

Every configuration matched its finite-difference estimate:

| Configuration | Parameters | Gradient-check ratio |
|---|---:|---:|
| linear, squared error | 6 | 8.292e-12 |
| tanh hidden, squared error | 27 | 1.384e-11 |
| sigmoid output, squared error | 22 | 2.690e-11 |
| binary cross-entropy | 17 | 5.408e-11 |
| softmax cross-entropy | 51 | 4.052e-10 |
| three hidden layers, mixed | 103 | 7.503e-10 |
| ReLU hidden layers | 81 | 4.897e-10 |

### The step size trade-off

| `epsilon` | Central | One-sided |
|---:|---:|---:|
| 1e-01 | 5.329e-04 | 3.373e-02 |
| 1e-02 | 5.337e-06 | 3.360e-03 |
| 1e-03 | 5.337e-08 | 3.358e-04 |
| 1e-04 | 5.336e-10 | 3.358e-05 |
| 1e-05 | **4.052e-10** | 3.358e-06 |
| 1e-06 | 3.818e-09 | 3.399e-07 |
| 1e-07 | 3.950e-08 | 1.161e-07 |
| 1e-08 | 3.350e-07 | 1.019e-06 |
| 1e-09 | 4.271e-06 | 9.562e-06 |
| 1e-10 | 3.408e-05 | 1.017e-04 |

The two orders of accuracy are visible directly: from `1e-1` down to `1e-4` the central error falls
by almost exactly 100 per decade and the one-sided error by 10, matching $O(\varepsilon^2)$ against
$O(\varepsilon)$. Below the optimum both curves turn and rise as cancellation takes over. The
measured minimum for the central difference is `4.052e-10` at `epsilon = 1e-5`, next to the
$u^{1/3} \approx 6\times10^{-6}$ predicted by balancing the two error terms.

A deliberate 0.1% error in one parameter raised the ratio to `1.051e-05`, more than a hundred times
the tolerance, and the report named parameter 7.

### Training

| Model | Parameters | Final loss | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| majority baseline | — | — | — | — | 0.3333 |
| linear softmax | 9 | 0.7943 | 0.5222 | 0.5222 | 0.5444 |
| 2x16 tanh MLP | 371 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |

The linear model beats the baseline but cannot bend around a spiral; the hidden layers take it to
perfect held-out accuracy.

### Where gradient checking stops working

| Point on the trajectory | Norm ratio | Max elementwise | Max absolute | Largest \|gradient\| |
|---|---:|---:|---:|---:|
| initialization | 7.248e-10 | 1.116e-06 | 1.078e-10 | 9.913e-02 |
| after 20 epochs | 1.748e-10 | 1.091e-07 | 5.570e-11 | 1.822e-01 |
| at convergence | 2.217e-07 | 2.915e-05 | 5.376e-12 | 1.670e-05 |

At convergence the norm ratio fails its `1e-7` tolerance, but nothing is wrong with the gradient:
the largest absolute disagreement is `5.376e-12`, smaller than at either healthy point. The whole
gradient has shrunk to `1.670e-05`, so the ratio is dividing round-off by almost nothing. This is
the degeneracy documented in the mathematics notes, and the reason gradient checks belong at
initialization or mid-training. The elementwise column shows why the norm ratio is the verdict:
even at initialization a single near-zero parameter reports `1.116e-06`, four orders worse than the
norm ratio, with no bug behind it.
