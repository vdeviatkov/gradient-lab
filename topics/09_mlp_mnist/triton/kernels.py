"""The two kernels a hidden layer adds to topic 08's softmax kernels, in float64.

A training step for layers 0..L-1 is:

* `dense_rows` for every hidden layer: gather the batch rows (the first layer scales uint8 pixels
  by 1/255 in registers), compute X W^T + b over feature tiles, and apply the activation, in one
  pass. Only the activation is stored; every derivative below is written through it.
* topic 08's `softmax_rows` for the output layer: loss, prediction, and dL/dz = p - y.
* `hidden_delta` walking back down: delta_prev = (delta W) * g'(a), using each W before any
  update touches it.
* topic 08's `reduce_and_update` for every layer: (delta^T X) / B and the SGD step, fused, one
  owner per weight tile, so there are no atomics and every result is deterministic.

Activations are compile-time codes in the C++ enum order: 0 identity, 1 sigmoid, 2 tanh, 3 ReLU.
Loop bounds are constexpr for the reasons given in topic 08's kernels.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

IDENTITY, SIGMOID, TANH, RELU = 0, 1, 2, 3


@triton.jit
def _activate(z, ACTIVATION: tl.constexpr):
    if ACTIVATION == SIGMOID:
        # exp(-|z|) <= 1, so neither branch overflows.
        e = tl.exp(-tl.abs(z))
        return tl.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    elif ACTIVATION == TANH:
        return libdevice.tanh(z)
    elif ACTIVATION == RELU:
        return tl.maximum(z, 0.0)
    else:
        return z


@triton.jit
def _slope(a, ACTIVATION: tl.constexpr):
    """g'(z) written through a = g(z), as the scalar network does."""
    if ACTIVATION == SIGMOID:
        return a * (1.0 - a)
    elif ACTIVATION == TANH:
        return 1.0 - a * a
    elif ACTIVATION == RELU:
        # a > 0 exactly when z > 0; the kink takes the subgradient zero.
        return tl.where(a > 0.0, 1.0, 0.0)
    else:
        return tl.full(a.shape, 1.0, tl.float64)


@triton.jit
def dense_rows(
    x_ptr,  # (n, D) uint8 pixels when BYTES, else float64 inputs
    rows_ptr,  # (B,) int64 row indices into x: the batch gather is fused into the loads
    w_ptr,  # (K, D) float64
    b_ptr,  # (K,) float64
    out_ptr,  # (B, K) float64 output: g(X W^T + b)
    B,
    K,
    D: tl.constexpr,
    BYTES: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    m_ok, k_ok = m < B, k < K
    rows = tl.load(rows_ptr + m, mask=m_ok, other=0)
    z = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    for start in tl.range(0, D, BLOCK_D):
        d = start + tl.arange(0, BLOCK_D)
        d_ok = d < D
        x = tl.load(
            x_ptr + rows[:, None] * D + d[None, :], mask=m_ok[:, None] & d_ok[None, :], other=0
        )
        if BYTES:
            x = x.to(tl.float64) / 255.0
        w = tl.load(
            w_ptr + k[:, None] * D + d[None, :], mask=k_ok[:, None] & d_ok[None, :], other=0.0
        )
        z += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    z += tl.load(b_ptr + k, mask=k_ok, other=0.0)[None, :]
    tl.store(
        out_ptr + m[:, None] * K + k[None, :],
        _activate(z, ACTIVATION),
        mask=m_ok[:, None] & k_ok[None, :],
    )


@triton.jit
def hidden_delta(
    delta_ptr,  # (B, K) float64: dL/dz of the layer above
    w_ptr,  # (K, H) float64: that layer's weights, before this step's update
    a_ptr,  # (B, H) float64: this layer's activations
    out_ptr,  # (B, H) float64 output: dL/dz of this layer
    B,
    H,
    K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m_ok, h_ok = m < B, h < H
    total = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float64)
    for start in tl.range(0, K, BLOCK_K):
        k = start + tl.arange(0, BLOCK_K)
        k_ok = k < K
        delta = tl.load(
            delta_ptr + m[:, None] * K + k[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0.0
        )
        w = tl.load(
            w_ptr + k[:, None] * H + h[None, :], mask=k_ok[:, None] & h_ok[None, :], other=0.0
        )
        # (M, K, 1) * (1, K, H) summed over K: one tile of delta W.
        total += tl.sum(delta[:, :, None] * w[None, :, :], axis=1)
    tile = m[:, None] * H + h[None, :]
    tile_ok = m_ok[:, None] & h_ok[None, :]
    a = tl.load(a_ptr + tile, mask=tile_ok, other=0.0)
    tl.store(out_ptr + tile, total * _slope(a, ACTIVATION), mask=tile_ok)
