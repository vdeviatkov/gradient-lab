"""Two fused float64 kernels: softmax/cross-entropy forward, and gradient reduction plus SGD.

Neither kernel materializes a float copy of the images, a batch copy, or the weight gradient:

* `softmax_rows` gathers each batch row through an index array, scales its uint8 pixels by 1/255
  in registers, accumulates the logits over feature tiles, and in the same pass produces the
  per-row loss, the argmax prediction, and dL/dz = p - y.
* `reduce_and_update` gives every program one (class tile, feature tile) of the weights. It sums
  (p - y)^T x over the batch for that tile and immediately applies SGD with weight decay, so the
  gradient lives only in registers. Each weight is owned by exactly one program, so there are no
  atomics and the reduction order, and therefore every result, is deterministic.

Triton passes a Python float argument as fp32, which would silently round the learning rate, the
L2 coefficient, and 1 / batch to about seven digits. So 1 / batch is computed in-kernel from the
integer size, and the two hyperparameters arrive as a float64 device array.

Loop trip counts are compile-time constants: the feature count D is fixed per model, and the batch
loop runs to MAX_B, the batch size rounded up to a power of two, with the true size B used only in
masks. A full and a partial training batch therefore share one compiled kernel. It also keeps the
kernels runnable under TRITON_INTERPRET=1, whose interpreter cannot use a runtime integer as a
loop bound with current NumPy.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_rows(
    x_ptr,  # (n, D) uint8 pixels when BYTES, else float64 features
    rows_ptr,  # (B,) int64 row indices into x: the batch gather is fused into the loads
    labels_ptr,  # (n,) int64
    w_ptr,  # (K, D) float64
    b_ptr,  # (K,) float64
    delta_ptr,  # (B, K) float64 output: p - y
    loss_ptr,  # (B,) float64 output: log-sum-exp(z) - z_y
    pred_ptr,  # (B,) int64 output: argmax_k z_k, lowest index on ties
    B,
    K,
    D: tl.constexpr,
    BYTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.arange(0, BLOCK_K)
    m_ok, k_ok = m < B, k < K
    rows = tl.load(rows_ptr + m, mask=m_ok, other=0)
    logits = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float64)
    for start in tl.range(0, D, BLOCK_D):
        d = start + tl.arange(0, BLOCK_D)
        d_ok = d < D
        x = tl.load(
            x_ptr + rows[:, None] * D + d[None, :], mask=m_ok[:, None] & d_ok[None, :], other=0
        )
        if BYTES:
            x = x.to(tl.float64) / 255.0  # The only preprocessing step, applied in registers.
        w = tl.load(
            w_ptr + k[:, None] * D + d[None, :], mask=k_ok[:, None] & d_ok[None, :], other=0.0
        )
        # (M, 1, D) * (1, K, D) summed over D: one tile of X W^T.
        logits += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    logits += tl.load(b_ptr + k, mask=k_ok, other=0.0)[None, :]
    # Padding classes get -inf, so they take no probability mass and never win the argmax.
    logits = tl.where(k_ok[None, :], logits, float("-inf"))

    largest = tl.max(logits, axis=1)
    shifted = tl.exp(logits - largest[:, None])  # Every exponent is <= 0: nothing overflows.
    total = tl.sum(shifted, axis=1)
    labels = tl.load(labels_ptr + rows, mask=m_ok, other=0)
    is_label = k[None, :] == labels[:, None]
    true_logit = tl.sum(tl.where(is_label, logits, 0.0), axis=1)
    loss = largest + tl.log(total) - true_logit
    # Softmax composed with cross-entropy collapses dL/dz to p - y.
    delta = shifted / total[:, None] - tl.where(is_label, 1.0, 0.0)
    tl.store(delta_ptr + m[:, None] * K + k[None, :], delta, mask=m_ok[:, None] & k_ok[None, :])
    tl.store(loss_ptr + m, loss, mask=m_ok)
    tl.store(pred_ptr + m, tl.argmax(logits, axis=1, tie_break_left=True), mask=m_ok)


@triton.jit
def reduce_and_update(
    x_ptr,  # (n, D) uint8 pixels when BYTES, else float64 features
    rows_ptr,  # (B,) int64 row indices into x
    delta_ptr,  # (B, K) float64: p - y from softmax_rows
    w_ptr,  # (K, D) float64, updated in place when APPLY
    b_ptr,  # (K,) float64, updated in place when APPLY
    gw_ptr,  # (K, D) float64 gradient output when not APPLY
    gb_ptr,  # (K,) float64 gradient output when not APPLY
    B,
    K,
    hyper_ptr,  # (2,) float64: learning rate, L2 coefficient
    D: tl.constexpr,
    MAX_B: tl.constexpr,  # a power of two >= B: the batch loop's compile-time trip count
    BYTES: tl.constexpr,
    APPLY: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    scale = 1.0 / B.to(tl.float64)
    learning_rate = tl.load(hyper_ptr)
    l2 = tl.load(hyper_ptr + 1)
    k_ok, d_ok = k < K, d < D
    grad_w = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float64)
    grad_b = tl.zeros((BLOCK_K,), dtype=tl.float64)
    for start in tl.range(0, MAX_B, BLOCK_R):
        r = start + tl.arange(0, BLOCK_R)
        r_ok = r < B
        rows = tl.load(rows_ptr + r, mask=r_ok, other=0)
        x = tl.load(
            x_ptr + rows[:, None] * D + d[None, :], mask=r_ok[:, None] & d_ok[None, :], other=0
        )
        if BYTES:
            x = x.to(tl.float64) / 255.0
        delta = tl.load(
            delta_ptr + r[:, None] * K + k[None, :], mask=r_ok[:, None] & k_ok[None, :], other=0.0
        )
        # (R, K, 1) * (R, 1, D) summed over the batch rows R: one tile of (P - Y)^T X.
        grad_w += tl.sum(delta[:, :, None] * x[:, None, :], axis=0)
        grad_b += tl.sum(delta, axis=0)
    tile = k[:, None] * D + d[None, :]
    tile_ok = k_ok[:, None] & d_ok[None, :]
    w = tl.load(w_ptr + tile, mask=tile_ok, other=0.0)
    first_column = tl.program_id(1) == 0  # One program per class tile owns the biases.
    if APPLY:
        tl.store(w_ptr + tile, w - learning_rate * (grad_w * scale + l2 * w), mask=tile_ok)
        if first_column:
            b = tl.load(b_ptr + k, mask=k_ok, other=0.0)
            tl.store(b_ptr + k, b - learning_rate * grad_b * scale, mask=k_ok)
    else:
        tl.store(gw_ptr + tile, grad_w * scale + l2 * w, mask=tile_ok)
        if first_column:
            tl.store(gb_ptr + k, grad_b * scale, mask=k_ok)
