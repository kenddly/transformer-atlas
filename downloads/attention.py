"""Inspectably small CPU attention, explicit multihead projections, and checks.

Educational implementation, not a training pipeline or optimized GPU kernel.
Mask convention: Boolean True means ALLOWED. All-masked rows return zeros.
No dropout is used, so numerical comparisons are deterministic in this example.
Run: python attention.py
"""
from __future__ import annotations

import math
import torch


def scaled_dot_product_attention(q, k, v, allowed=None):
    """Return (output, weights) from ordinary PyTorch tensor operations.

    q: [..., T_query, d_key], k: [..., T_key, d_key],
    v: [..., T_key, d_value]. Batch/head prefixes must match exactly.
    allowed: Boolean array broadcastable to [..., T_query, T_key].

    Policy for an entirely blocked row: its weights and output are zero. This
    deliberately defines a safe extension of softmax, whose all--infinity row
    would otherwise be undefined. Such rows still need separate loss handling.
    All inputs must be finite floating point arrays of one dtype/device.
    """
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError('q, k, v need at least a token and feature axis')
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError('batch/head prefixes must match')
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError('Q/K compatibility widths and K/V token counts must match')
    if q.shape[-1] == 0 or k.shape[-2] == 0:
        raise ValueError('empty compatibility/token axes are not supported')
    if not (q.is_floating_point() and q.dtype == k.dtype == v.dtype):
        raise TypeError('q, k, v must share a floating-point dtype')
    if not (q.device == k.device == v.device):
        raise ValueError('q, k, v must share a device')
    if not all(bool(torch.isfinite(t).all()) for t in (q, k, v)):
        raise ValueError('inputs must be finite')

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError('score overflow: reduce magnitude or use a wider dtype')
    if allowed is None:
        allowed = torch.ones_like(scores, dtype=torch.bool)
    if allowed.dtype != torch.bool or allowed.device != q.device:
        raise TypeError('allowed must be Boolean and on the input device')
    allowed = torch.broadcast_to(allowed, scores.shape)
    masked = scores.masked_fill(~allowed, -torch.inf)
    row_has_key = allowed.any(dim=-1, keepdim=True)
    row_max = masked.amax(dim=-1, keepdim=True)
    # Avoid subtracting -inf from -inf in fully blocked rows.
    row_max = torch.where(row_has_key, row_max, torch.zeros_like(row_max))
    unnormalized = torch.exp(masked - row_max)
    denominator = unnormalized.sum(dim=-1, keepdim=True)
    weights = unnormalized / denominator.clamp_min(torch.finfo(q.dtype).tiny)
    return weights @ v, weights


def analytic_attention_backward(q, k, v, weights, upstream, allowed=None):
    """Analytical Q/K/V gradients for a fixed mask and no dropout.

    No autograd is used in the algebra below. Use detached inputs to compare
    with automatic differentiation. Fully masked rows have zero score gradient.
    """
    d_v = weights.transpose(-1, -2) @ upstream
    d_a = upstream @ v.transpose(-1, -2)
    d_s = weights * (d_a - (weights * d_a).sum(dim=-1, keepdim=True))
    if allowed is not None:
        d_s = d_s * torch.broadcast_to(allowed, d_s.shape)
    scale = math.sqrt(q.shape[-1])
    d_q = (d_s @ k) / scale
    d_k = (d_s.transpose(-1, -2) @ q) / scale
    return d_q, d_k, d_v


def projection_backward(x_query, x_source, w_q, w_k, w_v, d_q, d_k, d_v):
    """Backward through simple [B,T,D] projections, without bias.

    For self-attention x_query and x_source are the same input, so sum the two
    returned input gradients to include Q, K, and V paths.
    """
    d_wq = (x_query.transpose(-1, -2) @ d_q).sum(dim=0)
    d_wk = (x_source.transpose(-1, -2) @ d_k).sum(dim=0)
    d_wv = (x_source.transpose(-1, -2) @ d_v).sum(dim=0)
    d_xq = d_q @ w_q.T
    d_xs = d_k @ w_k.T + d_v @ w_v.T
    return d_xq, d_xs, d_wq, d_wk, d_wv


def multihead_self_attention(x, w_q, w_k, w_v, w_o, heads, allowed=None):
    """Explicit projections, head reshapes, attention, concatenation, output.

    x: [B,T,D], each weight: [D,D], returned output: [B,T,D].
    A batch mask [B,T,T] must be expanded to [B,1,T,T] by the caller.
    """
    if x.ndim != 3 or heads <= 0 or x.shape[-1] % heads:
        raise ValueError('x must be [B,T,D] with D divisible by positive heads')
    batch, tokens, width = x.shape
    if any(w.shape != (width, width) for w in (w_q, w_k, w_v, w_o)):
        raise ValueError('each projection must be [D,D]')
    head_width = width // heads

    def split(projected):
        # Each head can access every original input feature via its projection.
        return projected.reshape(batch, tokens, heads, head_width).transpose(1, 2)

    q, k, v = split(x @ w_q), split(x @ w_k), split(x @ w_v)
    out, weights = scaled_dot_product_attention(q, k, v, allowed)
    concatenated = out.transpose(1, 2).contiguous().reshape(batch, tokens, width)
    return concatenated @ w_o, weights


def run_checks():
    """Execute small mathematical invariants on CPU in float64."""
    torch.manual_seed(17)
    dtype = torch.float64
    results = {}
    # Numerical example shown in the website.
    q = torch.tensor([[[1., 0.]]], dtype=dtype)
    k = torch.tensor([[[1., 0.], [0., 1.]]], dtype=dtype)
    v = torch.tensor([[[2., 0.], [0., 4.]]], dtype=dtype)
    out, a = scaled_dot_product_attention(q, k, v)
    results['numerical_weights'] = a.tolist()
    results['numerical_output'] = out.tolist()
    torch.testing.assert_close(a.sum(-1), torch.ones_like(a.sum(-1)))

    q = torch.randn(2, 3, 2, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 3, 2, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    allowed = torch.ones(3, 3, dtype=torch.bool).tril()
    allowed[0] = False  # Also exercise the fully masked row policy.
    out, a = scaled_dot_product_attention(q, k, v, allowed)
    g = torch.randn_like(out)
    (out * g).sum().backward()
    analytical = analytic_attention_backward(q.detach(), k.detach(), v.detach(), a.detach(), g, allowed)
    errors = []
    for computed, expected in zip(analytical, (q.grad, k.grad, v.grad)):
        torch.testing.assert_close(computed, expected, rtol=1e-9, atol=1e-10)
        errors.append(float((computed - expected).abs().max()))
    results['analytic_gradient_max_errors_QKV'] = errors
    assert bool(torch.isfinite(out).all()) and bool((out[:, 0] == 0).all())
    assert bool((a[:, 0] == 0).all())

    # Central finite difference on a permitted query element.
    index, epsilon = (0, 1, 0), 1e-6
    qp, qm = q.detach().clone(), q.detach().clone()
    qp[index] += epsilon
    qm[index] -= epsilon
    lp = (scaled_dot_product_attention(qp, k.detach(), v.detach(), allowed)[0] * g).sum()
    lm = (scaled_dot_product_attention(qm, k.detach(), v.detach(), allowed)[0] * g).sum()
    numerical_grad = (lp - lm) / (2 * epsilon)
    torch.testing.assert_close(numerical_grad, q.grad[index], rtol=1e-5, atol=1e-7)
    results['finite_difference_abs_error'] = float((numerical_grad - q.grad[index]).abs())

    # Causality: changing a future input cannot affect an earlier output.
    x = torch.randn(1, 4, 4, dtype=dtype)
    changed = x.clone()
    changed[:, -1] += 100
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    before = scaled_dot_product_attention(x, x, x, causal)[0]
    after = scaled_dot_product_attention(changed, changed, changed, causal)[0]
    torch.testing.assert_close(before[:, :-1], after[:, :-1], rtol=0, atol=0)
    results['causal_earlier_output_max_difference'] = float((before[:, :-1] - after[:, :-1]).abs().max())

    # Padding: valid outputs ignore changed padded source content.
    valid_keys = torch.tensor([True, True, False, False])[None, None, :]
    changed = x.clone()
    changed[:, 2:] += 50
    before = scaled_dot_product_attention(x, x, x, valid_keys)[0]
    after = scaled_dot_product_attention(changed, changed, changed, valid_keys)[0]
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    results['padding_valid_output_max_difference'] = float((before[:, :2] - after[:, :2]).abs().max())

    # Transparent multihead wrapper and an independent per-head loop agree.
    weights = [torch.randn(4, 4, dtype=dtype) * 0.2 for _ in range(4)]
    wq, wk, wv, wo = weights
    multi, multi_a = multihead_self_attention(x, wq, wk, wv, wo, 2, causal)
    pieces = []
    for head in range(2):
        sl = slice(2 * head, 2 * head + 2)
        pieces.append(scaled_dot_product_attention((x @ wq)[..., sl], (x @ wk)[..., sl], (x @ wv)[..., sl], causal)[0])
    loop_output = torch.cat(pieces, dim=-1) @ wo
    torch.testing.assert_close(multi, loop_output, rtol=1e-12, atol=1e-12)
    assert multi.shape == (1, 4, 4) and multi_a.shape == (1, 2, 4, 4)
    results['multihead_loop_max_difference'] = float((multi - loop_output).abs().max())

    # Gradients through projection matrices and all shared-input paths.
    shared = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    matrices = [torch.randn(4, 2, dtype=dtype, requires_grad=True) for _ in range(3)]
    pq, pk, pv = [shared @ w for w in matrices]
    projected_out, projected_a = scaled_dot_product_attention(pq, pk, pv)
    projected_g = torch.randn_like(projected_out)
    (projected_out * projected_g).sum().backward()
    pdq, pdk, pdv = analytic_attention_backward(pq.detach(), pk.detach(), pv.detach(), projected_a.detach(), projected_g)
    dxq, dxs, *dws = projection_backward(shared.detach(), shared.detach(), *[w.detach() for w in matrices], pdq, pdk, pdv)
    torch.testing.assert_close(dxq + dxs, shared.grad, rtol=1e-9, atol=1e-10)
    for actual, w in zip(dws, matrices):
        torch.testing.assert_close(actual, w.grad, rtol=1e-9, atol=1e-10)
    results['shared_input_projection_gradient_max_difference'] = float((dxq + dxs - shared.grad).abs().max())
    results['projection_weight_gradient_max_difference'] = max(float((dw - w.grad).abs().max()) for dw, w in zip(dws, matrices))

    # Compare a trusted library operation under the SAME convention and no dropout.
    reference = torch.nn.functional.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(), attn_mask=allowed, dropout_p=0.0)
    torch.testing.assert_close(out.detach(), reference, rtol=1e-9, atol=1e-10)
    results['torch_SDPA_max_difference'] = float((out.detach() - reference).abs().max())
    return results


if __name__ == '__main__':
    import json
    print(json.dumps({'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu', 'checks': run_checks()}, indent=2))
