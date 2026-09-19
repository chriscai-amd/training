#!/usr/bin/env python3
"""CPU FP64 reference for selected rows of captured SiLU-attention DQ.

Uses every key/value row in the selected sequence and the checked-in PyTorch
mask helper. This is a mathematical reference, not an emulation of Triton's
intermediate BF16 rounding or a certificate for an arbitrary small difference.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Optional

import torch

MASK_SOURCE = Path(__file__).resolve().parents[1] / 'generative_recommenders/ops/pytorch/pt_hstu_attention.py'
MASK_SOURCE_SHA256 = 'abf4027a9e8e3ba09e3652cde0d889c78a34488f25109ee7f30d8ead15d02790'


def mask_helper():
    body = MASK_SOURCE.read_bytes()
    if hashlib.sha256(body).hexdigest() != MASK_SOURCE_SHA256:
        raise ValueError('Reviewed PyTorch attention-mask source changed')
    functions = [n for n in ast.parse(body).body
                 if isinstance(n, ast.FunctionDef) and n.name == '_get_valid_attn_mask']
    if len(functions) != 1:
        raise ValueError('Expected exactly one production mask helper')
    functions[0].decorator_list = []
    namespace = {'torch': torch, 'Optional': Optional}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(MASK_SOURCE), 'exec'), namespace)
    return namespace['_get_valid_attn_mask']


def validate(inputs, scalars):
    for name in ('q', 'k', 'v', 'dout', 'seq_offsets'):
        value = inputs.get(name)
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError(f'Captured CPU tensor required: {name}')
    q, k, v, g = (inputs[n] for n in ('q', 'k', 'v', 'dout'))
    if (q.ndim != 3 or min(q.shape) < 1 or k.shape != q.shape
            or v.ndim != 3 or v.shape[:2] != q.shape[:2] or g.shape != v.shape):
        raise ValueError('Invalid attention input shapes')
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, g)):
        raise ValueError('This reference accepts captured BF16 operands')
    n, alpha = scalars.get('N'), scalars.get('alpha')
    if type(n) is not int or n < 1 or type(alpha) not in (int, float) or not math.isfinite(alpha):
        raise ValueError('Positive normalization N and finite alpha required')
    if scalars.get('enable_tma') is not False or scalars.get('num_softmax_heads') != 0:
        raise ValueError('Only ordinary non-TMA SiLU attention is supported')
    for key in ('contextual_seq_len', 'max_attn_len'):
        if type(scalars.get(key)) is not int or scalars[key] < 0:
            raise ValueError(f'Invalid mask argument: {key}')
    offsets = inputs['seq_offsets']
    if offsets.dtype not in (torch.int32, torch.int64) or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError('Integer sequence-offset vector required')
    offsets = offsets.tolist()
    if offsets[0] != 0 or offsets[-1] != q.shape[0] or any(not 0 <= b-a <= n for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Invalid disjoint sequence partition or length above N')
    targets = inputs.get('num_targets')
    if targets is not None:
        if (not isinstance(targets, torch.Tensor) or targets.device.type != 'cpu'
                or targets.dtype not in (torch.int32, torch.int64) or tuple(targets.shape) != (len(offsets)-1,)):
            raise ValueError('Invalid target-count vector')
        if any(not 0 <= t <= b-a for t, a, b in zip(targets.tolist(), offsets, offsets[1:])):
            raise ValueError('Target count outside sequence')
    return offsets


def reference_query(inputs, scalars, row, head):
    offsets = validate(inputs, scalars)
    q = inputs['q']
    if type(row) is not int or type(head) is not int or not 0 <= row < q.shape[0] or not 0 <= head < q.shape[1]:
        raise ValueError('Invalid query row/head')
    sequence = next(i for i, (a, b) in enumerate(zip(offsets, offsets[1:])) if a <= row < b)
    start, stop = offsets[sequence:sequence+2]
    local = row-start
    qr = q[row, head].double()
    kr, vr = (inputs[n][start:stop, head].double() for n in ('k', 'v'))
    gr = inputs['dout'][row, head].double()
    if any(not bool(torch.isfinite(t).all()) for t in (qr, kr, vr, gr)):
        raise ValueError('Selected complete query dependencies must be finite')
    targets = inputs.get('num_targets')
    # The actual-row submatrix is independent of the number of padded rows.
    mask = mask_helper()(device=torch.device('cpu'), causal=True, N=stop-start,
        seq_lengths=torch.tensor([stop-start]),
        num_targets=targets[sequence:sequence+1] if targets is not None else None,
        max_attn_len=scalars['max_attn_len'], contextual_seq_len=scalars['contextual_seq_len'])[0, local]
    score = (kr @ qr) * scalars['alpha']
    if not bool(torch.isfinite(score).all()):
        raise ValueError('FP64 score overflow; this reference is unsupported')
    sigmoid = torch.sigmoid(score)
    derivative = sigmoid * (1 + score * (1-sigmoid))
    ds = ((vr @ gr) * derivative / scalars['N']) * mask
    result = (ds @ kr) * scalars['alpha']
    if not bool(torch.isfinite(result).all()):
        raise ValueError('FP64 derivative overflow; this reference is unsupported')
    return {'dq_fp64': result, 'dq_nearest_bf16': result.bfloat16(),
            'sequence': sequence, 'sequence_start': start, 'sequence_length': stop-start,
            'query_position': local, 'row': row, 'head': head,
            'participating_keys': int(mask.sum()), 'normalization_N': scalars['N'],
            'alpha_as_reported': scalars['alpha'],
            'max_abs_score': float(score.abs().max()), 'max_abs_reference': float(result.abs().max()),
            'zero_incoming_query_gradient': bool((gr == 0).all()),
            'mask_source_sha256': MASK_SOURCE_SHA256,
            'scope': 'All keys/values for this query; FP64 mathematical derivative with original mask and N. No intermediate BF16-rounding emulation or universal floating-point tolerance.'}
