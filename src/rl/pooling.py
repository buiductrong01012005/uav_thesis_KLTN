"""Deterministic fixed average pooling, without persistent state-dict keys."""

import torch
from torch import nn


class FixedPool5To3(nn.Module):
    """AdaptiveAvgPool2d(3,3) for a5x5 map, without CUDA atomic backward.

    The fixed linear map is mathematically identical to PyTorch's adaptive
    average bins. It has no learned parameters or persistent state-dict keys.
    CUDA matmul is deterministic under the runner's CUBLAS workspace setting.
    """

    def __init__(self):
        super().__init__()
        matrix = torch.zeros((9, 25), dtype=torch.float64)
        for row in range(3):
            for col in range(3):
                r0, r1 = row * 5 // 3, ((row + 1) * 5 + 2) // 3
                c0, c1 = col * 5 // 3, ((col + 1) * 5 + 2) // 3
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        matrix[row * 3 + col, r * 5 + c] = 1 / ((r1 - r0) * (c1 - c0))
        self.register_buffer("matrix", matrix, persistent=False)

    def forward(self, x):
        if x.shape[-2:] != (5, 5):
            raise ValueError("Fixed pooling expects5x5 convolution maps")
        pooled = torch.matmul(x.flatten(-2), self.matrix.to(dtype=x.dtype).t())
        return pooled.reshape(*x.shape[:-2], 3, 3)
