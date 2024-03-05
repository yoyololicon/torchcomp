import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from numba import njit, prange
from typing import Tuple, Any, Optional, Callable
from torchlpc import sample_wise_lpc


@njit(parallel=True)
def compressor_kernel(
    x: np.ndarray, zi: np.ndarray, at: np.ndarray, rt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    B, T = x.shape
    y = np.empty_like(x)
    at_mask = np.zeros_like(x, dtype=np.bool_)

    for b in prange(B):
        g = zi[b]
        at_b = at[b]
        rt_b = rt[b]
        for t in range(T):
            f = x[b, t]
            flag = f < g
            # TODO: make if-else differentiable
            if flag:
                coeff = at_b
                at_mask[b, t] = True
            else:
                coeff = rt_b
            g *= 1 - coeff
            g += coeff * f
            y[b, t] = g

    return y, at_mask


class CompressorFunction(Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
    ) -> torch.Tensor:
        y, at_mask = compressor_kernel(
            x.detach().cpu().numpy(),
            zi.detach().cpu().numpy(),
            at.detach().cpu().numpy(),
            rt.detach().cpu().numpy(),
        )
        y = torch.from_numpy(y).to(x.device)
        at_mask = torch.from_numpy(at_mask).to(x.device)
        ctx.save_for_backward(x, y, zi, at, rt, at_mask)
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        x, y, zi, at, rt, at_mask = ctx.saved_tensors
        grad_x = grad_zi = grad_at = grad_rt = None

        coeffs = torch.where(at_mask, at.unsqueeze(1), rt.unsqueeze(1))
        lpc_a = coeffs.unsqueeze(2) - 1
        padded_lpc_a = F.pad(lpc_a.transpose(1, 2), (0, 1)).transpose(1, 2)

        if not ctx.needs_input_grad[1]:
            padded_grad_y = grad_y
            padded_lpc_a = padded_lpc_a[:, 1:]
        else:
            padded_grad_y = F.pad(grad_y.unsqueeze(1), (1, 0)).squeeze(1)

        grad_x_unscaled = sample_wise_lpc(
            padded_grad_y.flip(1), padded_lpc_a.flip(1)
        ).flip(1)

        if ctx.needs_input_grad[1]:
            grad_zi, grad_x_unscaled = grad_x_unscaled[:, 0], grad_x_unscaled[:, 1:]

        if ctx.needs_input_grad[0]:
            grad_x = grad_x_unscaled * coeffs

        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            grad_combined = grad_x_unscaled * (
                x - torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            )
            if ctx.needs_input_grad[2]:
                grad_at = torch.where(at_mask, grad_combined, 0).sum(1)
            if ctx.needs_input_grad[3]:
                grad_rt = torch.where(~at_mask, grad_combined, 0).sum(1)

        return grad_x, grad_zi, grad_at, grad_rt


compressor_core: Callable = CompressorFunction.apply
