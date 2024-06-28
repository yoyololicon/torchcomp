import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from numba import njit, prange, cuda
from typing import Tuple, Any, Optional, Callable
from torchlpc import sample_wise_lpc


@cuda.jit
def compressor_cuda_kernel(
    x: np.ndarray,
    zi: np.ndarray,
    at: np.ndarray,
    rt: np.ndarray,
    y: np.ndarray,
    at_mask: np.ndarray,
    B: int,
    T: int,
):
    b = cuda.blockIdx.x
    i = cuda.threadIdx.x

    if b >= B or i > 0:
        return

    g = zi[b]
    at_b = at[b]
    rt_b = rt[b]
    for t in range(T):
        f = x[b, t]
        if f < g:
            coeff = at_b
            at_mask[b, t] = 1
        else:
            coeff = rt_b
        g *= 1 - coeff
        g += coeff * f
        y[b, t] = g


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


def compressor_cuda(
    x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T = x.shape
    y = torch.empty_like(x)
    at_mask = torch.zeros_like(x, dtype=torch.uint8)

    threads_per_block = 1
    blocks_per_grid = B

    compressor_cuda_kernel[blocks_per_grid, threads_per_block](
        cuda.as_cuda_array(x),
        cuda.as_cuda_array(zi),
        cuda.as_cuda_array(at),
        cuda.as_cuda_array(rt),
        cuda.as_cuda_array(y),
        cuda.as_cuda_array(at_mask),
        B,
        T,
    )
    return y, at_mask.bool()


class CompressorFunction(Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
    ) -> torch.Tensor:
        if x.is_cuda:
            y, at_mask = compressor_cuda(
                x.detach(), zi.detach(), at.detach(), rt.detach()
            )
        else:
            y, at_mask = compressor_kernel(
                x.detach().cpu().numpy(),
                zi.detach().cpu().numpy(),
                at.detach().cpu().numpy(),
                rt.detach().cpu().numpy(),
            )
            y = torch.from_numpy(y).to(x.device)
            at_mask = torch.from_numpy(at_mask).to(x.device)
        ctx.save_for_backward(x, y, zi, at, rt, at_mask)

        # for jvp
        ctx.x = x
        ctx.y = y
        ctx.zi = zi
        ctx.at = at
        ctx.rt = rt
        ctx.at_mask = at_mask
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
                grad_at = torch.where(at_mask, grad_combined, 0.0).sum(1)
            if ctx.needs_input_grad[3]:
                grad_rt = torch.where(~at_mask, grad_combined, 0.0).sum(1)

        return grad_x, grad_zi, grad_at, grad_rt

    @staticmethod
    def jvp(
        ctx: Any,
        grad_x: torch.Tensor,
        grad_zi: torch.Tensor,
        grad_at: torch.Tensor,
        grad_rt: torch.Tensor,
    ) -> torch.Tensor:
        x, y, zi, at, rt, at_mask = ctx.x, ctx.y, ctx.zi, ctx.at, ctx.rt, ctx.at_mask
        coeffs = torch.where(at_mask, at.unsqueeze(1), rt.unsqueeze(1))

        fwd_x = 0 if grad_x is None else grad_x * coeffs

        if grad_at is None and grad_rt is None:
            fwd_combined = fwd_x
        else:
            grad_beta = torch.where(
                at_mask,
                0.0 if grad_at is None else grad_at.unsqueeze(1),
                0.0 if grad_rt is None else grad_rt.unsqueeze(1),
            )
            fwd_combined = fwd_x + grad_beta * (
                x - torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            )

        return sample_wise_lpc(
            fwd_combined,
            coeffs.unsqueeze(2) - 1,
            grad_zi if grad_zi is None else grad_zi.unsqueeze(1),
        )


compressor_core: Callable = CompressorFunction.apply
