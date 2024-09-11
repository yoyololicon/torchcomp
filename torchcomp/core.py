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
    b: int = cuda.blockIdx.x
    i: int = cuda.threadIdx.x

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
        x: torch.Tensor, zi: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return y, at_mask

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        x, zi, at, rt = inputs
        y, at_mask = output
        ctx.mark_non_differentiable(at_mask)
        ctx.save_for_backward(x, y, zi, at, rt, at_mask)
        ctx.save_for_forward(x, y, zi, at, rt, at_mask)
        return ctx

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor, _
    ) -> Tuple[Optional[torch.Tensor], ...]:
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
    ) -> Tuple[torch.Tensor, None]:
        x, y, zi, at, rt, at_mask = ctx.saved_tensors
        coeffs = torch.where(at_mask, at.unsqueeze(1), rt.unsqueeze(1))

        fwd_x = 0 if grad_x is None else grad_x * coeffs

        fwd_combined: torch.Tensor
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
        return (
            sample_wise_lpc(
                fwd_combined,
                coeffs.unsqueeze(2) - 1,
                grad_zi if grad_zi is None else grad_zi.unsqueeze(1),
            ),
            None,
        )

    @staticmethod
    def vmap(info, in_dims, *args):
        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        x, zi, at, rt = tuple(
            map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                map(maybe_expand_bdim_at_front, args, in_dims),
            )
        )

        y, at_mask = CompressorFunction.apply(x, zi, at, rt)
        return (
            y.reshape(info.batch_size, -1, *y.shape[1:]),
            at_mask.reshape(info.batch_size, -1, *at_mask.shape[1:]),
        ), 0


def compressor_core(*args, **kwargs) -> torch.Tensor:
    return CompressorFunction.apply(*args, **kwargs)[0]
