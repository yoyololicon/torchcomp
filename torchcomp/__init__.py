import torch
import torch.nn.functional as F
from typing import Union
from torchaudio.functional import lfilter

from .core import compressor_core

__all__ = [
    "compexp_gain",
    "limiter_gain",
    "ms2coef",
    "avg",
    "amp2db",
    "db2amp",
    "coef2ms",
]


def amp2db(x: torch.Tensor) -> torch.Tensor:
    """Convert amplitude to decibels.

    Args:
        x (torch.Tensor): Input amplitude.

    Returns:
        torch.Tensor: Output decibels.

    """
    return 20 * torch.log10(x)


def db2amp(x: torch.Tensor) -> torch.Tensor:
    """Convert decibels to amplitude.

    Args:
        x (torch.Tensor): Input decibels.

    Returns:
        torch.Tensor: Output amplitude.

    """
    return 10 ** (x / 20)


def ms2coef(ms: torch.Tensor, sr: int) -> torch.Tensor:
    """Convert milliseconds to coefficient.

    Args:
        ms (torch.Tensor): Input milliseconds.
        sr (int): Sample rate.

    Returns:
        torch.Tensor: Output coefficient.

    """
    return 1 - torch.exp(-2200 / ms / sr)


def coef2ms(coef: torch.Tensor, sr: int) -> torch.Tensor:
    """Convert coefficient to milliseconds.

    Args:
        coef (torch.Tensor): Input coefficient.
        sr (int): Sample rate.

    Returns:
        torch.Tensor: Output milliseconds.

    """
    return -2200 / (sr * torch.log(1 - coef))


def avg(rms: torch.Tensor, avg_coef: Union[torch.Tensor, float]):
    """Compute the running average of a signal.

    Args:
        rms (torch.Tensor): Input signal.
        avg_coef (torch.Tensor): Coefficient for the average RMS.

    Shape:
        - rms: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - avg_coef: :math:`(B,)` or a scalar.

    """

    avg_coef = torch.as_tensor(
        avg_coef, dtype=rms.dtype, device=rms.device
    ).broadcast_to(rms.shape[0])
    assert torch.all(avg_coef > 0) and torch.all(avg_coef <= 1)

    return lfilter(
        rms,
        torch.stack([torch.ones_like(avg_coef), avg_coef - 1], 1),
        torch.stack([avg_coef, torch.zeros_like(avg_coef)], 1),
        False,
    )


def compexp_gain(
    x_rms: torch.Tensor,
    comp_thresh: Union[torch.Tensor, float],
    comp_ratio: Union[torch.Tensor, float],
    exp_thresh: Union[torch.Tensor, float],
    exp_ratio: Union[torch.Tensor, float],
    at: Union[torch.Tensor, float],
    rt: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Compressor-Expander gain function.

    Args:
        x_rms (torch.Tensor): Input signal RMS.
        comp_thresh (torch.Tensor): Compressor threshold in dB.
        comp_ratio (torch.Tensor): Compressor ratio.
        exp_thresh (torch.Tensor): Expander threshold in dB.
        exp_ratio (torch.Tensor): Expander ratio.
        at (torch.Tensor): Attack time.
        rt (torch.Tensor): Release time.

    Shape:
        - x_rms: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - comp_thresh: :math:`(B,)` or a scalar.
        - comp_ratio: :math:`(B,)` or a scalar.
        - exp_thresh: :math:`(B,)` or a scalar.
        - exp_ratio: :math:`(B,)` or a scalar.
        - at: :math:`(B,)` or a scalar.
        - rt: :math:`(B,)` or a scalar.

    """
    device, dtype = x_rms.device, x_rms.dtype
    factory_func = lambda x: torch.as_tensor(
        x, device=device, dtype=dtype
    ).broadcast_to(x_rms.shape[0])
    comp_ratio = factory_func(comp_ratio)
    exp_ratio = factory_func(exp_ratio)
    comp_thresh = factory_func(comp_thresh)
    exp_thresh = factory_func(exp_thresh)
    at = factory_func(at)
    rt = factory_func(rt)

    assert torch.all(x_rms > 0)
    assert torch.all(comp_ratio > 1)
    assert torch.all(exp_ratio < 1) and torch.all(exp_ratio > 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    comp_slope = 1 - 1 / comp_ratio
    exp_slope = 1 - 1 / exp_ratio

    log_x_rms = amp2db(x_rms)
    g = (
        torch.minimum(
            comp_slope[:, None] * (comp_thresh[:, None] - log_x_rms),
            exp_slope[:, None] * (exp_thresh[:, None] - log_x_rms),
        )
        .neg()
        .relu()
        .neg()
    )
    f = db2amp(g)
    zi = x_rms.new_ones(f.shape[0])
    return compressor_core(f, zi, at, rt)


def limiter_gain(
    x: torch.Tensor,
    threshold: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
) -> torch.Tensor:
    """Limiter gain function.
    This implementation use the same attack and release time for level detection and gain smoothing.

    Args:
        x (torch.Tensor): Input signal.
        threshold (torch.Tensor): Limiter threshold in dB.
        at (torch.Tensor): Attack time.
        rt (torch.Tensor): Release time.

    Shape:
        - x: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - threshold: :math:`(B,)` or a scalar.
        - at: :math:`(B,)` or a scalar.
        - rt: :math:`(B,)` or a scalar.

    """
    assert torch.all(threshold <= 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    factory_func = lambda x: torch.as_tensor(
        x, device=x.device, dtype=x.dtype
    ).broadcast_to(x.shape[0])
    threshold = factory_func(threshold)
    at = factory_func(at)
    rt = factory_func(rt)

    zi = x.new_zeros(x.shape[0])
    lt = db2amp(threshold)
    x_peak = compressor_core(x.abs(), zi, rt, at)
    f = F.relu(1 - lt / x_peak).neg() + 1
    return compressor_core(f, torch.ones_like(zi), at, rt)
