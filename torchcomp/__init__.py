import torch
import torch.nn.functional as F

from .core import compressor_core

__all__ = ["compexp_gain", "limiter_gain"]

amp2db = lambda x: 20 * torch.log10(x)
db2amp = lambda x: 10 ** (x / 20)


def compexp_gain(
    x_rms: torch.Tensor,
    comp_thresh: torch.Tensor,
    comp_slope: torch.Tensor,
    exp_thresh: torch.Tensor,
    exp_slope: torch.Tensor,
    at: torch.Tensor,
    rt: torch.Tensor,
) -> torch.Tensor:
    """Compressor-Expander gain function.

    Args:
        x_rms (torch.Tensor): Input signal RMS.
        comp_thresh (torch.Tensor): Compressor threshold in dB.
        comp_slope (torch.Tensor): Compressor slope.
        exp_thresh (torch.Tensor): Expander threshold in dB.
        exp_slope (torch.Tensor): Expander slope.
        at (torch.Tensor): Attack time.
        rt (torch.Tensor): Release time.

    Shape:
        - x_rms: :math:`(B, T)` where :math:`B` is the batch size and :math:`T` is the number of samples.
        - comp_thresh: :math:`(B,)`.
        - comp_slope: :math:`(B,)`.
        - exp_thresh: :math:`(B,)`.
        - exp_slope: :math:`(B,)`.
        - at: :math:`(B,)`.
        - rt: :math:`(B,)`.

    """
    assert torch.all(x_rms > 0)
    assert torch.all(comp_slope >= 0) and torch.all(comp_slope <= 1)
    assert torch.all(exp_slope <= 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    log_x_rms = amp2db(x_rms)
    g = (
        torch.minimum(
            comp_slope * (comp_thresh - log_x_rms), exp_slope * (exp_thresh - log_x_rms)
        )
        .neg()
        .relu()
        .neg()
    )
    f = db2amp(g)
    zi = x_rms.new_zeros(f.shape[0])
    return compressor_core(f, zi, at, rt)


def limiter_gain(
    x: torch.Tensor, threshold: torch.Tensor, at: torch.Tensor, rt: torch.Tensor
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
        - threshold: :math:`(B,)`.
        - at: :math:`(B,)`.
        - rt: :math:`(B,)`.

    """
    assert torch.all(threshold <= 0)
    assert torch.all(at > 0) and torch.all(at < 1)
    assert torch.all(rt > 0) and torch.all(rt < 1)

    zi = x.new_zeros(x.shape[0])
    lt = db2amp(threshold)
    x_peak = compressor_core(x.abs(), zi, rt, at)
    f = F.relu(1 - lt / x_peak).neg() + 1
    return compressor_core(f, zi, at, rt)
