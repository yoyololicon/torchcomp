import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from comp import compressor_core


def create_test_inputs(batch_size, samples):
    x = torch.rand(batch_size, samples).double()
    zi = torch.rand(batch_size).double()
    at = torch.rand(batch_size).double() * 0.2 + 0.5
    rt = torch.rand(batch_size).double() * 0.1 + 0.8

    return x, zi, at, rt


@pytest.mark.parametrize(
    "x_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "at_requires_grad",
    [False, True],
)
@pytest.mark.parametrize(
    "rt_requires_grad",
    [False, True],
)
@pytest.mark.parametrize(
    "zi_requires_grad",
    [False, True],
)
@pytest.mark.parametrize(
    "samples",
    [64],
)
def test_low_order_cpu(
    x_requires_grad: bool,
    at_requires_grad: bool,
    rt_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
):
    batch_size = 8
    x, zi, at, rt = create_test_inputs(batch_size, samples)
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad
    at.requires_grad = at_requires_grad
    rt.requires_grad = rt_requires_grad

    assert gradcheck(compressor_core, (x, zi, at, rt))
    assert gradgradcheck(compressor_core, (x, zi, at, rt))
