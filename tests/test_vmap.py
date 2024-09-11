import torch
import torch.nn.functional as F
from torch.func import jacfwd
import pytest
from torchcomp.core import compressor_core


from .test_grad import create_test_inputs


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_vmap(device: str):
    batch_size = 4
    samples = 128
    x, zi, at, rt = tuple(x.to(device) for x in create_test_inputs(batch_size, samples))
    y = torch.randn_like(x)

    x.requires_grad = True
    zi.requires_grad = True
    at.requires_grad = True
    rt.requires_grad = True

    args = (x, zi, at, rt)

    def func(x, zi, at, rt):
        return F.mse_loss(compressor_core(x, zi, at, rt), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)
