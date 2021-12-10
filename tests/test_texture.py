import pytest
import torch

from stereomatch.cuda_texture import CUDATexture


def test_initialization():
    """
    Test initialization
    """
    tensor = torch.rand(100, 100, 3, device="cuda:0")
    tex1 = CUDATexture.from_tensor(tensor)

    assert tex1.width == 100
    assert tex1.height == 100
    assert tex1.channels == 3

    tensor = torch.rand(100, 100, device="cuda:0")
    tex2 = CUDATexture.from_tensor(tensor)

    assert tex2.width == 100
    assert tex2.height == 100
    assert tex2.channels == 1


def test_non_device_creation():
    """
    Tests that creating a texture from a non-cuda tensor
    raises an exception.
    """
    with pytest.raises(RuntimeError):
        CUDATexture.from_tensor(torch.rand(100, 100, 3))


def test_transfers():
    """
    Tests if the from_tensor and to_tensor returns the same tensor, 
    3-channel case.
    """
    tensor0 = torch.rand(100, 100, 3, device="cuda:0")
    tensor1 = CUDATexture.from_tensor(tensor0).to_tensor()
    torch.testing.assert_allclose(tensor0, tensor1)


def test_single_channel():
    """
    Tests if the from_tensor and to_tensor returns the same tensor,
    1-channel case.
    """
    tensor0 = torch.rand(100, 100, device="cuda:0")
    tensor1 = CUDATexture.from_tensor(tensor0).to_tensor().squeeze()

    torch.testing.assert_allclose(tensor0, tensor1)
