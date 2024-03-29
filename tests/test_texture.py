"""
Tests the CUDATexture class.
"""
import pytest
import torch

from stereomatch.cuda_texture import CUDATexture


def test_initialization():
    """
    Test initialization
    """
    tensor = torch.rand(100, 100, dtype=torch.float32,
                        device="cuda:0")
    tex1 = CUDATexture.from_tensor(tensor)

    assert tex1.width == 100
    assert tex1.height == 100
    assert tex1.channels == 1

    tensor = torch.rand(100, 100, 4, dtype=torch.float32,
                        device="cuda:0")
    tex2 = CUDATexture.from_tensor(tensor)

    assert tex2.width == 100
    assert tex2.height == 100
    assert tex2.channels == 4


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
    tensor0 = torch.rand(128, 128, 4, dtype=torch.float32)
    tex = CUDATexture.from_tensor(tensor0)
    tensor1 = tex.to_tensor()
    torch.testing.assert_allclose(tensor0, tensor1.cpu())


def test_single_channel():
    """
    Tests if the from_tensor and to_tensor returns the same tensor,
    1-channel case.
    """
    tensor0 = torch.rand(100, 100, device="cuda:0")
    tensor1 = CUDATexture.from_tensor(tensor0).to_tensor().squeeze()

    torch.testing.assert_allclose(tensor0, tensor1)


def test_access():
    """
    Test the texture-tensor transfer.
    """
    # pylint: disable=protected-access
    # tensor = (torch.rand(128, 128, device="cuda:0") * 255.0).float()
    # tensor = (torch.rand(512, 128) * 255.0).float().contiguous()
    tensor = (torch.rand(128, 128) * 255.0).byte().contiguous()
    texture = CUDATexture.from_tensor(tensor)
    output_tensor = torch.zeros_like(tensor).cuda()
    CUDATexture._run_transfer_test_kernel(texture, output_tensor)
    torch.testing.assert_allclose(output_tensor.cpu(), tensor)

    tensor = (torch.rand(128, 128) * 255.0).float().contiguous()
    texture = CUDATexture.from_tensor(tensor)
    output_tensor = torch.zeros_like(tensor).cuda()

    CUDATexture._run_transfer_test_kernel(texture, output_tensor)
    torch.testing.assert_allclose(output_tensor.cpu(), tensor)

def test_bindings():
    """
    Used for exploring internal CUDA texture parameters.
    """
    # pylint: disable=protected-access
    tensor = (torch.rand(128, 128, device="cuda:0") * 255.0).float()
    output_tensor = torch.zeros_like(tensor)

    CUDATexture._run_binding_test_kernel(tensor, output_tensor)

    torch.testing.assert_allclose(tensor.cpu(), output_tensor.cpu())


def test_completeness():
    """
    Test texture-to-tensor and tensor-to-texture transfer for multiple data types.
    """
    # pylint: disable=protected-access
    for dtype in [torch.uint8, torch.float32]:
        for device in ["cpu", "cuda:0"]:
            tensor = torch.rand(513, 123, 1) * 255.0
            tensor = tensor.to(dtype).to(device)

            tex = CUDATexture.from_tensor(tensor)
            assert tex.width == tensor.size(1)
            assert tex.height == tensor.size(0)
            assert tex.channels == tensor.size(2)

            output_tensor = torch.zeros_like(tensor).squeeze().to("cuda:0")
            CUDATexture._run_transfer_test_kernel(tex, output_tensor)

            torch.testing.assert_allclose(tensor.squeeze().cpu(),
                                          output_tensor.cpu())
