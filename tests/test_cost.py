
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
import torch

from stereomatch.cuda_texture import CUDATexture
from stereomatch.cost import ssd_texture, ssd, birchfield


@pytest.fixture
def images_rgb():
    image_base_dir = Path(__file__).parent.parent / \
        "test-data/middleburry/teddy/"

    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im2.png").convert('L'))).float() / 255.0
    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im6.png").convert('L'))).float() / 255.0

    return left_image, right_image

# pylint: disable=redefined-outer-name


def test_ssd(images_rgb):
    left, right = images_rgb
    print("use texture")
    tex1 = CUDATexture.from_tensor(left, normalized_coords=False)
    tex2 = CUDATexture.from_tensor(right, normalized_coords=False)
    cost_volume_tex = ssd_texture(tex1, tex2, pytest.STM_MAX_DISPARITY)
    print("done texture")
    cost_volume_gpu = ssd(left.cuda(), right.cuda(), pytest.STM_MAX_DISPARITY)

    cost_volume_cpu = ssd(left, right, pytest.STM_MAX_DISPARITY)

    torch.testing.assert_allclose(cost_volume_gpu.cpu(), cost_volume_cpu)
    torch.testing.assert_allclose(cost_volume_tex.cpu(), cost_volume_cpu)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_ssd(images_rgb, benchmark):
    left, right = images_rgb[0].cuda(), images_rgb[1].cuda()

    benchmark(ssd, left, right, pytest.STM_MAX_DISPARITY)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_ssd_texture(images_rgb, benchmark):
    left, right = images_rgb
    left = CUDATexture.from_tensor(left, normalized_coords=False)
    right = CUDATexture.from_tensor(right, normalized_coords=False)
    benchmark(ssd_texture, left, right, pytest.STM_MAX_DISPARITY)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_upload_ssd_texture(images_rgb, benchmark):
    left, right = images_rgb

    def _target(left=left, right=right):
        left = CUDATexture.from_tensor(left, normalized_coords=False)
        right = CUDATexture.from_tensor(right, normalized_coords=False)
        ssd_texture(left, right, pytest.STM_MAX_DISPARITY)
    benchmark(_target)


def test_birchfield(images_rgb):
    left, right = images_rgb[0].cuda(), images_rgb[1].cuda()
    cost_volume = birchfield(left, right, pytest.STM_MAX_DISPARITY)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_birchfield(images_rgb, benchmark):
    left, right = images_rgb[0].cuda(), images_rgb[1].cuda()
    benchmark(ssd, left, right, pytest.STM_MAX_DISPARITY)
