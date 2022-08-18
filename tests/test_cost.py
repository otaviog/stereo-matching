"""
Unit testing and benchmarking of cost functions.
"""

import pytest
import torch

from stereomatch.cuda_texture import CUDATexture
from stereomatch.cost import SSDTexture, SSD, Birchfield
from stereomatch.disparity_reduce import WinnerTakesAll

from viz import save_depthmap


# pylint: disable=redefined-outer-name


def test_ssd(sample_stereo_pair):
    left, right = sample_stereo_pair

    tex1 = CUDATexture.from_tensor(left, normalized_coords=False)
    tex2 = CUDATexture.from_tensor(right, normalized_coords=False)
    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)
    cost_volume_tex = ssd_texture(tex1, tex2)

    ssd = SSD(pytest.STM_MAX_DISPARITY)
    cost_volume_gpu = ssd(left.cuda(), right.cuda())
    cost_volume_cpu = ssd(left, right)

    torch.testing.assert_allclose(cost_volume_gpu.cpu(), cost_volume_cpu)
    torch.testing.assert_allclose(cost_volume_tex.cpu(), cost_volume_cpu)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_ssd(sample_stereo_pair, benchmark):
    left, right = sample_stereo_pair[0].cuda(), sample_stereo_pair[1].cuda()

    ssd = SSD(pytest.STM_MAX_DISPARITY)
    benchmark(ssd, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_ssd_texture(sample_stereo_pair, benchmark):
    left, right = sample_stereo_pair
    left = CUDATexture.from_tensor(left, normalized_coords=False)
    right = CUDATexture.from_tensor(right, normalized_coords=False)
    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)
    benchmark(ssd_texture, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_upload_ssd_texture(sample_stereo_pair, benchmark):
    left, right = sample_stereo_pair
    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)

    def _target(left=left, right=right):
        left = CUDATexture.from_tensor(left, normalized_coords=False)
        right = CUDATexture.from_tensor(right, normalized_coords=False)
        ssd_texture(left, right)
    benchmark(_target)


def test_birchfield(sample_stereo_pair):
    left, right = sample_stereo_pair[0].cuda(), sample_stereo_pair[1].cuda()
    # left, right = sample_stereo_pair[0], sample_stereo_pair[1]
    birchfield = Birchfield(pytest.STM_MAX_DISPARITY)
    cost_volume = birchfield(left, right)

    wta = WinnerTakesAll()
    disparity_image = wta(cost_volume)
    save_depthmap(disparity_image, "birchfield-wta")


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_birchfield(sample_stereo_pair, benchmark):
    left, right = sample_stereo_pair[0].cuda(), sample_stereo_pair[1].cuda()
    birchfield = Birchfield(pytest.STM_MAX_DISPARITY)
    benchmark(birchfield, left, right)
