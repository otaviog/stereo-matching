"""
Unit testing and benchmarking of cost functions.
"""

import pytest
import torch

from stereomatch.cuda_texture import CUDATexture
from stereomatch.cost import SSDTexture, SSD, Birchfield
from stereomatch.disparity_reduce import WinnerTakesAll

from .viz import save_depthmap


# pylint: disable=redefined-outer-name


def test_ssd_cpu_result(sample_stereo_pair):
    """
    Integration test of the gpu result. Outputs WTA image into
    the test result folder.
    """
    left, right = sample_stereo_pair

    save_depthmap(WinnerTakesAll()(SSD(pytest.STM_MAX_DISPARITY)(left, right)),
                  pytest.STM_TEST_OUTPUT_PATH / "ssd")


def test_ssd_gpu_result(sample_stereo_pair):
    """
    Integration test of the gpu result. Outputs WTA image into
    the test result folder.
    """
    left, right = sample_stereo_pair

    save_depthmap(WinnerTakesAll()(SSD(pytest.STM_MAX_DISPARITY)(
        left.cuda(), right.cuda())),
        pytest.STM_TEST_OUTPUT_PATH / "ssd")


def test_ssd_should_cpu_gpu_equal(sample_stereo_pair):
    """
    Asserts that the cpu, gpu and gpu-tex implementations output
    the same results.
    """
    left, right = sample_stereo_pair

    tex1 = CUDATexture.from_tensor(left)
    tex2 = CUDATexture.from_tensor(right)
    cost_volume_tex = SSDTexture(pytest.STM_MAX_DISPARITY)(tex1, tex2)

    ssd = SSD(pytest.STM_MAX_DISPARITY)
    cost_volume_gpu = ssd(left.cuda(), right.cuda())
    cost_volume_cpu = ssd(left, right)

    torch.testing.assert_allclose(cost_volume_gpu.cpu(), cost_volume_cpu)
    torch.testing.assert_allclose(cost_volume_tex.cpu(), cost_volume_cpu)


def test_ssd_texture_result(sample_stereo_pair):
    """
    Tests the SSD with WTA.
    """
    left, right = sample_stereo_pair

    tex1 = CUDATexture.from_tensor(left)
    tex2 = CUDATexture.from_tensor(right)

    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)
    cost_volume = ssd_texture(tex1, tex2)

    wta = WinnerTakesAll()
    disparity_image = wta(cost_volume)
    save_depthmap(disparity_image,
                  pytest.STM_TEST_OUTPUT_PATH / "ssdtexture-wta")


@pytest.mark.benchmark(
    group="cost"
)
def test_benchmark_ssd_cpu(sample_stereo_pair, benchmark):
    """
    Benchmark SSD on CPU.
    """
    left, right = sample_stereo_pair[0], sample_stereo_pair[1]

    ssd = SSD(pytest.STM_MAX_DISPARITY)
    benchmark(ssd, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_bm_ssd_gpu(sample_stereo_pair, benchmark):
    """
    Benchmark SSD on GPU.
    """
    left, right = sample_stereo_pair[0].cuda(), sample_stereo_pair[1].cuda()

    ssd = SSD(pytest.STM_MAX_DISPARITY)
    benchmark(ssd, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_bm_ssd_texture(sample_stereo_pair, benchmark):
    """
    Benchmark the SSD using textures excluding the CPU to GPU upload.
    """
    left, right = sample_stereo_pair
    left = CUDATexture.from_tensor(left, normalized_coords=False)
    right = CUDATexture.from_tensor(right, normalized_coords=False)
    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)
    benchmark(ssd_texture, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_bm_upload_ssd_texture(sample_stereo_pair, benchmark):
    """
    Benchmark the SSD using textures including the CPU to GPU upload.
    """
    left, right = sample_stereo_pair
    ssd_texture = SSDTexture(pytest.STM_MAX_DISPARITY)

    def _target(left=left, right=right):
        left = CUDATexture.from_tensor(left, normalized_coords=False)
        right = CUDATexture.from_tensor(right, normalized_coords=False)
        ssd_texture(left, right)
    benchmark(_target)


def test_birchfield(sample_stereo_pair):
    """
    Saves sample and verifies the consistency of the Birchfield implementations.
    """
    left, right = sample_stereo_pair

    birchfield = Birchfield(pytest.STM_MAX_DISPARITY)

    cost_volume = birchfield(left, right)
    cost_volume_gpu = birchfield(left.cuda(), right.cuda())

    wta = WinnerTakesAll()
    save_depthmap(wta(cost_volume),
                  pytest.STM_TEST_OUTPUT_PATH / "birchfield-wta")
    save_depthmap(wta(cost_volume_gpu),
                  pytest.STM_TEST_OUTPUT_PATH / "birchfield-wta")

    torch.testing.assert_allclose(cost_volume, cost_volume_gpu.cpu())


@pytest.mark.benchmark(
    group="cost"
)
def test_bm_birchfield_gpu(sample_stereo_pair, benchmark):
    """
    GPU benchmark.
    """
    left, right = sample_stereo_pair[0].cuda(), sample_stereo_pair[1].cuda()
    birchfield = Birchfield(pytest.STM_MAX_DISPARITY)
    benchmark(birchfield, left, right)


@pytest.mark.benchmark(
    group="cost"
)
def test_bm_birchfield_cpu(sample_stereo_pair, benchmark):
    """
    CPU benchmark.
    """
    left, right = sample_stereo_pair[0], sample_stereo_pair[1]
    birchfield = Birchfield(pytest.STM_MAX_DISPARITY)
    benchmark(birchfield, left, right)
