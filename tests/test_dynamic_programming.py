"""
Tests the dynamic programming disparity_reduce.
"""

import pytest
import torch

import stereomatch

from .viz import save_depthmap


def _run_test(cost_volume):
    matcher = stereomatch.disparity_reduce.DynamicProgramming()
    depthmap = matcher(cost_volume)

    save_depthmap(depthmap, pytest.STM_TEST_OUTPUT_PATH / "dynprog")
    return depthmap


def test_cpu(ssd_cost):
    """
    Runs the dynamic programing CPU implementations. It will output the disparity image.
    """
    _run_test(ssd_cost.volume)


def test_gpu(ssd_cost):
    """
    Runs the dynamic programing GPU implementations. It will output the disparity image.
    """
    _run_test(ssd_cost.volume.to("cuda"))


def test_cpu_gpu_should_equal():
    """
    Asserts that the dynamic programming have equivalent implementations
    on GPU and CPU.
    """
    # Because of the parallel reductions, we need to test them using
    # a cost volume without equal values.
    sample_volume = torch.arange(300*300*128).reshape(300, 300, 128).float()
    matcher = stereomatch.disparity_reduce.WinnerTakesAll()

    lfs = matcher(sample_volume)
    rhs = matcher(sample_volume.to("cuda:0")).cpu()

    torch.testing.assert_allclose(lfs, rhs)


def _benchmark_dynprog(cost_volume, benchmark):
    matcher = stereomatch.disparity_reduce.DynamicProgramming()
    benchmark(matcher, cost_volume)


@pytest.mark.benchmark(
    group="disparity_reduce"
)
def test_bm_dynprog_cpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (CPU implementation).
    """
    _benchmark_dynprog(ssd_cost.volume, benchmark)


@pytest.mark.benchmark(
    group="disparity_reduce"
)
def test_bm_dynprog_gpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (GPU implementation).
    """
    _benchmark_dynprog(ssd_cost.volume.to("cuda:0"), benchmark)
