"""
Tests the semiglobal match
"""

import pytest
import torch

import stereomatch

from viz import save_depthmap


def _test_impl(ssd_cost, suffix):
    sgm = stereomatch.aggregation.Semiglobal()

    sgm_cost = sgm(
        ssd_cost.volume, ssd_cost.left_image.float())

    reducer = stereomatch.disparity_reduce.WinnerTakesAll()
    save_depthmap(reducer(sgm_cost), pytest.STM_TEST_OUTPUT_PATH / "sgm")


def test_cpu(ssd_cost):
    """
    Integration test of the semiglobal CPU method. Outputs a depthmap into the
    test results folder.
    """
    _test_impl(ssd_cost, "cpu")


def test_gpu(ssd_cost):
    """
    Integration test of the semiglobal GPU method. Outputs a depthmap into the
    test results folder.
    """
    _test_impl(ssd_cost.to("cuda:0"), "cuda")


def test_cpu_gpu_should_equal():
    """
    Tests whether the semiglobal implementation in CPU and GPU outputs the same
    results.
    """
    # Because of the parallel reductions, we need to test them using
    # a cost volume without equal values.
    sample_image = torch.ones(300, 300).float()
    sample_volume = torch.arange(300*300*128).reshape(300, 300, 128).float()
    sgm = stereomatch.aggregation.Semiglobal()

    sgm_cost = sgm(
        sample_volume, sample_image.float())
    sgm_cost_gpu = sgm(
        sample_volume.cuda(), sample_image.float().cuda())

    torch.testing.assert_allclose(sgm_cost, sgm_cost_gpu.cpu())


@pytest.mark.benchmark(
    group="aggregation")
def test_bm_sgm_cpu(ssd_cost, benchmark):
    """
    Benchmarks the CPU implementation of the semiglobal method.
    """
    matcher = stereomatch.aggregation.Semiglobal()
    benchmark(matcher,
              ssd_cost.volume, ssd_cost.left_image.float())


@pytest.mark.benchmark(
    group="aggregation")
def test_bm_sgm_gpu(ssd_cost, benchmark):
    """
    Benchmarks the GPU implementation of the semiglobal method.
    """
    matcher = stereomatch.aggregation.Semiglobal()
    benchmark(matcher,
              ssd_cost.volume.cuda(), ssd_cost.left_image.float().cuda())
