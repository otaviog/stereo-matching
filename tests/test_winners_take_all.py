"""
Benchmark the winners takes all based matching.
"""

import pytest
import torch

import stereomatch

from viz import save_depthmap


def _test(ssd_cost):
    matcher = stereomatch.aggregation.WinnerTakesAll()
    depthmap = matcher.estimate(ssd_cost.volume)
    save_depthmap(depthmap, "wta")
    return depthmap


def test_cpu(ssd_cost):
    """
    Tests winners take all (CPU implementation).
    """
    _test(ssd_cost)


def test_gpu(ssd_cost):
    """
    Tests winners take all (GPU implementation).
    """

    gpu_result = _test(ssd_cost.to("cuda:0"))

    sample_volume = torch.arange(300*300*128).reshape(300, 300, 128).float()
    matcher = stereomatch.aggregation.WinnerTakesAll()
    torch.testing.assert_allclose(
        matcher.estimate(sample_volume),
        matcher.estimate(sample_volume.to("cuda:0")).cpu())


def _benchmark_wta(ssd_cost, benchmark):
    matcher = stereomatch.aggregation.WinnerTakesAll()
    benchmark(matcher.estimate, ssd_cost.volume)


@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_wta_cpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (CPU implementation).
    """
    _benchmark_wta(ssd_cost, benchmark)


@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_wta_gpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (GPU implementation).
    """
    _benchmark_wta(ssd_cost.to("cuda:0"), benchmark)


@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_wta_with_argmax_cpu(ssd_cost, benchmark):
    """
    Benchmarks the torch.argmax as winners takes all implementation (CPU).
    """
    benchmark(torch.argmax, ssd_cost.volume, 2)


@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_wta_with_argmax_gpu(ssd_cost, benchmark):
    """
    Benchmarks the torch.argmax as winners takes all implementation (CPU).
    """
    benchmark(torch.argmax, ssd_cost.volume.to("cuda:0"), 2)
