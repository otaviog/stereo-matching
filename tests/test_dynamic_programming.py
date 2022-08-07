"""
Tests the dynamic programming aggregation.
"""

import pytest
import torch

import stereomatch

from viz import save_depthmap


def _run_test(cost_volume):
    matcher = stereomatch.aggregation.DynamicProgramming()
    depthmap = matcher.estimate(cost_volume)

    save_depthmap(depthmap, "dynprog")
    return depthmap


def test_cpu(ssd_cost):
    _run_test(ssd_cost.volume)


def test_gpu(ssd_cost):
    cpu_result = _run_test(ssd_cost.volume)
    gpu_result = _run_test(ssd_cost.volume.to("cuda"))

    torch.testing.assert_allclose(cpu_result, gpu_result.cpu())


def _benchmark_dynprog(cost_volume, benchmark):
    matcher = stereomatch.aggregation.DynamicProgramming()
    benchmark(matcher.estimate, cost_volume)

    
@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_dynprog_cpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (CPU implementation).
    """
    _benchmark_dynprog(ssd_cost.volume, benchmark)


@pytest.mark.benchmark(
    group="aggregation"
)
def test_benchmark_dynprog_gpu(ssd_cost, benchmark):
    """
    Benchmarks the winners takes all (GPU implementation).
    """
    _benchmark_dynprog(ssd_cost.volume.to("cuda:0"), benchmark)
