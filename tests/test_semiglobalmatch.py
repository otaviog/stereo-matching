"""
Tests the semiglobal match
"""

import pytest
import matplotlib.pyplot as plt

import stereomatch


def _test_impl(ssd_cost, suffix):
    matcher = stereomatch.aggregation.SemiglobalAggregation()

    sgm_cost = matcher.estimate(
        ssd_cost.volume, ssd_cost.left_image.float())

    matcher2 = stereomatch.aggregation.WinnerTakesAll()
    depthmap = matcher2.estimate(sgm_cost).cpu().numpy()

    plt.figure()
    plt.imshow(depthmap)
    plt.axis('off')
    plt.savefig(f'sgm-{suffix}.png')


def test_cpu(ssd_cost):
    _test_impl(ssd_cost, "cpu")


def test_gpu(ssd_cost):
    _test_impl(ssd_cost.to("cuda:0"), "cuda")

@pytest.mark.benchmark(
    group="aggregation")
def test_benchmark_cpu_sgm(ssd_cost, benchmark):
    matcher = stereomatch.aggregation.SemiglobalAggregation()
    benchmark(matcher.estimate,
              ssd_cost.volume, ssd_cost.left_image.float())

@pytest.mark.benchmark(
    group="aggregation")
def test_benchmark_gpu_sgm(ssd_cost, benchmark):
    matcher = stereomatch.aggregation.SemiglobalAggregation()
    benchmark(matcher.estimate,
              ssd_cost.volume.cuda(), ssd_cost.left_image.float().cuda())
