"""
Tests the aggregation functions.
"""

import torch
import matplotlib.pyplot as plt

import stereomatch


class TestWinnersTakeAll:
    @staticmethod
    def test_cpu(ssd_cost):
        matcher = stereomatch.aggregation.WinnerTakesAll()
        depthmap = matcher.estimate(ssd_cost.volume).cpu().numpy()

        fig = plt.figure()
        plt.imshow(depthmap)
        plt.savefig('wta-cpu.png')

    @staticmethod
    def test_benchmark_wta(ssd_cost, benchmark):
        matcher = stereomatch.aggregation.WinnerTakesAll()
        benchmark(matcher.estimate, ssd_cost.volume)

    @staticmethod
    def test_benchmark_torch(ssd_cost, benchmark):
        benchmark(torch.argmax, ssd_cost.volume, 2)


class TestDynamicProgramming:
    @staticmethod
    def _run_test(cost_volume, device):
        cost_volume = cost_volume.to(device)
        matcher = stereomatch.aggregation.DynamicProgramming()
        depthmap = matcher.estimate(cost_volume).cpu().numpy()

        plt.figure()
        plt.imshow(depthmap)
        plt.savefig(f'dynamic_programming-{device}.png')

    @staticmethod
    def test_cpu(ssd_cost):
        TestDynamicProgramming._run_test(ssd_cost.volume, "cpu")

    @staticmethod
    def test_cuda(ssd_cost):
        TestDynamicProgramming._run_test(ssd_cost.volume, "cuda")
