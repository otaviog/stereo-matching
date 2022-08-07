"""
Tests the aggregation functions.
"""

import torch
import matplotlib.pyplot as plt

import stereomatch



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
