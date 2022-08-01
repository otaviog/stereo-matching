"""
Tests the aggregation functions.
"""

from pathlib import Path
import dataclasses

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import pytest

import stereomatch


@dataclasses.dataclass
class CostFixture:
    volume: torch.Tensor
    left_image: torch.Tensor


@pytest.fixture
def ssd_cost():
    cache_file = Path(__file__).parent / \
        "test_cache/cost_volume_teddy.torch"

    image_base_dir = Path(__file__).parent.parent / \
        "test-data/middleburry/teddy/"

    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im2.png").convert('L')))

    if cache_file.exists():
        return CostFixture(volume=torch.load(str(cache_file)),
                           left_image=left_image)

    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im6.png").convert('L')))

    cost_volume = stereomatch.cost.ssd(
        left_image, right_image, 100)

    cache_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(cost_volume, str(cache_file))
    return CostFixture(volume=cost_volume, left_image=left_image)


class TestWinnersTakeAll:
    @staticmethod
    def test_cpu(ssd_cost):
        matcher = stereomatch.aggregation.WinnerTakesAll()
        depthmap = matcher.estimate(ssd_cost.volume).cpu().numpy()

        fig = plt.figure()
        plt.imshow(depthmap)
        plt.savefig('wta-cpu.png')


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


class TestSGM:
    @staticmethod
    def test_cpu(ssd_cost):
        matcher = stereomatch.aggregation.SemiglobalAggregation()

        sgm_cost = matcher.estimate(
            ssd_cost.volume, ssd_cost.left_image.float())

        matcher2 = stereomatch.aggregation.WinnerTakesAll()
        depthmap = matcher2.estimate(sgm_cost).cpu().numpy()

        plt.figure()
        plt.imshow(depthmap)
        plt.savefig('sgm-cpu.png')

    @staticmethod
    def test_benchmark(ssd_cost, benchmark):
        matcher = stereomatch.aggregation.SemiglobalAggregation()
        benchmark(matcher.estimate,
                  ssd_cost.volume, ssd_cost.left_image.float())
