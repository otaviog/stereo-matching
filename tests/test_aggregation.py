from pathlib import Path

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

import stereomatch


class AbstractAggregationTestCase:
    def get_ssd_cost_volume(self):
        cache_file = Path(__file__).parent / \
            "test_cache/cost_volume_teddy.torch"

        if cache_file.exists():
            return torch.load(str(cache_file))

        image_base_dir = Path(__file__).parent.parent / \
            "test-data/middleburry/teddy/"

        left_image = torch.from_numpy(
            np.array(Image.open(image_base_dir / "im2.png").convert('L')))
        right_image = torch.from_numpy(
            np.array(Image.open(image_base_dir / "im6.png").convert('L')))

        cost_volume = stereomatch.cost.ssd(
            left_image, right_image, 100)

        cache_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(cost_volume, str(cache_file))
        return cost_volume


class aTestWinnersTakeAll(AbstractAggregationTestCase):
    def test_cpu(self):
        cost_volume = self.get_ssd_cost_volume()
        matcher = stereomatch.aggregation.WinnerTakesAll()
        depthmap = matcher.estimate(cost_volume).cpu().numpy()

        fig = plt.figure()
        plt.imshow(depthmap)
        plt.savefig('wta-cpu.png')


class TestDynamicProgramming(AbstractAggregationTestCase):
    def _run_test(self, device):
        cost_volume = self.get_ssd_cost_volume().to(device)
        matcher = stereomatch.aggregation.DynamicProgramming()
        depthmap = matcher.estimate(cost_volume).cpu().numpy()

        plt.figure()
        plt.imshow(depthmap)
        plt.savefig(f'dynamic_programming-{device}.png')

    def test_cpu(self):
        self._run_test("cpu")

    def test_cuda(self):
        self._run_test("cuda")
