"""
Simple script for testing with a single image.
"""

import argparse

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

import stereomatch


def _ssd_texture_wrapper(left_image: torch.Tensor, right_image: torch.Tensor,
                         max_disparity: int):
    return stereomatch.cost.ssd_texture(
        stereomatch.cuda_texture.CUDATexture.from_tensor(left_image.cuda()),
        stereomatch.cuda_texture.CUDATexture.from_tensor(right_image.cuda()),
        max_disparity)


COST_METHODS = {
    "ssd": stereomatch.cost.ssd,
    "ssd-texture": _ssd_texture_wrapper,
    "birchfield": stereomatch.cost.birchfield
}

AGGREGATION_METHODS = {
    "wta": stereomatch.aggregation.WinnerTakesAll,
    "dyn": stereomatch.aggregation.DynamicProgramming
}


class Pipeline:
    def __init__(self, cost, aggregation, max_disparity):
        self.cost = cost
        self.aggregation = aggregation
        self.max_disparity = max_disparity

        self._cost_volume = None
        self._disparity_image = None

    def estimate(self, left_image, right_image):
        self._cost_volume = self.cost(left_image, right_image,
                                      self.max_disparity, self._cost_volume)
        self._disparity_image = self.aggregation(self._cost_volume,
                                                 self._disparity_image)
        return self._disparity_image


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image", metavar="left-image")
    parser.add_argument("right_image", metavar="right-image")
    parser.add_argument("max_disparity", metavar="max-disparity", type=int)
    parser.add_argument("output_depthmap", metavar="output-depthmap")
    parser.add_argument("-cm", "--cost-method", choices=COST_METHODS.keys(),
                        default="ssd")
    parser.add_argument("-am", "--aggregation-method",
                        choices=AGGREGATION_METHODS.keys(), default="wta")
    parser.add_argument("-g", "--ground-truth")
    parser.add_argument("-c", "--cuda-on", action="store_true")
    parser.add_argument("-sd", "--show-depthmap", action="store_true")

    args = parser.parse_args()

    cost_method = COST_METHODS[args.cost_method]
    aggregation_method = AGGREGATION_METHODS[args.aggregation_method]()

    left_image = torch.from_numpy(
        np.array(Image.open(args.left_image).convert('L')))
    right_image = torch.from_numpy(
        np.array(Image.open(args.right_image).convert('L')))

    if args.cuda_on:
        left_image = left_image.to("cuda:0")
        right_image = right_image.to("cuda:0")

    cost_volume = cost_method(
        left_image, right_image, args.max_disparity)

    depthmap = aggregation_method.estimate(cost_volume)

    depthmap = depthmap.cpu().numpy().astype(np.uint16)

    if args.show_depthmap:
        plt.imshow(depthmap)
        plt.show()

    imageio.imsave(args.output_depthmap, depthmap)


if __name__ == '__main__':
    _main()
