#!/usr/bin/env python
"""
CLI for estimating disparity from stereo images.
"""
import argparse

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from .cost import SSD, SSDTexture, Birchfield
from .aggregation import Semiglobal
from .disparity_reduce import WinnerTakesAll, DynamicProgramming
from .pipeline import Pipeline


COST_METHODS = {
    "ssd": SSD,
    "ssd-texture": SSDTexture,
    "birchfield": Birchfield
}

AGGREGATION_METHODS = {
    "sgm": Semiglobal,
}

DISPARITY_METHODS = {
    "wta": WinnerTakesAll,
    "dyn": DynamicProgramming
}


def main():
    """
    CLI for estimating disparity from stereo images.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_image", metavar="left-image", help="Left image")
    parser.add_argument("right_image", metavar="right-image", help="Right image")
    parser.add_argument("max_disparity", metavar="max-disparity", type=int,
                        help="Maximum disparity for stereo matching.")
    parser.add_argument("output_depthmap", metavar="output-depthmap",
                        help="Output file of the depth map.")
    parser.add_argument("-cm", "--cost-method", choices=COST_METHODS.keys(),
                        default="ssd", help="Cost function of the stereo pair.")
    parser.add_argument("-am", "--aggregation-method",
                        choices=AGGREGATION_METHODS.keys(), default=None,
                        help="Aggregation method to improve the cost function.")
    parser.add_argument("-dm", "--disparity-method",
                        choices=DISPARITY_METHODS.keys(), default="wta",
                        help="Disparity method for matching the pixels")
    parser.add_argument("-c", "--cuda-on", action="store_true",
                        help="Whether to use Cuda-based GPU acceleration.")
    parser.add_argument("-sd", "--show-depthmap", action="store_true",
                        help="Shows depthmap.")

    args = parser.parse_args()
    aggregation_method = AGGREGATION_METHODS.get(args.aggregation_method, None)
    if aggregation_method is not None:
        aggregation_method = aggregation_method()

    pipeline = Pipeline(COST_METHODS[args.cost_method](args.max_disparity),
                        DISPARITY_METHODS[args.disparity_method](),
                        aggregation=aggregation_method)

    left_image = torch.from_numpy(
        np.array(Image.open(args.left_image).convert('L'))).float()
    right_image = torch.from_numpy(
        np.array(Image.open(args.right_image).convert('L'))).float()

    if args.cuda_on:
        left_image = left_image.to("cuda:0")
        right_image = right_image.to("cuda:0")

    depthmap = pipeline.estimate(
        left_image, right_image).cpu().numpy().astype(np.uint16)

    plt.figure()
    plt.imshow(depthmap)
    plt.axis("off")
    if args.show_depthmap:
        plt.show()

    plt.savefig(args.output_depthmap)
    plt.close()


if __name__ == '__main__':
    main()
