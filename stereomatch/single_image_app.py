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


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left_image", metavar="left-image")
    parser.add_argument("right_image", metavar="right-image")
    parser.add_argument("max_disparity", metavar="max-disparity", type=int)
    parser.add_argument("output_depthmap", metavar="output-depthmap")
    parser.add_argument("-g", "--ground-truth")
    parser.add_argument("-c", "--cuda-on", action="store_true")
    parser.add_argument("-sd", "--show-depthmap", action="store_true")

    args = parser.parse_args()

    left_image = torch.from_numpy(
        np.array(Image.open(args.left_image).convert('L')))
    right_image = torch.from_numpy(
        np.array(Image.open(args.right_image).convert('L')))

    if args.cuda_on:
        left_image = left_image.to("cuda:0")
        right_image = right_image.to("cuda:0")

    cost_volume = stereomatch.cost.ssd(
        left_image, right_image, args.max_disparity)

    matcher = stereomatch.aggregation.WinnerTakesAll()
    depthmap = matcher.estimate(cost_volume)

    depthmap = depthmap.cpu().numpy().astype(np.uint16)

    if args.show_depthmap:
        plt.imshow(depthmap)
        plt.show()

    imageio.imsave(args.output_depthmap, depthmap)


if __name__ == '__main__':
    _main()
