#!/usr/bin/env python
"""
CLI for estimating disparity from stereo videos.
"""
import argparse
import pickle

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from .capture import StereoCapture
from .calibration import StereoRectifier
from .cli_common import (create_pipeline, COST_METHODS, AGGREGATION_METHODS,
                        DISPARITY_METHODS)


def _print_instructions():
    print("""Keys:
                  q/Q: Quit the execution.
                  h/H: Show this help message.
    """)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_mode", choices=[
                        "dev", "file"], metavar="input-mode",
                        help="Opens either a camera `dev`ice or reads from a video `file`.")
    parser.add_argument("input", type=str,
                        help="Input source use an integer for devices or strings for video files.")
    parser.add_argument("max_disparity", metavar="max-disparity", type=int,
                        help="Maximum disparity")

    parser.add_argument("-cal", "--calib", help="Calibration pickle.")

    parser.add_argument("-cm", "--cost-method", choices=COST_METHODS.keys(),
                        default="ssd")
    parser.add_argument("-am", "--aggregation-method",
                        choices=AGGREGATION_METHODS.keys(), default=None)
    parser.add_argument("-dm", "--disparity-method",
                        choices=DISPARITY_METHODS.keys(), default="wta")

    parser.add_argument("-c", "--cuda-on", action="store_true", help="Wether it should use cuda")

    return parser.parse_args()

def main():
    """
    Entry point.
    """
    # pylint: disable=too-many-branches,too-many-statements
    args = _parse_args()
    pipeline = create_pipeline(args.cost_method,
                               args.disparity_method,
                               args.aggregation_method)

    capture = dict(
        dev=lambda dev_idx: StereoCapture.from_device(int(dev_idx)),
        file=StereoCapture.from_file)[args.input_mode](args.input)
    rectifier = None
    if args.calib:
        with open(args.calib, 'rb') as file:
            rectifier = StereoRectifier.from_state_dict(pickle.load(file))

    device = "cpu"
    if args.cuda_on:
        device = "cuda:0"

    _print_instructions()
    cmap = plt.get_cmap("rainbow", args.max_disparity)
    do_quit = False
    show_rectified = show_rgb = False
    pause = False

    while not do_quit:
        if not pause:
            ok, cap = capture.read_next()
        if not ok:
            do_quit = True
            break

        joined = cap.joined
        frame0, frame1, _ = cap.to_grayscale()

        if rectifier is not None:
            frame0, frame1 = rectifier(frame0, frame1)

        if show_rectified:
            cv2.imshow("rgb", joined)

        if show_rgb:
            cv2.imshow("rectified", np.hstack([frame0, frame1]))

        depthmap = pipeline.estimate(
            torch.from_numpy(frame0).to(torch.float32).to(device),
            torch.from_numpy(frame1).to(torch.float32).to(device)
        ).cpu().numpy()

        rgb_depthmap = (cmap(depthmap)[:, :, :3]*255).astype(np.uint8)
        # rgb_depthmap = rectifier.invert(rgb_depthmap, 1)

        cv2.imshow("depthmap", rgb_depthmap)
        key = cv2.waitKey(1)
        chr_key = chr(key & 0xff).lower()

        if chr_key == 'q':
            do_quit = True
        elif chr_key == 'h':
            _print_instructions()
        elif chr_key == 'i':
            plt.imshow(depthmap)
            plt.show()
        elif chr_key == 'w':
            show_rectified = not show_rectified
            if not show_rectified:
                cv2.destroyWindow("rectified")
        elif chr_key == 'e':
            show_rgb = not show_rgb
            if not show_rgb:
                cv2.destroyWindow("rgb")
        elif chr_key == 'r':
            pause = not pause

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
