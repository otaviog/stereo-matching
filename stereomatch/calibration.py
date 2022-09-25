"""
Camera calibration tools.
"""

from typing import Tuple

import cv2
import numpy as np


class StereoRectifier:
    def __init__(self, hmgp0: np.ndarray, hmgp1: np.ndarray):
        self.hmgp0 = hmgp0
        self.hmgp1 = hmgp1

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(state_dict["homography0"], state_dict["homography1"])

    def get_state_dict(self):
        return {
            "homography0": self.hmgp0,
            "homography1": self.hmgp1
        }

    def __call__(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        return (cv2.warpPerspective(img0, self.hmgp0, (w0, h0)),
                cv2.warpPerspective(img1, self.hmgp1, (w1, h1)))

    def invert(self, rectified_img: np.ndarray, img_idx: int) -> np.ndarray:
        hmgp = (self.hmgp0, self.hmgp1)[img_idx]
        h, w = rectified_img.shape[:2]

        return cv2.warpPerspective(rectified_img, hmgp, (w, h), flags=cv2.WARP_INVERSE_MAP)
