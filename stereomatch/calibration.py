"""
Camera calibration tools.
"""

from typing import Tuple

import cv2
import numpy as np


class StereoRectifier:
    """
    Stereo rectification using homography pairs.

    Attributes:
        homography0: Homography for the first camera
        homography1: Homography for the second camera.
    """
    def __init__(self, homography0: np.ndarray, homography1: np.ndarray):
        self.homography0 = homography0
        self.homography1 = homography1

    @classmethod
    def from_state_dict(cls, state_dict):
        """
        Loads the rectification using a state dictionary containing homography 0 and 1
        """
        return cls(state_dict["homography0"], state_dict["homography1"])

    def get_state_dict(self):
        """
        Gets the state dictionary.
        """
        return {
            "homography0": self.homography0,
            "homography1": self.homography1
        }

    def __call__(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo pair.

        Args:
            img0: left eye image.
            img1: right eye image.

        Returns:
            Both input images rectified.
        """
        height0, width0 = img0.shape[:2]
        height1, width1 = img1.shape[:2]

        return (cv2.warpPerspective(img0, self.homography0, (width0, height0)),
                cv2.warpPerspective(img1, self.homography1, (width1, height1)))

    def invert(self, rectified_img: np.ndarray, stereo_view_idx: int) -> np.ndarray:
        """
        Invert the rectification for one image.

        Args:
            rectified_img: The target image.
            stereo_view: Which eye the image is from.
             Use either 0 for the left eye or 1 for the right eye.
        Returns:
            Unrectified image.
        """

        assert 0 <= stereo_view_idx <= 1
        hmgp = (self.homography0, self.homography1)[stereo_view_idx]
        h, w = rectified_img.shape[:2]

        return cv2.warpPerspective(rectified_img, hmgp, (w, h), flags=cv2.WARP_INVERSE_MAP)
