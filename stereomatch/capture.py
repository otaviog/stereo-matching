"""
Stereo capture functions.
"""
from dataclasses import dataclass
from typing import Tuple, Union
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoCaptureImage:
    """
    Data type to hold stereo images.

    Expected format of the arrays are BGR [HxWx3] with uint8 types.
    """
    left: np.ndarray = None
    right: np.ndarray = None
    joined: np.ndarray = None

    def __iter__(self):
        return iter((self.left, self.right, self.joined))

    def to_grayscale(self):
        """
        Convert the images to grayscale.
        """
        return StereoCaptureImage(
            cv2.cvtColor(self.left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.right, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.joined, cv2.COLOR_BGR2GRAY))


class StereoCapture:
    """
    Wrapper around OpenCV's VideoCapture to read stereo videos.
    """
    def __init__(self, video_capture: cv2.VideoCapture):
        self.video_capture = video_capture

    def __del__(self):
        self.close()

    @classmethod
    def from_device(cls, dev_idx: int):
        """
        Opens the stream from a device.

        Args:
            dev_idx: Device index number.
        """
        cap = cv2.VideoCapture(dev_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera {dev_idx}")
        return cls(cap)

    @classmethod
    def from_file(cls, filepath: Union[str, Path]):
        """
        Opens a video file.

        Args:
            filepath: Video file path.
        """
        filepath = str(filepath)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open file {filepath}")
        return cls(cap)

    def read_next(self) -> Tuple[bool, StereoCaptureImage]:
        """
        Reads the next frame.

        Returns:
            status: `True` if the call was sussceful.
            image: Stereo image (or None).
        """
        ok, frame = self.video_capture.read()
        if not ok:
            return False, StereoCaptureImage()

        half_width = frame.shape[1] // 2
        if True:
            return True, StereoCaptureImage(
                frame[:, :half_width, :],
                frame[:, half_width:, :],
                frame)
        else:
            return True, StereoCaptureImage(
                frame[:, half_width:, :],
                frame[:, :half_width, :],
                frame)

    def close(self):
        """
        Closes the stream.
        """
        self.video_capture.release()
