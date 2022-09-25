"""
Dataset loaders and data utilities.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image
import torch


def _parse_middlebury_calib(filepath):
    props = {}
    with open(filepath, 'r', encoding='ascii') as file:
        for line in file:
            prop_name, prop_value = line.split('=')
            props[prop_name] = prop_value.strip()

    return dict(width=int(props["width"]),
                height=int(props["height"]),
                ndisp=int(props["ndisp"]))


class MiddleburyDataset:
    """
    Pytorch like dataset for parsing the Middlebury dataset format.
    """

    def __init__(self, dataset_dir: str, max_size: Optional[int] =None):
        """
        Args:
            dataset_dir: The dataset base directory.
            max_size: Optional maximum number of entries to read.
        """
        dataset_dir = Path(dataset_dir)

        if not dataset_dir.is_dir():
            raise RuntimeError(
                f"MiddleburyDataset: {dataset_dir} must be a directory")

        self.images = []
        self.disps = []
        self.calibs = []
        sample_dirs = natsorted(list(dataset_dir.iterdir()))

        if max_size is not None:
            sample_dirs = sample_dirs[:max_size]
        for sample_dir in sample_dirs:
            if not sample_dir.is_dir():
                continue

            self.images.append(
                (sample_dir / "im0.png", sample_dir / "im1.png"))
            self.disps.append(
                (sample_dir / "disp0.pfm", sample_dir / "disp1.pfm"))
            self.calibs.append(_parse_middlebury_calib(
                sample_dir / "calib.txt"))

    def __getitem__(self, idx:int):
        item = self.get_stereo_pair(idx)
        item.update(self.get_ground_truth(idx))
        return item

    def get_stereo_pair(self, idx:int):
        """
        Gets a stereo pair.
        """
        left_img_path, right_img_path = self.images[idx]

        left_img = np.array(Image.open(left_img_path))
        right_img = np.array(Image.open(right_img_path))

        return dict(
            stereo_name=left_img_path.parent.name,
            left=torch.from_numpy(left_img),
            right=torch.from_numpy(right_img),
            max_disparity=self.calibs[idx]["ndisp"])

    def get_ground_truth(self, idx):
        """
        Gets the ground-truth depthmap from an entry.
        """
        disp_path = self.disps[idx][0]
        disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
        return dict(
            stereo_name=disp_path.parent.name,
            gt_disparity=torch.from_numpy(disp),
            max_disparity=self.calibs[idx]["ndisp"])

    def __len__(self):
        return len(self.images)
