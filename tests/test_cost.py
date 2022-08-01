
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
import torch

from stereomatch.cuda_texture import CUDATexture
from stereomatch.cost import ssd_texture, ssd


@pytest.fixture
def images_rgb():
    image_base_dir = Path(__file__).parent.parent / \
        "test-data/middleburry/teddy/"

    left_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im2.png").convert('L'))).float() / 255.0
    right_image = torch.from_numpy(
        np.array(Image.open(image_base_dir / "im6.png").convert('L'))).float() / 255.0

    return left_image, right_image

# pylint: disable=redefined-outer-name


def test_ssd(images_rgb):
    left, right = images_rgb
    max_disparity = 20
    print("use texture")
    tex1 = CUDATexture.from_tensor(left, normalized_coords=False)
    tex2 = CUDATexture.from_tensor(right, normalized_coords=False)
    cost_volume_tex = ssd_texture(tex1, tex2, max_disparity)
    print("done texture")
    cost_volume_gpu = ssd(left.cuda(), right.cuda(), max_disparity)

    cost_volume_cpu = ssd(left, right, max_disparity)
    __import__("ipdb").set_trace()
    torch.testing.assert_allclose(cost_volume_gpu.cpu(), cost_volume_cpu)
    torch.testing.assert_allclose(cost_volume_tex.cpu(), cost_volume_cpu)

