"""
In Cuda, textures offer an alternative for faster data fetching when image access patterns are used.

This module offers mechanism to handle Cuda textures.
"""

import torch
from ._cstereomatch import CUDATexture as _CCUDATexture


class CUDATexture(_CCUDATexture):
    """
    CUDA Texture.
    """
    def __del__(self):
        self.release()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, normalized_coords: bool = False):
        """
        Creates a CUDA texture from a tensor.

        Args:
            tensor: A [HxW] or [HxWx3] tensor image.
            normalized_coords: Indicates whether texture reads are normalized or not. That is,
             kernels with normalized coords must:
             `tex2D<float>(text, float(col) / float(width), float(row) / float(height))`
        """
        sizes = tensor.size()
        texture = CUDATexture()
        texture.copy_from_tensor(tensor.view(
            sizes[0], sizes[1], -1), normalized_coords)
        return texture

    def __str__(self):
        return (f"CUDATexture(texture_object={self.cuda_texture_object}, "
                f"width={self.width}, height={self.height}, channels={self.channels})")

    def __repr__(self):
        return str(self)
