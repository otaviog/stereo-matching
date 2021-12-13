from ._cstereomatch import CUDATexture as _CCUDATexture


class CUDATexture(_CCUDATexture):
    def __del__(self):
        self.release()

    @classmethod
    def from_tensor(cls, tensor, normalized_coords=False):
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
