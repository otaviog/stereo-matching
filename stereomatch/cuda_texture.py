from ._cstereomatch import CUDATexture as _CCUDATexture


class CUDATexture(_CCUDATexture):
    def __init__(self):
        pass

    @classmethod
    def from_tensor(cls, tensor):
        sizes = tensor.size()
        return _CCUDATexture.from_tensor(tensor.float().view(
            sizes[0], sizes[1], -1))

    def __str__(self):
        pass

    def __repr__(self):
        return str(self)
