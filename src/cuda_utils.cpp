#include "cuda_utils.hpp"

namespace stereomatch {
CudaKernelDims Get2DKernelDims(int width, int height) {
  dim3 block_dim = dim3(16, 16);
  dim3 grid_size(width / block_dim.x + 1, height / block_dim.y + 1);
  return CudaKernelDims(grid_size, block_dim);
}

CudaKernelDims Get1DKernelDims(int size) {
  const int block_size = 256;
  const int num_blocks = size / block_size + 1;

  return CudaKernelDims(dim3(num_blocks), dim3(block_size));
}
}  // namespace stereomatch
