#pragma once

#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <torch/torch.h>

// Taken from:
// https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/

#define CudaSafeCall(err) stereomatch::_CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheck() stereomatch::_CudaCheck(__FILE__, __LINE__)

#ifdef __CUDACC__
#define STB_DEVICE __device__
#define STB_DEVICE_HOST __device__ __host__
#else
#define STB_DEVICE
#define STB_DEVICE_HOST
#endif
namespace stereomatch {

inline void _CudaSafeCall(cudaError err, const char *file, const int line) {
  if (err != cudaSuccess) {
    std::stringstream msg;
    msg << "Cuda call error " << file << "(" << line
        << "): " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}
inline void _CudaCheck(const char *file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::stringstream msg;
    msg << "Cuda check error " << file << "(" << line
        << "): " << cudaGetErrorString(err);
    throw std::runtime_error(msg.str());
  }
}

struct CudaKernelDims {
  CudaKernelDims(dim3 grid_dims, dim3 block_dims)
      : grid(grid_dims), block(block_dims) {}

  dim3 grid;
  dim3 block;
};

CudaKernelDims Get1DKernelDims(int size);

CudaKernelDims Get2DKernelDims(int width, int height);

}  // namespace stereomatch
