#pragma once

#include "accessor.hpp"
#include "cuda_utils.hpp"

namespace stereomatch {

template <Device dev>
struct KernelLauncher {};

#ifdef __CUDACC__
/**
 * Used by KernelLauncher<kCUDA>
 *
 */
template <typename Kernel>
static __global__ void Exec1DKernel(Kernel kern, int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    kern(idx);
  }
}

template <typename Kernel>
static __global__ void Exec2DKernel(Kernel kern, int width, int height) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    kern(row, col);
  }
}

template <>
struct KernelLauncher<kCUDA> {
  /**
   * Launches a kernel functor on CUDA. The launch configuration is
   * determined automatically by the given size.
   *
   * After execution errors are verified and the device is synchronized.
   *
   * @param kern Kernel functor.
   * @param size the number of elements to process.
   */
  template <typename Kernel>
  static void Launch1D(Kernel &kern, int size, bool sequential = false) {
    CudaKernelDims kl = Get1DKernelDims(size);
    Exec1DKernel<<<kl.grid, kl.block>>>(kern, size);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
  }


  template <typename Kernel>
  static void Launch2D(Kernel &kern, int width, int height,
                       bool sequential = false) {
    CudaKernelDims kl = Get2DKernelDims(width, height);
    Exec2DKernel<<<kl.grid, kl.block>>>(kern, width, height);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
  }

  template <typename Kernel>
  static void Launch2D(Kernel &kern, int width, int height,
                       dim3 grid_size, dim3 block_size) {
    Exec2DKernel<<<grid_size, block_size>>>(kern, width, height);
    CudaCheck();
    CudaSafeCall(cudaDeviceSynchronize());
  }
};
#endif

template <>
struct KernelLauncher<kCPU> {
  template <typename Kernel>
  static void Launch1D(Kernel &kern, int size, bool sequential = false) {
    if (!sequential) {
#ifdef NDEBUG
#pragma omp parallel for
#endif
      for (int i = 0; i < size; ++i) {
        kern(i);
      }
    } else {
      for (int i = 0; i < size; ++i) {
        kern(i);
      }
    }
  }

  template <typename Kernel>
  static void Launch2D(Kernel &kern, int width, int height,
                       bool sequential = false) {
    if (!sequential) {
#pragma omp parallel for
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          kern(row, col);
        }
      }
    } else {
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          kern(row, col);
        }
      }
    }
  }
};

}  // namespace slamtb

#define STM_DISPATCH_KERNEL_FLOATING_TYPES(TYPE, DEVICE, NAME, ...) \
  if (DEVICE.is_cuda()) {                                           \
    const Device device = kCUDA;                                    \
    AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);            \
  } else {                                                          \
    const Device device = kCPU;                                     \
    AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__);            \
  }

#define STM_DISPATCH_KERNEL_ALL_TYPES(TYPE, DEVICE, NAME, ...) \
  if (DEVICE.is_cuda()) {                                      \
    const Device device = kCUDA;                               \
    AT_DISPATCH_ALL_TYPES(TYPE, NAME, __VA_ARGS__);            \
  } else {                                                     \
    const Device device = kCPU;                                \
    AT_DISPATCH_ALL_TYPES(TYPE, NAME, __VA_ARGS__);            \
  }

#define STM_DISPATCH_KERNEL(DEVICE, ...) \
  if (DEVICE.is_cuda()) {                \
    const Device device = kCUDA;         \
    __VA_ARGS__();                       \
  } else {                               \
    const Device device = kCPU;          \
    __VA_ARGS__();                       \
  }
