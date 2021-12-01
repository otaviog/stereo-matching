#pragma once

#include <torch/torch.h>

namespace stereomatch {

enum Device { kCPU = 0, kCUDA = 1 };

template <Device dev, typename scalar_t, unsigned long dims>
struct Accessor {
  typedef torch::TensorAccessor<scalar_t, dims> T;
  typedef torch::TensorAccessor<scalar_t, dims> Ts;

  static T Get(torch::Tensor &tensor) {
    return tensor.accessor<scalar_t, dims>();
  }

  static T Get(const torch::Tensor &tensor) {
    return tensor.accessor<scalar_t, dims>();
  }
};

template <typename scalar_t, unsigned long dims>
using CPUAccessor = Accessor<kCPU, scalar_t, dims>;

#ifdef __CUDACC__
template <typename scalar_t, unsigned long dims>
struct Accessor<kCUDA, scalar_t, dims> {
  typedef torch::PackedTensorAccessor32<scalar_t, dims,
                                        torch::RestrictPtrTraits>
      T;

  typedef torch::TensorAccessor<scalar_t, dims, torch::RestrictPtrTraits,
                                int32_t>
      Ts;

  static T Get(torch::Tensor &tensor) {
    return tensor.packed_accessor32<scalar_t, dims, torch::RestrictPtrTraits>();
  }

  static T Get(const torch::Tensor &tensor) {
    return tensor.packed_accessor32<scalar_t, dims, torch::RestrictPtrTraits>();
  }
};

template <typename scalar_t, unsigned long dims>
using CUDAAccessor = Accessor<kCUDA, scalar_t, dims>;
#endif
}  // namespace stereomatch
