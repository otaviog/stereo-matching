#pragma once

#include <limits>
#include <math_constants.h>

#include "accessor.hpp"

namespace stereomatch {

template <Device dev, typename scalar_t>
struct NumericLimits {};

#ifdef __CUDACC__
template <>
struct NumericLimits<kCUDA, float> {
  static inline __device__ float infinity() noexcept { return CUDART_INF_F; }
};

template <>
struct NumericLimits<kCUDA, double> {
  static inline __device__ float infinity() noexcept { return CUDART_INF; }
};
#endif

template <typename scalar_t>
struct NumericLimits<kCPU, scalar_t> {
  static constexpr scalar_t infinity() noexcept {
    return std::numeric_limits<scalar_t>::infinity();
  }
};
}  // namespace stereomatch
