#pragma once

#include <limits>
#include <cmath>
#include <math_constants.h>

#include "accessor.hpp"

namespace stereomatch {

#ifdef __CUDACC__
template <typename scalar_t>
struct NumericLimits {};

template <>
struct NumericLimits<float> {
  static constexpr __device__ __host__ float infinity() noexcept { return HUGE_VALF; }
};

template <>
struct NumericLimits<double> {
  static constexpr __device__ __host__ double infinity() noexcept { return HUGE_VAL; }
};
#else
template <typename scalar_t>
struct NumericLimits {
  static constexpr scalar_t infinity() noexcept {
    return std::numeric_limits<scalar_t>::infinity();
  }
};
#endif

}  // namespace stereomatch
