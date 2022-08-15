#pragma once

#include <math_constants.h>
#include <cmath>
#include <limits>

#include "accessor.hpp"

namespace stereomatch {

#ifdef __CUDACC__
template <typename scalar_t>
struct NumericLimits {};

template <>
struct NumericLimits<float> {
  static constexpr __device__ __host__ float infinity() noexcept {
    return HUGE_VALF;
  }
};

template <>
struct NumericLimits<double> {
  static constexpr __device__ __host__ double infinity() noexcept {
    return HUGE_VAL;
  }
};

template <>
struct NumericLimits<int32_t> {
  static constexpr __device__ __host__ int32_t infinity() noexcept {
    return INT_MAX;
  }
};

template <>
struct NumericLimits<uint8_t> {
  static constexpr __device__ __host__ uint8_t infinity() noexcept {
    return 255;
  }
};
#else
template <typename scalar_t>
struct NumericLimits {
  static constexpr scalar_t infinity() noexcept {
    return std::numeric_limits<scalar_t>::infinity();
  }
};
#endif

template <typename T>
constexpr T get_min(T v1, T v2, T v3, T v4) {
  return fmin(v1, fmin(v2, fmin(v3, v4)));
}

template <typename T>
constexpr T get_min(T v1, T v2, T v3) {
  return fmin(v1, fmin(v2, v3));
}

template <typename T>
constexpr T get_max(T v1, T v2, T v3) {
  return fmax(v1, fmax(v2, v3));
}

}  // namespace stereomatch
