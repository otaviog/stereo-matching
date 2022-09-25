#pragma once

#include <inttypes.h>
#include <vector>

#include "cuda_utils.hpp"

namespace stereomatch {

template <typename T>
struct Point2 {
  T x, y;

  STM_DEVICE_HOST Point2(T x = 0, T y = 0) : x{x}, y{y} {}

  Point2<T> &operator+=(const Point2<T> &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  friend Point2<T> operator+(Point2<T> lhs, const Point2<T> &rhs) {
    lhs += rhs;
    return lhs;
  }
};

template <typename S, typename T>
STM_DEVICE_HOST S cast_point2(const Point2<T> &pt) {
  return S{pt.x, pt.y};
}

struct SGPixelPath {
  Point2<int16_t> start, end;
  Point2<int16_t> direction;
  uint16_t size;

  STM_DEVICE_HOST SGPixelPath(Point2<int16_t> start, Point2<int16_t> end,
              Point2<int16_t> direction, int16_t size) noexcept
      : start(start), end(end), direction(direction), size(size) {}

  STM_DEVICE_HOST SGPixelPath inverse() const {
    return SGPixelPath(end, start, Point2<int16_t>(-direction.x, -direction.y),
                       size);
  }
  static std::vector<SGPixelPath> GeneratePaths(size_t width, size_t height) noexcept;
};

}  // namespace stereomatch
