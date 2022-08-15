#pragma once

#include <torch/torch.h>

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

namespace stereomatch {
namespace detail {
inline at::ScalarType scalar_type(at::ScalarType s) { return s; }
}  // namespace detail
}  // namespace stereomatch

#define STM_DISPATCH_COSTFUNC_TYPES(TYPE1, TYPE2, NAME, ...)        \
  [&] {                                                             \
    const c10::ScalarType type1 =                                   \
        stereomatch::detail::scalar_type(left_image.scalar_type()); \
    const c10::ScalarType type2 = right_image.scalar_type();        \
    switch (type1) {                                                \
      case at::ScalarType::Byte: {                                  \
        using scalar1_t = uint8_t;                                  \
        switch (type2) {                                            \
          case at::ScalarType::Int: {                               \
            using scalar2_t = int32_t;                              \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          case at::ScalarType::Float: {                             \
            using scalar2_t = float;                                \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          default:                                                  \
            throw std::runtime_error(NAME ": type not supported");  \
        }                                                           \
        break;                                                      \
      }                                                             \
      case at::ScalarType::Short: {                                 \
        using scalar1_t = int16_t;                                  \
        switch (type2) {                                            \
          case at::ScalarType::Int: {                               \
            using scalar2_t = int32_t;                              \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          case at::ScalarType::Float: {                             \
            using scalar2_t = float;                                \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          default:                                                  \
            throw std::runtime_error(NAME ": type not supported");  \
        }                                                           \
        break;                                                      \
      }                                                             \
      case at::ScalarType::Float: {                                 \
        using scalar1_t = float;                                    \
        switch (type2) {                                            \
          case at::ScalarType::Int: {                               \
            using scalar2_t = int32_t;                              \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          case at::ScalarType::Float: {                             \
            using scalar2_t = float;                                \
            __VA_ARGS__();                                          \
            break;                                                  \
          }                                                         \
          default:                                                  \
            throw std::runtime_error(NAME ": type not supported");  \
        }                                                           \
        break;                                                      \
      }                                                             \
      default:                                                      \
        throw std::runtime_error(NAME ": type not supported");      \
    }                                                               \
  }()

#define STM_DISPATCH_COSTFUNC_TYPES_DEVICE(TYPE1, TYPE2, DEVICE, NAME, ...) \
  if (DEVICE.is_cuda()) {                                                   \
    const Device device = kCUDA;                                            \
    STM_DISPATCH_COSTFUNC_TYPES(TYPE1, TYPE2, NAME, __VA_ARGS__);           \
  } else {                                                                  \
    const Device device = kCPU;                                             \
    STM_DISPATCH_COSTFUNC_TYPES(TYPE1, TYPE2, NAME, __VA_ARGS__);           \
  }
