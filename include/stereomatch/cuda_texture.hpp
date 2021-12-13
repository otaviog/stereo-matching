#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

#include "cuda_utils.hpp"

namespace stereomatch {

struct CUDATextureAccessor {
  CUDATextureAccessor(cudaTextureObject_t texture, int width, int height,
                      int depth)
      : texture(texture), width(width), height(height), depth(depth) {}

  cudaTextureObject_t texture;
  int width, height, depth;
};

namespace detail {
template <typename scalar_t>
struct ScalarCheck {
  static void Check(torch::ScalarType type) {
    throw std::runtime_error("Type not supported");
  }
};

template <>
struct ScalarCheck<float> {
  static void Check(torch::ScalarType type) {
    if (type != torch::ScalarType::Float) {
      throw std::runtime_error("Expectating a float scalar");
    }
  }
};

template <>
struct ScalarCheck<uint8_t> {
  static void Check(torch::ScalarType type) {
    if (type != torch::ScalarType::Byte) {
      throw std::runtime_error("Error");
    }
  }
};
}  // namespace detail

class CUDATexture {
 public:
  CUDATexture() {
    texture_object_ = 0;
    array_ = nullptr;
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    scalar_type_ = torch::ScalarType::Float;
  }

  CUDATexture(CUDATexture &&other)
      : texture_object_(other.texture_object_),
        array_(other.array_),
        width_(other.width_),
        height_(other.height_),
        channels_(other.channels_),
        scalar_type_(other.scalar_type_) {
    other.texture_object_ = 0;
    other.array_ = nullptr;
  }

  ~CUDATexture() {}

  // CUDATexture(const CUDATexture &) = delete;
  // const CUDATexture &operator=(const CUDATexture &) = delete;

  static void RegisterPybind(pybind11::module &m);

  static void RunTestKernel(const CUDATexture &input_texture,
                            torch::Tensor output_tensor);

  static void RunTestKernel2(const torch::Tensor &input_tensor,
                             torch::Tensor output_tensor);

  void CopyFromTensor(const torch::Tensor &data, bool normalize_coords);

  torch::Tensor ToTensor() const;

  void Release();

  bool is_empty() const { return texture_object_ == 0; }

  operator cudaTextureObject_t() const { return texture_object_; }

  int get_width() const { return width_; }

  int get_height() const { return height_; }

  int get_channels() const { return channels_; }

  torch::ScalarType get_scalar_type() const { return scalar_type_; }

  template <typename scalar_t>
  CUDATextureAccessor accessor() const {
    detail::ScalarCheck<scalar_t>::Check(scalar_type_);
    return CUDATextureAccessor(texture_object_, width_, height_, channels_);
  }

 private:
  cudaTextureObject_t texture_object_;
  cudaArray *array_;
  int width_, height_, channels_;
  torch::ScalarType scalar_type_;
};
}  // namespace stereomatch
