#pragma once

#include <memory>

#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

#include "cuda_utils.hpp"

namespace stereomatch {

class CUDATexture {
 public:
  CUDATexture(cudaTextureObject_t texture_object = 0,
              cudaArray *array = nullptr, int width = 0, int height = 0,
              int channels = 0)
      : texture_object_(texture_object),
        array_(array),
        width_(width),
        height_(height),
        channels_(channels) {}

  CUDATexture(CUDATexture &&other)
      : texture_object_(other.texture_object_),
        array_(other.array_),
        width_(other.width_),
        height_(other.height_),
        channels_(other.channels_) {
    other.texture_object_ = 0;
  }

  ~CUDATexture() { Release(); }

  CUDATexture(const CUDATexture &) = delete;
  const CUDATexture &operator=(const CUDATexture &) = delete;

  static void RegisterPybind(pybind11::module &m);

  static CUDATexture FromTensor(const torch::Tensor &data);

  torch::Tensor ToTensor() const;

  void Release() {
    if (texture_object_ != 0) {
      CudaSafeCall(cudaDestroyTextureObject(texture_object_));
      CudaSafeCall(cudaFreeArray(array_));
      texture_object_ = 0;
    }
  }

  operator cudaTextureObject_t() const { return texture_object_; }

  int get_width() const { return width_; }

  int get_height() const { return height_; }

  int get_channels() const { return channels_; }

 private:
  cudaTextureObject_t texture_object_;
  cudaArray *array_;
  int width_, height_, channels_;
};
}  // namespace stereomatch
