#pragma once

#include <torch/torch.h>
#include <torch/csrc/utils/pybind.h>

#include "cuda_texture.hpp"

namespace stereomatch {
struct CostOps {
  static void ComputeSSD(const torch::Tensor &left_image,
                         const torch::Tensor &right_image,
                         torch::Tensor cost_volume,
                         int kernel_size);
  
  static void ComputeSSD(const CUDATexture &left_image,
                         const CUDATexture &right_image,
                         torch::Tensor cost_volume,
                         int kernel_size);

  static void ComputeBirchfield(const torch::Tensor &left_image,
                                const torch::Tensor &right_image,
                                torch::Tensor cost_volume,
                                int kernel_size);
      
  static void RegisterPybind(pybind11::module &m);
};

}  // namespace stereomatch
