#pragma once

#include <torch/torch.h>
#include <torch/csrc/utils/pybind.h>

namespace stereomatch {
struct CostOps {
  static void ComputeSSD(const torch::Tensor &left_image,
                         const torch::Tensor &right_image,
                         torch::Tensor cost_volume,
                         int kernel_size);

  static void RegisterPybind(pybind11::module &m);
};

}  // namespace stereomatch
