#pragma once

#include <torch/torch.h>
#include <torch/csrc/utils/pybind.h>

namespace stereomatch {
struct AggregationOps {
  static void RunWinnersTakeAll(const torch::Tensor &cost_volume,
                                torch::Tensor disparity_image);
  static void RunDynamicProgramming(const torch::Tensor &cost_volume,
                                    torch::Tensor path_volume,
                                    torch::Tensor accumulated_costs,
                                    torch::Tensor disparity_image);
  static void RegisterPybind(pybind11::module &m);
};
}  // namespace stereomatch
