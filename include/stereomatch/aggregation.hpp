#pragma once

#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace stereomatch {
struct AggregationModule {
  static void RunWinnersTakeAll(const torch::Tensor &cost_volume,
                                torch::Tensor disparity_image);
  static void RunDynamicProgramming(const torch::Tensor &cost_volume,
                                    torch::Tensor path_volume,
                                    torch::Tensor accumulated_costs,
                                    torch::Tensor disparity_image);
  static void RunSemiglobal(const torch::Tensor &cost_volume,
                            const torch::Tensor &left_image, float penalty1,
                            float penalty2, torch::Tensor &output_cost_volume);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace stereomatch
