#pragma once

#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace stereomatch {
struct AggregationOps {
  static void RunSemiglobal(const torch::Tensor &cost_volume,
                            const torch::Tensor &left_image, float penalty1,
                            float penalty2, torch::Tensor &output_cost_volume);

  static void RegisterPybind(pybind11::module &m);
};
}  // namespace stereomatch
