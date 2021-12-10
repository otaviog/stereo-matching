#include "cost.hpp"

namespace stereomatch {
void CostOps::RegisterPybind(pybind11::module &m) {
  pybind11::class_<CostOps>(m, "CostOps")
      .def_static(
          "compute_ssd",
          pybind11::overload_cast<const torch::Tensor &, const torch::Tensor &,
                                  torch::Tensor, int>(&CostOps::ComputeSSD))
      .def_static(
          "compute_ssd",
          pybind11::overload_cast<const CUDATexture &, const CUDATexture &,
                                  torch::Tensor, int>(&CostOps::ComputeSSD));
}
}  // namespace stereomatch
