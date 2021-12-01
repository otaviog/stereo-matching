#include "cost.hpp"

namespace stereomatch {
void CostOps::RegisterPybind(pybind11::module &m) {
  pybind11::class_<CostOps>(m, "CostOps")
      .def_static("compute_ssd", &CostOps::ComputeSSD);
}
}  // namespace stereomatch
