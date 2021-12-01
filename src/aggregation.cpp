#include "aggregation.hpp"

namespace stereomatch {
void AggregationOps::RegisterPybind(pybind11::module &m) {
  pybind11::class_<AggregationOps>(m, "AggregationOps")
      .def_static("run_winners_take_all", &AggregationOps::RunWinnersTakeAll)
      .def_static("run_dynamic_programming", &AggregationOps::RunDynamicProgramming);
}
}  // namespace stereomatch
