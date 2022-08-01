#include "aggregation.hpp"

namespace stereomatch {
void AggregationModule::RegisterPybind(pybind11::module &m) {
  pybind11::class_<AggregationModule>(m, "AggregationOps")
      .def_static("run_winners_take_all", &AggregationModule::RunWinnersTakeAll)
      .def_static("run_dynamic_programming", &AggregationModule::RunDynamicProgramming)
      .def_static("run_semiglobal", &AggregationModule::RunSemiglobal);
  
}
}  // namespace stereomatch
