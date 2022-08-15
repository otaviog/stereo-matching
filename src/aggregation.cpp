#include "aggregation.hpp"

namespace stereomatch {
void AggregationOps::RegisterPybind(pybind11::module &m) {
  pybind11::class_<AggregationOps>(m, "AggregationOps")
      .def_static("run_semiglobal", &AggregationOps::RunSemiglobal);
  
}
}  // namespace stereomatch
