#include <pybind11/eigen.h>
#include <torch/csrc/utils/pybind.h>

#include "aggregation.hpp"
#include "cost.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cstereomatch, m) {
  using namespace stereomatch;

  CostOps::RegisterPybind(m);
  AggregationOps::RegisterPybind(m);
}
