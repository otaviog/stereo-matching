#include <torch/csrc/utils/pybind.h>

#include "aggregation.hpp"
#include "cost.hpp"
#include "cuda_texture.hpp"
#include "disparity_reduce.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cstereomatch, m) {
  using namespace stereomatch;

  CostOps::RegisterPybind(m);
  DisparityReduceOps::RegisterPybind(m);
  AggregationOps::RegisterPybind(m);
  CUDATexture::RegisterPybind(m);
}
