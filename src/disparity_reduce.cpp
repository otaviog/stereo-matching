#include "disparity_reduce.hpp"

namespace stereomatch {
void DisparityReduceOps::RegisterPybind(pybind11::module &m) {
  pybind11::class_<DisparityReduceOps>(m, "DisparityReduceOps")
      .def_static("run_winners_take_all",
                  &DisparityReduceOps::RunWinnersTakeAll)
      .def_static("run_dynamic_programming",
                  &DisparityReduceOps::RunDynamicProgramming);
}
}  // namespace stereomatch
