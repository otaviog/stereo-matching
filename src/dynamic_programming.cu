#include "aggregation.hpp"

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"
#include "math.hpp"

#define MAX_DISP 1048

namespace stereomatch {

template <typename scalar_t>
struct ExpandPathsKernelCPU {
  const torch::TensorAccessor<scalar_t, 3> cost_volume;
  torch::TensorAccessor<int8_t, 3> path_volume;
  torch::TensorAccessor<scalar_t, 2> row_final_costs;

  ExpandPathsKernelCPU(const torch::Tensor &cost_volume,
                       torch::Tensor path_volume, torch::Tensor row_final_costs)
      : cost_volume(cost_volume.accessor<scalar_t, 3>()),
        path_volume(path_volume.accessor<int8_t, 3>()),
        row_final_costs(row_final_costs.accessor<scalar_t, 2>()) {}

  void operator()(int row) {
    const auto max_disparity = cost_volume.size(2);

    std::unique_ptr<scalar_t[]> previous_col_cost(new scalar_t[max_disparity]);

    for (int disp = 0; disp < max_disparity; disp++) {
      path_volume[disp][row][0] = 0;
      previous_col_cost[disp] = cost_volume[disp][row][0];
    }

    const auto width = cost_volume.size(1);
    for (int col = 1; col < width; col++) {
      const auto cost_channel_acc = cost_volume[row][col];
      auto path_channel_acc = path_volume[row][col];
      for (int disp = 0; disp < max_disparity; disp++) {
        const scalar_t current_cost = cost_channel_acc[disp];
        const scalar_t cost1 = (disp > 0)
                                   ? previous_col_cost[disp - 1]
                                   : std::numeric_limits<scalar_t>::infinity();
        const scalar_t cost2 = previous_col_cost[disp];
        const scalar_t cost3 = (disp < max_disparity - 1)
                                   ? previous_col_cost[disp + 1]
                                   : std::numeric_limits<scalar_t>::infinity();

        scalar_t min_cost;
        int8_t path_direction;
        if (cost1 < cost2 && cost1 < cost3) {
          min_cost = cost1;
          path_direction = -1;
        } else if (cost2 < cost3) {
          min_cost = cost2;
          path_direction = 0;
        } else {
          min_cost = cost3;
          path_direction = 1;
        }

        path_channel_acc[disp] = path_direction;
        previous_col_cost[disp] = current_cost + min_cost;
      }
    }

    for (int disp = 0; disp < max_disparity; ++disp)
      row_final_costs[row][disp] = previous_col_cost[disp];
  }
};

template <typename scalar_t>
struct ExpandPathsKernelCUDA {
  const typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;
  typename Accessor<kCUDA, int8_t, 3>::T path_volume;
  typename Accessor<kCUDA, scalar_t, 2>::T row_final_costs;

  ExpandPathsKernelCUDA(const torch::Tensor &cost_volume,
                        torch::Tensor path_volume,
                        torch::Tensor row_final_costs)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        path_volume(Accessor<kCUDA, int8_t, 3>::Get(path_volume)),
        row_final_costs(Accessor<kCUDA, scalar_t, 2>::Get(row_final_costs)) {}
  __device__ void operator()(int row, int disp) {
    __shared__ float shr_previous_col_cost[MAX_DISP + 2];
    const auto max_disparity = cost_volume.size(2);

    if (disp == 0) {
      // pad the border with infinity for avoiding checking disparity
      // indices.
      shr_previous_col_cost[0] = NumericLimits<scalar_t>::infinity();
      shr_previous_col_cost[max_disparity + 1] =
          NumericLimits<scalar_t>::infinity();
    }

    shr_previous_col_cost[disp + 1] = cost_volume[row][0][disp];
    __syncthreads();

    path_volume[disp][row][0] = 0;

    const auto width = cost_volume.size(1);
    for (ushort col = 1; col < width; col++) {
      const scalar_t current_cost = cost_volume[row][col][disp];
      __syncthreads();
      const scalar_t cost1 = shr_previous_col_cost[disp];
      const scalar_t cost2 = shr_previous_col_cost[disp + 1];
      const scalar_t cost3 = shr_previous_col_cost[disp + 2];

      scalar_t min_cost;
      int8_t path_direction;
      if (cost1 < cost2 && cost1 < cost3) {
        min_cost = cost1;
        path_direction = -1;
      } else if (cost2 < cost3) {
        min_cost = cost2;
        path_direction = 0;
      } else {
        min_cost = cost3;
        path_direction = 1;
      }

      path_volume[row][col][disp] = path_direction;

      __syncthreads();
      shr_previous_col_cost[disp + 1] = current_cost + min_cost;
    }

    row_final_costs[disp][row] = shr_previous_col_cost[disp + 1];
  }
};

template <typename Kernel>
static __global__ void ExecuteDynamicProgrammingKernel(Kernel kern, int height,
                                                       int max_disp) {
  const int row = blockIdx.x;
  const int disp = threadIdx.x;
  if (row < height && disp < max_disp) {
    kern(row, disp);
  }
}

static constexpr int32_t clamp(int32_t val, int32_t min, int32_t max) {
  if (val < min) {
    return min;
  } else if (val > max) {
    return max;
  } else {
    return val;
  }
}

template <Device dev, typename scalar_t>
struct ReducePathsKernel {
  const typename Accessor<dev, int8_t, 3>::T path_volume;
  const typename Accessor<dev, int64_t, 1>::T end_disparities;
  typename Accessor<dev, int32_t, 2>::T disparity_image;

  ReducePathsKernel(const torch::Tensor &path_volume,
                    const torch::Tensor &end_disparities,
                    torch::Tensor &disparity_image)
      : path_volume(Accessor<dev, int8_t, 3>::Get(path_volume)),
        end_disparities(Accessor<dev, int64_t, 1>::Get(end_disparities)),
        disparity_image(Accessor<dev, int32_t, 2>::Get(disparity_image)) {}

  __device__ __host__ void operator()(int row) {
    const auto width = path_volume.size(1);
    const auto max_disparity = path_volume.size(2);
    auto current_disp = end_disparities[row];

    auto disp_image_row_acc = disparity_image[row];
    disp_image_row_acc[width - 1] = current_disp;

    auto path_volume_row_acc = path_volume[row];

    for (auto col = width - 2; col >= 0; col--) {
      const auto path_direction = path_volume_row_acc[col][current_disp];
      current_disp = clamp(current_disp + path_direction, 0, max_disparity - 1);
      disparity_image[row][col] = current_disp;
    }
  }
};

void AggregationModule::RunDynamicProgramming(const torch::Tensor &cost_volume,
                                              torch::Tensor path_volume,
                                              torch::Tensor row_final_costs,
                                              torch::Tensor disparity_image) {
  const auto ref_device = cost_volume.device();

  STM_CHECK_DEVICE(ref_device, path_volume);
  STM_CHECK_DEVICE(ref_device, row_final_costs);
  STM_CHECK_DEVICE(ref_device, disparity_image);

  const auto height = cost_volume.size(0);
  const auto width = cost_volume.size(1);
  const auto max_disparity = cost_volume.size(2);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunDynamicprogrammingGPU", ([&] {
          ExpandPathsKernelCUDA<scalar_t> path_kernel(cost_volume, path_volume,
                                                      row_final_costs);
          ExecuteDynamicProgrammingKernel<<<height, max_disparity>>>(
              path_kernel, height, max_disparity);

          auto end_disparities = torch::argmin(row_final_costs, {1});
          ReducePathsKernel<kCUDA, scalar_t> reduce_kernel(
              path_volume, end_disparities, disparity_image);
          KernelLauncher<kCUDA>::Launch1D(reduce_kernel, height);
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "DynamicProgrammingCPU", ([&] {
          ExpandPathsKernelCPU<scalar_t> path_kernel(cost_volume, path_volume,
                                                     row_final_costs);
          KernelLauncher<kCPU>::Launch1D(path_kernel, height);
          
          auto end_disparities = torch::argmin(row_final_costs, {1});
          ReducePathsKernel<kCPU, scalar_t> reduce_kernel(
              path_volume, end_disparities, disparity_image);
          KernelLauncher<kCPU>::Launch1D(reduce_kernel, height);
        }));
  }
}

}  // namespace stereomatch
