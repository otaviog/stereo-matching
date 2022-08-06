#include "aggregation.hpp"

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"
#include "math.hpp"

#define MAX_DISP 1048

namespace stereomatch {

template <typename scalar_t>
struct ComputePathCPUKernel {
  const torch::TensorAccessor<scalar_t, 3> cost_volume;
  torch::TensorAccessor<int8_t, 3> path_volume;
  torch::TensorAccessor<scalar_t, 2> disp_costsum_per_row;

  ComputePathCPUKernel(const torch::Tensor &cost_volume,
                       torch::Tensor path_volume,
                       torch::Tensor disp_costsum_per_row)
      : cost_volume(cost_volume.accessor<scalar_t, 3>()),
        path_volume(path_volume.accessor<int8_t, 3>()),
        disp_costsum_per_row(disp_costsum_per_row.accessor<scalar_t, 2>()) {}

  void operator()(int row) {
    std::unique_ptr<scalar_t[]> previous_col_cost(
        new scalar_t[cost_volume.size(0)]);
    for (int disp = 0; disp < cost_volume.size(0); disp++) {
      path_volume[disp][row][0] = 0;
      previous_col_cost[disp] = cost_volume[disp][row][0];
    }

    for (int col = 1; col < cost_volume.size(2); col++) {
      for (int disp = 0; disp < cost_volume.size(0); disp++) {
        const scalar_t current_cost = cost_volume[disp][row][col];
        const scalar_t cost1 = (disp > 0)
                                   ? previous_col_cost[disp - 1]
                                   : std::numeric_limits<scalar_t>::infinity();
        const scalar_t cost2 = previous_col_cost[disp];
        const scalar_t cost3 = (disp < cost_volume.size(0) - 1)
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

        path_volume[disp][row][col] = path_direction;
        previous_col_cost[disp] = current_cost + min_cost;
        disp_costsum_per_row[disp][row] = previous_col_cost[disp];
      }
    }
  }
};

template <typename scalar_t>
struct ComputePathCUDAKernel {
  const typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;
  typename Accessor<kCUDA, int8_t, 3>::T path_volume;
  typename Accessor<kCUDA, scalar_t, 2>::T disp_costsum_per_row;

  ComputePathCUDAKernel(const torch::Tensor &cost_volume,
                        torch::Tensor path_volume,
                        torch::Tensor disp_costsum_per_row)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        path_volume(Accessor<kCUDA, int8_t, 3>::Get(path_volume)),
        disp_costsum_per_row(
            Accessor<kCUDA, scalar_t, 2>::Get(disp_costsum_per_row)) {}
  __device__ void operator()(int row, int disp) {
    __shared__ float shr_previous_col_cost[MAX_DISP + 2];
    if (disp == 0) {
      // pad the border with infinity for avoiding checking disparity
      // indices.
      shr_previous_col_cost[0] = NumericLimits<scalar_t>::infinity();
      shr_previous_col_cost[cost_volume.size(0) + 1] =
          NumericLimits<scalar_t>::infinity();
    }

    shr_previous_col_cost[disp + 1] = cost_volume[disp][row][0];
    __syncthreads();

    path_volume[disp][row][0] = 0;

    for (ushort col = 1; col < cost_volume.size(2); col++) {
      const scalar_t current_cost = cost_volume[disp][row][col];

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

      path_volume[disp][row][col] = path_direction;

      __syncthreads();
      shr_previous_col_cost[disp + 1] = current_cost + min_cost;
    }

    disp_costsum_per_row[disp][row] = shr_previous_col_cost[disp + 1];
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

template <Device dev, typename scalar_t>
struct ReducePathKernel {
  const typename Accessor<dev, int8_t, 3>::T path_volume;
  const typename Accessor<dev, scalar_t, 2>::T disp_costsum_per_row;
  typename Accessor<dev, int32_t, 2>::T disparity_image;

  ReducePathKernel(const torch::Tensor &path_volume,
                   const torch::Tensor &disp_costsum_per_row,
                   torch::Tensor &disparity_image)
      : path_volume(Accessor<dev, int8_t, 3>::Get(path_volume)),
        disp_costsum_per_row(
            Accessor<dev, scalar_t, 2>::Get(disp_costsum_per_row)),
        disparity_image(Accessor<dev, int32_t, 2>::Get(disparity_image)) {}

  __device__ __host__ void operator()(int row) {
    int min_disparity = 0;
    float min_cost = disp_costsum_per_row[0][row];

    const auto max_disp = path_volume.size(0);
    for (int disp = 1; disp < max_disp; disp++) {
      const float current_cost = disp_costsum_per_row[disp][row];
      if (current_cost < min_cost) {
        min_disparity = disp;
        min_cost = current_cost;
      }
    }

    disparity_image[row][disparity_image.size(1) - 1] = min_disparity;

    for (int col = path_volume.size(2) - 1; col >= 0; col--) {
      const int8_t path_direction = path_volume[min_disparity][row][col];
      const int new_disparity = min_disparity + path_direction;

      if (new_disparity >= 0 && new_disparity < path_volume.size(0)) {
        min_disparity = new_disparity;
      }

      disparity_image[row][col] = min_disparity;
    }
  }
};

void AggregationModule::RunDynamicProgramming(const torch::Tensor &cost_volume,
                                           torch::Tensor path_volume,
                                           torch::Tensor disp_costsum_per_row,
                                           torch::Tensor disparity_image) {
  const auto ref_device = cost_volume.device();

  STM_CHECK_DEVICE(ref_device, path_volume);
  STM_CHECK_DEVICE(ref_device, disp_costsum_per_row);
  STM_CHECK_DEVICE(ref_device, disparity_image);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunDynamicprogramming", ([&] {
          ComputePathCUDAKernel<scalar_t> path_kernel(cost_volume, path_volume,
                                                      disp_costsum_per_row);
          ExecuteDynamicProgrammingKernel<<<cost_volume.size(1),
                                            cost_volume.size(0)>>>(
              path_kernel, cost_volume.size(1), cost_volume.size(0));

          ReducePathKernel<kCUDA, scalar_t> reduce_kernel(
              path_volume, disp_costsum_per_row, disparity_image);
          KernelLauncher<kCUDA>::Launch1D(reduce_kernel, cost_volume.size(1));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunDynamicProgramming", ([&] {
          ComputePathCPUKernel<scalar_t> path_kernel(cost_volume, path_volume,
                                                     disp_costsum_per_row);
          KernelLauncher<kCPU>::Launch1D(path_kernel, cost_volume.size(1));

          ReducePathKernel<kCPU, scalar_t> reduce_kernel(
              path_volume, disp_costsum_per_row, disparity_image);
          KernelLauncher<kCPU>::Launch1D(reduce_kernel, cost_volume.size(1));
        }));
  }
}

}  // namespace stereomatch
