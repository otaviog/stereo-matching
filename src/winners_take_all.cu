#include "disparity_reduce.hpp"

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"

namespace stereomatch {

template <Device dev, typename scalar_t>
struct WTAKernel {};

template <typename scalar_t>
struct WTAKernel<kCPU, scalar_t> {
  const typename Accessor<kCPU, scalar_t, 3>::T cost_volume;
  typename Accessor<kCPU, int32_t, 2>::T disparity_image;

  WTAKernel(const torch::Tensor &cost_volume, torch::Tensor disparity_image)
      : cost_volume(Accessor<kCPU, scalar_t, 3>::Get(cost_volume)),
        disparity_image(Accessor<kCPU, int, 2>::Get(disparity_image)) {}

  __host__ void operator()(int row, int col) noexcept {
    const auto disparity_costs = cost_volume[row][col];
    const auto max_disparity = cost_volume.size(2);
    const auto width = cost_volume.size(1);

    scalar_t min_cost = disparity_costs[0];
    int won_disparity = 0;
    
    for (int disp = 1; disp < max_disparity;
                    //&& (disp + col) < width;
                    ++disp) {
      const auto cost = disparity_costs[disp];
      if (cost < min_cost) {
        min_cost = cost;
        won_disparity = disp;
      }
    }

    disparity_image[row][col] = won_disparity;
  }
};

template <typename scalar_t>
struct WTAKernel<kCUDA, scalar_t> {
  const typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;
  typename Accessor<kCUDA, int32_t, 2>::T disparity_image;

  WTAKernel(const torch::Tensor &cost_volume, torch::Tensor disparity_image)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        disparity_image(Accessor<kCUDA, int, 2>::Get(disparity_image)) {}

  __device__ void operator()(int row, int col, int disp) {
    extern __shared__ __align__(sizeof(float)) uint8_t _shared_memory[];
	const auto max_disparity = cost_volume.size(2);

	scalar_t *disparity_costs = (scalar_t*)_shared_memory;
	int *indices = (int*)&_shared_memory[max_disparity*sizeof(scalar_t)];

    disparity_costs[disp] = cost_volume[row][col][disp];
    indices[disp] = disp;

    __syncthreads();
    const auto width = cost_volume.size(1);
        
    for (auto s = max_disparity >> 1; s >= 1; s = s >> 1) {
      if (disp < s) {
        const auto rhs_idx = s + disp;
        const auto rhs_cost = disparity_costs[rhs_idx];
        if (disparity_costs[disp] >= rhs_cost) {
          disparity_costs[disp] = rhs_cost;
          indices[disp] = indices[rhs_idx];
        }
      }
      __syncthreads();
    }

    if (disp == 0) {
      disparity_image[row][col] = indices[0];
    }
  }
};

template <typename scalar_t>
__global__ void LaunchKernel(WTAKernel<kCUDA, scalar_t> kernel, int width,
                             int height, int max_disparity) {
  kernel(blockIdx.y, blockIdx.x, threadIdx.x);
}

void DisparityReduceOps::RunWinnersTakeAll(const torch::Tensor &cost_volume,
                                          torch::Tensor disparity_image) {
  const auto ref_device = cost_volume.device();

  STM_CHECK_DEVICE(ref_device, disparity_image);
  const auto max_disparity = cost_volume.size(2);
  const auto width = cost_volume.size(1);
  const auto height = cost_volume.size(0);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunWinnersTakeAll", [&] {
          WTAKernel<kCUDA, scalar_t> kernel(cost_volume, disparity_image);
          LaunchKernel<<<dim3(width, height), max_disparity,
			sizeof(scalar_t)*max_disparity + sizeof(int)*max_disparity
					  >>>(
              kernel, width, height, max_disparity);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunWinnersTakeAll", [&] {
          WTAKernel<kCPU, scalar_t> kernel(cost_volume, disparity_image);
          KernelLauncher<kCPU>::Launch2D(kernel, width, height);
        });
  }
}
}  // namespace stereomatch
