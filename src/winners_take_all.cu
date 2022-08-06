#include "aggregation.hpp"

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"

namespace stereomatch {

template <Device dev, typename scalar_t>
struct WTAKernel {
  const typename Accessor<dev, scalar_t, 3>::T cost_volume;
  typename Accessor<dev, int32_t, 2>::T disparity_image;

  WTAKernel(const torch::Tensor &cost_volume, torch::Tensor disparity_image)
      : cost_volume(Accessor<dev, scalar_t, 3>::Get(cost_volume)),
        disparity_image(Accessor<dev, int, 2>::Get(disparity_image)) {}

  __device__ __host__ void operator()(int row, int col) {
    scalar_t min_cost = 9999999;
    int won_disparity = 0;

	const auto disparity_costs = cost_volume[row][col];
	const auto max_disparaty = cost_volume.size(2);
	const auto width = cost_volume.size(1);
	for (int disp = 0; disp < max_disparaty && (disp + col) < width; ++disp) {
	  const auto cost = disparity_costs[disp];
      if (cost < min_cost) {
        min_cost = cost;
        won_disparity = disp;
      }
    }

    disparity_image[row][col] = won_disparity;
  }
};

void AggregationModule::RunWinnersTakeAll(const torch::Tensor &cost_volume,
                                       torch::Tensor disparity_image) {
  const auto ref_device = cost_volume.device();

  STM_CHECK_DEVICE(ref_device, disparity_image);
  const auto width = cost_volume.size(1);
  const auto height = cost_volume.size(0);
  
	
  STM_DISPATCH_KERNEL_FLOATING_TYPES(
      cost_volume.scalar_type(), ref_device, "RunWinnersTakeAll", ([&] {
        WTAKernel<device, scalar_t> kernel(cost_volume, disparity_image);
		
        KernelLauncher<device>::Launch2D(kernel, width, height);
      }));
}
}  // namespace stereomatch
