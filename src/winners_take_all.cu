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
    float leastDiff = 9999999;
    int wonDisparity = 0;

    for (int d = 0; d < cost_volume.size(0) && (d + col) < cost_volume.size(2);
         d++) {
      const float diff = cost_volume[d][row][col];

      if (diff < leastDiff) {
        leastDiff = diff;
        wonDisparity = d;
      }
    }

    disparity_image[row][col] = wonDisparity;
  }
};

void AggregationModule::RunWinnersTakeAll(const torch::Tensor &cost_volume,
                                       torch::Tensor disparity_image) {
  const auto ref_device = cost_volume.device();

  STM_CHECK_DEVICE(ref_device, disparity_image);

  STM_DISPATCH_KERNEL_FLOATING_TYPES(
      cost_volume.scalar_type(), ref_device, "RunWinnersTakeAll", ([&] {
        WTAKernel<device, scalar_t> kernel(cost_volume, disparity_image);

        KernelLauncher<device>::Launch2D(kernel, cost_volume.size(2),
                                         cost_volume.size(1));
      }));
}
}  // namespace stereomatch
