#include "cost.hpp"

#include <limits>

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"

using namespace std;

namespace stereomatch {

template <Device dev, typename scalar_t>
struct SSDKernel {
  const typename Accessor<dev, scalar_t, 2>::T left_image;
  const typename Accessor<dev, scalar_t, 2>::T right_image;
  typename Accessor<dev, scalar_t, 3>::T cost_volume;
  const int kernel_size;
  const scalar_t empty_value;

  SSDKernel(const torch::Tensor &left_image, const torch::Tensor &right_image,
            torch::Tensor &cost_volume, int kernel_size, scalar_t empty_value)
      : left_image(Accessor<dev, scalar_t, 2>::Get(left_image)),
        right_image(Accessor<dev, scalar_t, 2>::Get(right_image)),
        cost_volume(Accessor<dev, scalar_t, 3>::Get(cost_volume)),
        kernel_size(kernel_size),
        empty_value(empty_value) {}

  __device__ __host__ void operator()(int row, int col) {
    for (int disp = 0; disp < cost_volume.size(0); ++disp) {
      if (col - disp < 0) {
        // cost_volume[disp][row][col] = empty_value;
        return;
      }

      const int row_start = max(row - kernel_size, 0);
      const int row_end = min(row + kernel_size, int(cost_volume.size(1)));

      // const int col_start = max(col - kernel_size, 0);
      const int col_start =
          abs(min(col - disp - kernel_size, 0)) + col - kernel_size;
      const int col_end = min(col + kernel_size, int(cost_volume.size(2)));

      scalar_t cost_value = 0;

      for (int krow = row_start; krow < row_end; ++krow) {
        for (int kcol = col_start; kcol < col_end; ++kcol) {
          const scalar_t left_intensity = left_image[krow][kcol];
          const scalar_t right_intensity =
              right_image[krow][kcol - disp];  // TODO: review
          const scalar_t diff = left_intensity - right_intensity;

          cost_value += diff * diff;
        }
      }
      cost_volume[disp][row][col] = cost_value;
    }
  }
};

void CostOps::ComputeSSD(const torch::Tensor &left_image,
                         const torch::Tensor &right_image,
                         torch::Tensor cost_volume, int kernel_size) {
  const auto ref_device = left_image.device();

  STM_CHECK_DEVICE(ref_device, right_image);
  STM_CHECK_DEVICE(ref_device, cost_volume);

  STM_DISPATCH_KERNEL_FLOATING_TYPES(
      left_image.scalar_type(), ref_device, "ComputeSSD", ([&] {
        SSDKernel<device, scalar_t> kernel(
            left_image, right_image, cost_volume, kernel_size, -1.0
            // numeric_limits<scalar_t>::infinity()
        );

        KernelLauncher<device>::Launch2D(kernel, left_image.size(1),
                                         left_image.size(0));
      }));
}

template <typename scalar_t>
struct SSDTextureKernel {
  const cudaTextureObject_t left_image, right_image;
  torch::PackedTensorAccessor32<scalar_t, 3> cost_volume;
  const int kernel_size;
  const scalar_t empty_value;

  SSDTextureKernel(const CUDATexture &left_image,
                   const CUDATexture &right_image, torch::Tensor &cost_volume,
                   int kernel_size, scalar_t empty_value)
      : left_image(left_image),
        right_image(right_image),
        cost_volume(cost_volume.packed_accessor32<scalar_t, 3>()),
        kernel_size(kernel_size),
        empty_value(empty_value) {}

  __device__ void operator()(int row, int col) {
    for (int disp = 0; disp < cost_volume.size(0); ++disp) {
      if (col - disp < 0) {
        // cost_volume[disp][row][col] = empty_value;
        return;
      }

      const int row_start = max(row - kernel_size, 0);
      const int row_end = min(row + kernel_size, int(cost_volume.size(1)));

      // const int col_start = max(col - kernel_size, 0);
      const int col_start =
          abs(min(col - disp - kernel_size, 0)) + col - kernel_size;
      const int col_end = min(col + kernel_size, int(cost_volume.size(2)));

      scalar_t cost_value = 0;

      for (int krow = row_start; krow < row_end; ++krow) {
        for (int kcol = col_start; kcol < col_end; ++kcol) {
          const scalar_t left_intensity = tex2D<float>(left_image, kcol, krow);
          const scalar_t right_intensity =
              tex2D<float>(right_image, kcol, krow);
          const scalar_t diff = left_intensity - right_intensity;

          cost_value += diff * diff;
        }
      }
      cost_volume[disp][row][col] = cost_value;
    }
  }
};

void CostOps::ComputeSSD(const CUDATexture &left_image,
                         const CUDATexture &right_image,
                         torch::Tensor cost_volume, int kernel_size) {
  if (!cost_volume.device().is_cuda()) {
    throw std::runtime_error("Cost volume must be a CUDA tensor.");
  }

  AT_DISPATCH_FLOATING_TYPES(
      cost_volume.scalar_type(), "ComputeSSD with Texture", ([&] {
        SSDTextureKernel<scalar_t> ssd_kernel(left_image, right_image,
                                              cost_volume, kernel_size, -1.0);
        KernelLauncher<kCUDA>::Launch2D(ssd_kernel, left_image.get_height(),
                                        left_image.get_width());
      }));
}
}  // namespace stereomatch
