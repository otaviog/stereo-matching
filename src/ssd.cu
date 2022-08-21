#include "cost.hpp"

#include <limits>

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"
#include "numeric.hpp"
#include "type_dispatch.hpp"

using namespace std;

namespace stereomatch {
template <Device dev, typename in_scalar_t, typename out_scalar_t>
struct SSDKernel {
  const typename Accessor<dev, in_scalar_t, 2>::T left_image;
  const typename Accessor<dev, in_scalar_t, 2>::T right_image;
  typename Accessor<dev, out_scalar_t, 3>::T cost_volume;
  const int kernel_size;
  const out_scalar_t empty_value;

  SSDKernel(const torch::Tensor &left_image, const torch::Tensor &right_image,
            torch::Tensor &cost_volume, int kernel_size,
            out_scalar_t empty_value)
      : left_image(Accessor<dev, in_scalar_t, 2>::Get(left_image)),
        right_image(Accessor<dev, in_scalar_t, 2>::Get(right_image)),
        cost_volume(Accessor<dev, out_scalar_t, 3>::Get(cost_volume)),
        kernel_size(kernel_size),
        empty_value(empty_value) {}

  __device__ __host__ void operator()(int row, int col) {
    const int height = cost_volume.size(0);
    const int width = cost_volume.size(1);
    const int max_disparity = cost_volume.size(2);

    for (auto disp = 0; disp < min(col + 1, max_disparity); ++disp) {
      const int row_start = max(row - kernel_size, 0);
      const int row_end = min(row + kernel_size, int(height));

      const int col_start =
          abs(min(col - disp - kernel_size, 0)) + col - kernel_size;
      const int col_end = min(col + kernel_size, int(width));

      out_scalar_t cost_value = 0;

      for (int krow = row_start; krow < row_end; ++krow) {
        for (int kcol = col_start; kcol < col_end; ++kcol) {
          const in_scalar_t left_intensity = left_image[krow][kcol];
          const in_scalar_t right_intensity = right_image[krow][kcol - disp];
          const out_scalar_t diff = left_intensity - right_intensity;

          cost_value += diff * diff;
        }
      }
      cost_volume[row][col][disp] = cost_value;
    }

    for (auto disp = col + 1; disp < max_disparity; ++disp) {
      cost_volume[row][col][disp] = NumericLimits<out_scalar_t>::infinity();
    }
  }
};

void CostOps::ComputeSSD(const torch::Tensor &left_image,
                         const torch::Tensor &right_image,
                         torch::Tensor cost_volume, int kernel_size) {
  const auto ref_device = left_image.device();

  STM_CHECK_DEVICE(ref_device, right_image);
  STM_CHECK_DEVICE(ref_device, cost_volume);
  STM_DISPATCH_COSTFUNC_TYPES_DEVICE(
      left_image.scalar_type(), cost_volume.scalar_type(), left_image.device(),
      "SSDKernel", [&] {
        SSDKernel<device, scalar1_t, scalar2_t> kernel(
            left_image, right_image, cost_volume, kernel_size,
            std::numeric_limits<scalar2_t>::infinity());

        KernelLauncher<device>::Launch2D(kernel, left_image.size(1),
                                         left_image.size(0));
      });
}

template <typename scalar_t>
struct SSDTextureKernel {
  const CUDATextureAccessor left_image, right_image;
  torch::PackedTensorAccessor32<scalar_t, 3> cost_volume;
  const int kernel_size;
  const scalar_t empty_value;

  SSDTextureKernel(const CUDATexture &left_image,
                   const CUDATexture &right_image, torch::Tensor &cost_volume,
                   int kernel_size, scalar_t empty_value)
      : left_image(left_image.accessor<float>()),
        right_image(right_image.accessor<float>()),
        cost_volume(cost_volume.packed_accessor32<scalar_t, 3>()),
        kernel_size(kernel_size),
        empty_value(empty_value) {}

  __device__ void operator()(int row, int col) {
    const int height = cost_volume.size(0);
    const int width = cost_volume.size(1);

    const float heightf = float(height);
    const float widthf = float(width);

    const auto max_disparity = cost_volume.size(2);

    for (auto disp = 0; disp < max_disparity; ++disp) {
      if (col - disp < 0) {
        cost_volume[row][col][disp] = NumericLimits<scalar_t>::infinity();
        continue;
      }

      const int row_start = max(row - kernel_size, 0);
      const int row_end = min(row + kernel_size, int(height));

      const int col_start =
          abs(min(col - disp - kernel_size, 0)) + col - kernel_size;
      const int col_end = min(col + kernel_size, int(width));

      scalar_t cost_value = 0;
      for (int krow = row_start; krow < row_end; ++krow) {
        for (int kcol = col_start; kcol < col_end; ++kcol) {
          const scalar_t left_intensity =
              tex2D<float>(left_image.texture, kcol, krow);
          const scalar_t right_intensity =
              tex2D<float>(right_image.texture, kcol - disp, krow);
          const scalar_t diff = left_intensity - right_intensity;

          cost_value += diff * diff;
        }
      }
      cost_volume[row][col][disp] = cost_value;
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
        SSDTextureKernel<scalar_t> ssd_kernel(
            left_image, right_image, cost_volume, kernel_size,
            std::numeric_limits<scalar_t>::infinity());
        KernelLauncher<kCUDA>::Launch2D(ssd_kernel, left_image.get_width(),
                                        left_image.get_height());
      }));
}
}  // namespace stereomatch
