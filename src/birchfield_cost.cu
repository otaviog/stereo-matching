
#include "cost.hpp"

#include <limits>

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"
#include "numeric.hpp"

namespace stereomatch {

template <typename scalar_t, int kKernelSize = 4>
struct BirchfieldKernelCPU {
  const typename Accessor<kCPU, scalar_t, 2>::T left_image;
  const typename Accessor<kCPU, scalar_t, 2>::T right_image;
  typename Accessor<kCPU, scalar_t, 3>::T cost_volume;

  BirchfieldKernelCPU(const torch::Tensor &left_image,
                      const torch::Tensor &right_image,
                      torch::Tensor &cost_volume)
      : left_image(Accessor<kCPU, scalar_t, 2>::Get(left_image)),
        right_image(Accessor<kCPU, scalar_t, 2>::Get(right_image)),
        cost_volume(Accessor<kCPU, scalar_t, 3>::Get(cost_volume)) {}

  void operator()(int y, int x) noexcept {
    const int width = cost_volume.size(1);
    const int max_disparity = cost_volume.size(2);

    auto left_scan_line = left_image[y];
    auto right_scan_line = right_image[y];

    for (auto disp = 0; disp < std::min(max_disparity, x + 1); ++disp) {
      const auto start = std::max(0, x - disp - kKernelSize) + disp;
      const auto end = std::min(width, x + kKernelSize);
      scalar_t cost_value = 0.0f;

      for (ushort p = start; p < end; p++) {
        const ushort lIdx = p + 1;
        const ushort rIdx = p - disp + 1;

        const scalar_t left_intensity = left_scan_line[lIdx];
        const scalar_t right_intensity = right_scan_line[rIdx];

        const scalar_t laI = 0.5f * (left_intensity + left_scan_line[lIdx - 1]);
        const scalar_t lbI = 0.5f * (left_intensity + left_scan_line[lIdx + 1]);

        const scalar_t raI =
            0.5f * (right_intensity + right_scan_line[rIdx - 1]);
        const scalar_t rbI =
            0.5f * (right_intensity + right_scan_line[rIdx + 1]);

        const scalar_t lImi = get_min(laI, lbI, left_intensity);
        const scalar_t lIma = get_max(laI, lbI, left_intensity);

        const scalar_t rImi = get_min(raI, rbI, right_intensity);
        const scalar_t rIma = get_max(raI, rbI, right_intensity);

        cost_value += min(
            get_max(scalar_t(0), left_intensity - rIma, rImi - left_intensity),
            get_max(scalar_t(0), right_intensity - lIma,
                    lImi - right_intensity));
      }

      cost_volume[y][x][disp] = cost_value;
    }
  }
};

template <typename scalar_t, int kKernelSize = 4>
struct BirchfieldKernelCUDA {
  const typename Accessor<kCUDA, scalar_t, 2>::T left_image;
  const typename Accessor<kCUDA, scalar_t, 2>::T right_image;
  typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;

  BirchfieldKernelCUDA(const torch::Tensor &left_image,
                       const torch::Tensor &right_image,
                       torch::Tensor &cost_volume)
      : left_image(Accessor<kCUDA, scalar_t, 2>::Get(left_image)),
        right_image(Accessor<kCUDA, scalar_t, 2>::Get(right_image)),
        cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)) {}

  __device__ void operator()(int y, int x) {
    const int width = cost_volume.size(1);

    extern __shared__ float shared_mem[];
    auto left_scan_line = shared_mem;
    auto right_scan_line = &shared_mem[width + 2];
    //__shared__ scalar_t left_scan_line[1024];
    //__shared__ scalar_t right_scan_line[1024];

    left_scan_line[x + 1] = left_image[y][x];
    right_scan_line[x + 1] = right_image[y][x];

    if (x == 0) {
      left_scan_line[0] = 0.0f;
      right_scan_line[0] = 0.0f;
      left_scan_line[width + 1] = 0.0f;
      right_scan_line[width + 1] = 0.0f;
    }

    __syncthreads();

    const int max_disparity = cost_volume.size(2);
    for (auto disp = 0; disp < min(max_disparity, x + 1); ++disp) {
      const auto start = max(0, x - disp - kKernelSize) + disp;
      const auto end = min(width, x + kKernelSize);
      scalar_t cost_value = 0.0f;

      for (ushort p = start; p < end; p++) {
        const ushort lIdx = p + 1;
        const ushort rIdx = p - disp + 1;

        const scalar_t left_intensity = left_scan_line[lIdx];
        const scalar_t right_intensity = right_scan_line[rIdx];

        const scalar_t laI = 0.5f * (left_intensity + left_scan_line[lIdx - 1]);
        const scalar_t lbI = 0.5f * (left_intensity + left_scan_line[lIdx + 1]);

        const scalar_t raI =
            0.5f * (right_intensity + right_scan_line[rIdx - 1]);
        const scalar_t rbI =
            0.5f * (right_intensity + right_scan_line[rIdx + 1]);

        const scalar_t lImi = get_min(laI, lbI, left_intensity);
        const scalar_t lIma = get_max(laI, lbI, left_intensity);

        const scalar_t rImi = get_min(raI, rbI, right_intensity);
        const scalar_t rIma = get_max(raI, rbI, right_intensity);

        cost_value += min(
            get_max(scalar_t(0), left_intensity - rIma, rImi - left_intensity),
            get_max(scalar_t(0), right_intensity - lIma,
                    lImi - right_intensity));
      }

      cost_volume[y][x][disp] = cost_value;
    }
  }
};

template <typename T>
static __global__ void LaunchKernel(BirchfieldKernelCUDA<T> kernel, int width,
                                    int height) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;

  if (row < height && col < width) {
    kernel(row, col);
  }
}

void CostOps::ComputeBirchfield(const torch::Tensor &left_image,
                                const torch::Tensor &right_image,
                                torch::Tensor cost_volume, int kernel_size) {
  const auto ref_device = left_image.device();

  STM_CHECK_DEVICE(ref_device, right_image);
  STM_CHECK_DEVICE(ref_device, cost_volume);
  const auto width = left_image.size(1);
  const auto height = left_image.size(0);

  if (ref_device.is_cuda()) {
    AT_DISPATCH_FLOATING_TYPES(
        left_image.scalar_type(), "BirchfieldKernel", [&] {
          const auto width = cost_volume.size(1);
          BirchfieldKernelCUDA<scalar_t, 4> kernel(left_image, right_image,
                                                   cost_volume);
          LaunchKernel<<<height, width, ((width + 2) * 2) * sizeof(scalar_t)>>>(
              kernel, width, height);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        left_image.scalar_type(), "BirchfieldKernel", [&] {
          BirchfieldKernelCPU<scalar_t, 4> kernel(left_image, right_image,
                                                  cost_volume);

          KernelLauncher<kCPU>::Launch2D(kernel, width, height);
        });
  }
}

}  // namespace stereomatch
