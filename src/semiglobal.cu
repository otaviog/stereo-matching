/**
 *
 * Hirschmüller, Heiko. "Semi-global matching-motivation, developments and
 * applications." Photogrammetric Week 11 (2011): 173-184.
 */

#include "semiglobal.hpp"

#include <exception>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/device_vector.h>

#include "accessor.hpp"
#include "aggregation.hpp"
#include "check.hpp"

#define SG_MAX_DISP 256

namespace stereomatch {
template <typename data_t>
class CUDAArray {
 public:
  static CUDAArray FromCPU(const data_t* memory, size_t size);

  void size() {}

 private:
  data_t* array_;
};

inline __device__ float pathCost(float cost, float keep_disp_cost,
                                 float change_1_disp_cost1,
                                 float change_1_disp_cost2,
                                 float change_N_disp_cost,
                                 float min_cost_all_disps, float penalty1,
                                 float penalty2) {
  return (cost +
          fmin(change_1_disp_cost1 + penalty1,
               fmin(change_1_disp_cost2 + penalty1,
                    min_cost_all_disps + penalty2)) -
          min_cost_all_disps);
}

template <typename scalar_t>
struct SemiglobalKernel {
  const typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;
  const typename Accessor<kCUDA, scalar_t, 2>::T left_image;
  const SGPixelPath* path_descriptors;
  const float penalty1, penalty2;
  typename Accessor<kCUDA, scalar_t, 3>::T output_cost_volume;

  SemiglobalKernel(const torch::Tensor& cost_volume,
                   const torch::Tensor& left_image,
                   const SGPixelPath* path_descriptors,
                   float penalty1, float penalty2,
                   torch::Tensor output_cost_volume)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        left_image(Accessor<kCUDA, scalar_t, 2>::Get(left_image)),
        path_descriptors(thrust::raw_pointer_cast(path_descriptors)),
        penalty1(penalty1),
        penalty2(penalty2),
        output_cost_volume(
            Accessor<kCUDA, scalar_t, 3>::Get(output_cost_volume)) {}

  __device__ void operator()(int disp, int path) {
    __shared__ float prev_cost[SG_MAX_DISP + 2];
    __shared__ float prev_min_cost[SG_MAX_DISP];

    __shared__ float min_cost;
    __shared__ int2 current_pixel;

    __shared__ float adapted_penalty2;
    __shared__ float last_intensity;

    const SGPixelPath path_desc(path_descriptors[path]);

    __shared__ int2 current_path;
    if (disp == 0) {
      current_pixel = cast_point2<int2>(path_desc.start);
      prev_min_cost[0] = CUDART_INF_F;
      last_intensity = left_image[current_pixel.y][current_pixel.x];
    }

    __syncthreads();

    const float initial_cost =
        cost_volume[disp][current_pixel.y][current_pixel.x];

    prev_cost[disp + 1] = initial_cost + penalty1;
    prev_min_cost[disp] = initial_cost;
    output_cost_volume[disp][current_pixel.y][current_pixel.x] += initial_cost;

    __syncthreads();

    const ushort max_disparity = cost_volume.size(0);
    for (int x = 1; x < path_desc.size; x++) {
      int i = max_disparity >> 1;
      while (i != 0) {
        if (disp < i) {
          prev_min_cost[disp] =
              fminf(prev_min_cost[disp], prev_min_cost[disp + i]);
        }
        __syncthreads();
        i = i >> 1;
      }

      if (disp == 0) {
        min_cost = prev_min_cost[0];
        current_pixel.x += path_desc.direction.x;
        current_pixel.y += path_desc.direction.y;

        const float intensity = left_image[current_pixel.y][current_pixel.x];

        adapted_penalty2 = penalty2 / abs(intensity - last_intensity);
        last_intensity = intensity;
      }

      __syncthreads();

      float current_cost =
          pathCost(cost_volume[disp][current_pixel.y][current_pixel.x], prev_cost[disp],
				   prev_cost[disp - 1], prev_cost[disp + 1],
                   min_cost,
				   penalty1, penalty1, adapted_penalty2);

      __syncthreads();

      prev_cost[disp + 1] = prev_min_cost[disp] = current_cost;

      output_cost_volume[disp][current_pixel.y][current_pixel.x] +=
          current_cost;

      __syncthreads();
    }
  }
};

template <typename T>
static __global__ void LaunchKernel(SemiglobalKernel<T> kernel,
                                    int path_descriptor_count,
                                    int max_disparity) {
  const int path_descriptor_idx = threadIdx.x;
  const int disparity = blockIdx.x;

  if (path_descriptor_idx < path_descriptor_count &&
      disparity < max_disparity) {
    kernel(path_descriptor_idx, disparity);
  }
}

static void RunSemiglobalAggregation(const torch::Tensor& cost_volume,
                                     const torch::Tensor& left_image,
                                     torch::Tensor& output_cost_volume) {
  const auto ref_device = cost_volume.device();
  STM_CHECK_DEVICE(ref_device, cost_volume);
  STM_CHECK_DEVICE(ref_device, output_cost_volume);

  if (ref_device.is_cuda()) {
    thrust::device_vector<SGPixelPath> path_descriptors(
        SGPixelPath::GeneratePaths(left_image.size(1),
                                                 left_image.size(0)));

    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunSemiglobalAggregation", [&] {
          SemiglobalKernel<scalar_t> kernel(
              cost_volume, left_image,
              thrust::raw_pointer_cast(path_descriptors.data()), 0.1f, 0.5f,
              output_cost_volume);
          LaunchKernel<<<path_descriptors.size(), cost_volume.size(0)>>>(
              kernel, path_descriptors.size(), cost_volume.size(0));
        });
  } else {
    throw std::runtime_error("RunSemiglobalAggregation not implemented on CPU");
  }
}
}  // namespace stereomatch
