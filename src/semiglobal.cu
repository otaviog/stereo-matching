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
#include "math.hpp"

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
  const scalar_t penalty1, penalty2;
  typename Accessor<kCUDA, scalar_t, 3>::T output_cost_volume;

  SemiglobalKernel(const torch::Tensor& cost_volume,
                   const torch::Tensor& left_image,
                   const SGPixelPath* path_descriptors, float penalty1,
                   float penalty2, torch::Tensor output_cost_volume)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        left_image(Accessor<kCUDA, scalar_t, 2>::Get(left_image)),
        path_descriptors(thrust::raw_pointer_cast(path_descriptors)),
        penalty1(penalty1),
        penalty2(penalty2),
        output_cost_volume(
            Accessor<kCUDA, scalar_t, 3>::Get(output_cost_volume)) {}

  __device__ void operator()(int disp, int path) {
    __shared__ scalar_t prev_cost[SG_MAX_DISP + 2];
    __shared__ scalar_t prev_min_cost[SG_MAX_DISP];

    __shared__ scalar_t min_cost;
    __shared__ int2 current_pixel;

    __shared__ scalar_t adapted_penalty2;
    __shared__ scalar_t last_intensity;

    const auto path_desc(path_descriptors[path]);
    const auto max_disparity = cost_volume.size(2);
    
    __shared__ int2 current_path;
    if (disp == 0) {
      current_pixel = cast_point2<int2>(path_desc.start);
      prev_min_cost[0] = NumericLimits<scalar_t>::infinity();
      last_intensity = left_image[current_pixel.y][current_pixel.x];
      prev_cost[0] = prev_cost[max_disparity] = NumericLimits<scalar_t>::infinity();
    }

    __syncthreads();

    const auto disp_costs_acc = cost_volume[current_pixel.y][current_pixel.x];
    const auto initial_cost = disp_costs_acc[disp];

    prev_cost[disp + 1] = initial_cost + penalty1;
    prev_min_cost[disp] = initial_cost;

    auto output_disp_cost_acc = output_cost_volume[current_pixel.y][current_pixel.x];
    output_disp_cost_acc[disp] += initial_cost;

    __syncthreads();

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

        const auto intensity = left_image[current_pixel.y][current_pixel.x];

        adapted_penalty2 = penalty2 / abs(intensity - last_intensity);
        last_intensity = intensity;
      }

      __syncthreads();

      const auto current_cost =
          pathCost(disp_costs_acc[disp],
                   prev_cost[disp + 1], prev_cost[disp], prev_cost[disp + 2],
                   min_cost, penalty1, penalty1, adapted_penalty2);

      __syncthreads();

      prev_cost[disp + 1] = prev_min_cost[disp] = current_cost;

      output_disp_cost_acc[disp] += current_cost;

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
        SGPixelPath::GeneratePaths(left_image.size(1), left_image.size(0)));
    //const auto width = cost_volume.size(1);
    // const auto height = cost_volume.size(0);
    const auto max_disparity = cost_volume.size(2);
    AT_DISPATCH_FLOATING_TYPES(
        cost_volume.scalar_type(), "RunSemiglobalAggregation", [&] {
          SemiglobalKernel<scalar_t> kernel(
              cost_volume, left_image,
              thrust::raw_pointer_cast(path_descriptors.data()), 0.1f, 0.5f,
              output_cost_volume);
          LaunchKernel<<<path_descriptors.size(), max_disparity>>>(
              kernel, path_descriptors.size(), max_disparity);
        });
  } else {
    throw std::runtime_error("RunSemiglobalAggregation not implemented on CPU");
  }
}
}  // namespace stereomatch
