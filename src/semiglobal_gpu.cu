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
#include "numeric.hpp"

namespace stereomatch {

template <typename scalar_t>
struct SemiglobalKernel {
  const typename Accessor<kCUDA, scalar_t, 3>::T cost_volume;
  const typename Accessor<kCUDA, scalar_t, 2>::T left_image;
  const SGPixelPath* path_descriptors;
  const scalar_t penalty1, penalty2;
  typename Accessor<kCUDA, scalar_t, 3>::T output_cost_volume;

  SemiglobalKernel(const torch::Tensor& cost_volume,
                   const torch::Tensor& left_image,
                   const SGPixelPath* path_descriptors, scalar_t penalty1,
                   scalar_t penalty2, torch::Tensor output_cost_volume)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        left_image(Accessor<kCUDA, scalar_t, 2>::Get(left_image)),
        path_descriptors(thrust::raw_pointer_cast(path_descriptors)),
        penalty1{penalty1},
        penalty2{penalty2},
        output_cost_volume(
            Accessor<kCUDA, scalar_t, 3>::Get(output_cost_volume)) {}

  __device__ void operator()(int disp, int path) {
    const auto max_disparity = cost_volume.size(2);

    extern __shared__ __align__(sizeof(float)) uint8_t _shared_mem[];
    scalar_t* shr_prev_cost = (scalar_t*)_shared_mem;
    scalar_t* shr_prev_cost_min_search = &shr_prev_cost[max_disparity + 2];

    const auto path_desc(path_descriptors[path]);
    auto current_pixel = cast_point2<int2>(path_desc.start);

    const auto initial_cost =
        cost_volume[current_pixel.y][current_pixel.x][disp];
    shr_prev_cost[disp + 1] = initial_cost;
    shr_prev_cost_min_search[disp] = initial_cost;
    output_cost_volume[current_pixel.y][current_pixel.x][disp] += initial_cost;

    if (disp == 0) {
      // Pading borders
      shr_prev_cost[0] = shr_prev_cost[max_disparity + 1] =
          NumericLimits<scalar_t>::infinity();
    }

    scalar_t prev_intensity = left_image[current_pixel.y][current_pixel.x];
    for (auto i = 1; i < path_desc.size; ++i) {
      __syncthreads();  // Wait writes into of sgm_cost into the search array
      for (auto s = max_disparity >> 1; s >= 1; s = s >> 1) {
        if (disp < s) {
          const auto rhs_idx = s + disp;
          const auto rhs_cost = shr_prev_cost_min_search[rhs_idx];
          if (shr_prev_cost_min_search[disp] >= rhs_cost) {
            shr_prev_cost_min_search[disp] = rhs_cost;
          }
        }
        __syncthreads();
      }

      const auto prev_min_cost = shr_prev_cost_min_search[0];
      current_pixel.x += path_desc.direction.x;
      current_pixel.y += path_desc.direction.y;

      const auto intensity = left_image[current_pixel.y][current_pixel.x];
      const auto p2_adjusted =
          max(penalty1, penalty2 / abs(intensity - prev_intensity));

      prev_intensity = intensity;

      const auto match_cost =
          cost_volume[current_pixel.y][current_pixel.x][disp];
      const auto sgm_cost =
          match_cost +
          get_min(shr_prev_cost[disp + 1], shr_prev_cost[disp] + penalty1,
                  shr_prev_cost[disp + 2] + penalty1,
                  prev_min_cost + p2_adjusted) -
          prev_min_cost;

      output_cost_volume[current_pixel.y][current_pixel.x][disp] += sgm_cost;

      __syncthreads();  // Wait for all threads to read their neighbor costs
      shr_prev_cost[disp + 1] = sgm_cost;
      shr_prev_cost_min_search[disp] = sgm_cost;
    }
  }
};

template <typename T>
static __global__ void LaunchKernel(SemiglobalKernel<T> kernel,
                                    int path_descriptor_count,
                                    int max_disparity) {
  const int path_descriptor_idx = blockIdx.x;
  const int disparity = threadIdx.x;

  if (path_descriptor_idx < path_descriptor_count &&
      disparity < max_disparity) {
    kernel(disparity, path_descriptor_idx);
  }
}

void RunSemiglobalAggregationGPU(const torch::Tensor& cost_volume,
                                 const torch::Tensor& left_image,
                                 float penalty1, float penalty2,
                                 torch::Tensor& output_cost_volume) {
  thrust::device_vector<SGPixelPath> path_descriptors(
      SGPixelPath::GeneratePaths(left_image.size(1), left_image.size(0)));
  const auto max_disparity = cost_volume.size(2);
  AT_DISPATCH_FLOATING_TYPES(
      cost_volume.scalar_type(), "RunSemiglobalAggregation", [&] {
        SemiglobalKernel<scalar_t> kernel(
            cost_volume, left_image,
            thrust::raw_pointer_cast(path_descriptors.data()), penalty1,
            penalty2, output_cost_volume);
        LaunchKernel<<<path_descriptors.size(), max_disparity,
                       (2 * max_disparity + 3) * sizeof(scalar_t)>>>(
            kernel, path_descriptors.size(), max_disparity);
      });
}
}  // namespace stereomatch
