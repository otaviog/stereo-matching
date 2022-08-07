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

template <typename T>
constexpr T get_min(T v1, T v2, T v3, T v4) {
  return fmin(v1, fmin(v2, fmin(v3, v4)));
}
                     
template <typename scalar_t>
inline __device__ scalar_t GetCost(scalar_t cost, scalar_t keep_disp_cost,
                                   scalar_t change_1_disp_cost1,
                                   scalar_t change_1_disp_cost2,
                                   scalar_t min_cost_all_disps,
                                   scalar_t penalty1, scalar_t penalty2) {
  return cost + get_min(
      keep_disp_cost,
      change_1_disp_cost1 + penalty1,
      change_1_disp_cost2 + penalty2,
      min_cost_all_disps + penalty2) - min_cost_all_disps;
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
                   const SGPixelPath* path_descriptors, scalar_t penalty1,
                   scalar_t penalty2, torch::Tensor output_cost_volume)
      : cost_volume(Accessor<kCUDA, scalar_t, 3>::Get(cost_volume)),
        left_image(Accessor<kCUDA, scalar_t, 2>::Get(left_image)),
        path_descriptors(thrust::raw_pointer_cast(path_descriptors)),
        penalty1(penalty1),
        penalty2(penalty2),
        output_cost_volume(
            Accessor<kCUDA, scalar_t, 3>::Get(output_cost_volume)) {}

  __device__ void operator()(int disp, int path) {
    __shared__ scalar_t min_cost;
    __shared__ int2 current_pixel;

    __shared__ scalar_t adapted_penalty2;
    __shared__ scalar_t last_intensity;

    // extern __shared__ scalar_t prev_cost_memory[];

    const auto max_disparity = cost_volume.size(2);

    // scalar_t *prev_cost = prev_cost_memory;
    // scalar_t *prev_cost_min_search = &prev_cost_memory[max_disparity + 2];

    __shared__ scalar_t prev_cost[SG_MAX_DISP + 2];
    __shared__ scalar_t prev_cost_min_search[SG_MAX_DISP];

    const auto path_desc(path_descriptors[path]);

    __shared__ int2 current_path;
    if (disp == 0) {
      current_pixel = cast_point2<int2>(path_desc.start);
      // prev_min_cost[0] = NumericLimits<scalar_t>::infinity();
      last_intensity = left_image[current_pixel.y][current_pixel.x];
      prev_cost[0] = prev_cost[max_disparity] =
                     NumericLimits<scalar_t>::infinity();
    }

    __syncthreads();

    const auto initial_cost = cost_volume[current_pixel.y][current_pixel.x][disp];

    prev_cost[disp + 1] = initial_cost;
    prev_cost_min_search[disp] = initial_cost;
    output_cost_volume[current_pixel.y][current_pixel.x][disp] += initial_cost;

    __syncthreads();

    for (auto i = 1; i < path_desc.size; i++) {
      int search_idx = max_disparity >> 1;
      while (search_idx != 0) {
        if (disp < search_idx) {
          prev_cost_min_search[disp] =
              fminf(prev_cost_min_search[disp], prev_cost_min_search[disp + search_idx]);
        }
        __syncthreads();
        search_idx = search_idx >> 1;
      }

      if (disp == 0) {
        min_cost = prev_cost_min_search[0];
        current_pixel.x += path_desc.direction.x;
        current_pixel.y += path_desc.direction.y;

        const auto intensity = left_image[current_pixel.y][current_pixel.x];

        adapted_penalty2 = penalty2 / abs(intensity - last_intensity);
        last_intensity = intensity;
      }

      __syncthreads();

      
      const auto current_cost =
          GetCost(cost_volume[current_pixel.y][current_pixel.x][disp],
                  prev_cost[disp + 1], prev_cost[disp],
                  prev_cost[disp + 2], min_cost, penalty1, adapted_penalty2);

      __syncthreads();

      prev_cost[disp + 1] = current_cost;
      prev_cost_min_search[disp] = current_cost;

      output_cost_volume[current_pixel.y][current_pixel.x][disp] += current_cost;
      __syncthreads();
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
        LaunchKernel<<<path_descriptors.size(), max_disparity
                       //,(max_disparity*2 + 2)*sizeof(scalar_t)
                       >>>(kernel, path_descriptors.size(), max_disparity);
      });
}
}  // namespace stereomatch


