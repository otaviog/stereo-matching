#include "aggregation.hpp"

#define SG_MAX_DISP 256

namespace stereomatch {

struct AggregationPathDescriptor {
  ushort2 start_point, end_point;
  short2 direction;
  ushort size;
};

template <typename data_t>
class CUDAArray {
 public:
  static CUDAArray FromCPU(const data_t* memory, size_t size);

  void size() {}

 private:
  data_t* array_;
};

inline __device__ float pathCost(float cost, float lcD, float lcDm1,
                                 float lcDp1, float minDisp, float P2Adjust) {
  return cost + fmin4f(lcD, lcDm1 + P1, lcDp1 + P1, minDisp + P2Adjust) -
         minDisp;
}

template <Device dev, typename scalar_t>
struct SemiglobalKernel {
  const typename Accessor<dev, scalar_t, 3>::T cost_volume;
  typename Accessor<dev, int32_t, 2>::T disparity_image;

  SemiglobalKernel(const torch::Tensor& cost_volume,
                   torch::Tensor semiglobal_cost)
      : cost_volume(Accessor<dev, scalar_t, 3>::Get(cost_volume)),

        disparity_image(Accessor<dev, int, 2>::Get(disparity_image)) {}

  __device__ __host__ void operator()(int disp, int path) {
    __shared__ float prev_cost[SG_MAX_DISP + 2];
    __shared__ float prev_min_cost[SG_MAX_DISP];

    __shared__ float min_cost;
    __shared__ int2 current_point;

    __shared__ float2 sP2Adjust;
    __shared__ float2 sLastIntensity;

    const AggregationPathDescriptor path_desc(aggregation_paths[path]);

    __shared__ int2 current_path;
    if (z == 0) {
      current_point = path_desc.start;
      prev_min_cost[0] = CUDART_INF_F;

      last_intesity = left_image[current_point.y][current_point.x];
    }

    __syncthreads();

    const float initial_cost =
        cost_volume[disp][current_point.y][current_point.x];

    prev_cost[dist + 1] = initial_cost + P1;
    prev_min_cost[disp] = initial_cost;
    aggregation[disp][current_point.y][current_point.x] += initial_cost;

    __syncthreads();

    float fLr, bLr;
    ushort dimz = dsiDim.z;

    for (int x = 1; x < pathSize; x++) {
      int i = dimz >> 1;
      while (i != 0) {
        if (z < i) {
          previous_min_cost[z] =
              fminf(previous_min_cost[z], previous_min_cost[z + i]);
        }
        __syncthreads();
        i = i >> 1;
      }

      if (z == 0) {
        min_cost = previous_min_cost[0];
        current_point.x += dir.x;
        current_point.y += dir.y;

        const float intensity = left_image[current_point.y][current_point.x];

        sP2Adjust.forward = P2 / abs(intensity - last_intesity);
        last_intesity = intensity;
      }

      __syncthreads();

      fLr = pathCost(sCostDSIRowsPtrF[z],
                     // tex3D(texCost, forwardCPt.x, forwardCPt.y, z),
                     sPrevCostF[dz], sPrevCostF[dz - 1], sPrevCostF[dz + 1],
                     sMinCostF, sP2Adjust.forward);

      __syncthreads();

      previous_cost[dz] = fLr;
      previous_min_cost[z] = fLr;

      sAggregDSIRowsPtrF[z] += fLr;

      __syncthreads();
    }
  }
};
}  // namespace stereomatch
