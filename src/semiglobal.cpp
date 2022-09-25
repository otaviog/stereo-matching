#include "semiglobal.hpp"

#include <cassert>

#include "aggregation.hpp"
#include "bordered_buffer.hpp"
#include "check.hpp"

namespace stereomatch {

template <typename H, typename... T>
constexpr H get_min(H head, T... tail) {
  auto min = head;
  ((tail < min ? min = tail, 0 : 0), ...);
  return min;
}

std::vector<SGPixelPath> SGPixelPath::GeneratePaths(size_t width,
                                                    size_t height) noexcept {
  std::vector<SGPixelPath> path_descs;

  /**
   * Horizontal paths
   */
  for (auto i = 0; i < height; i++) {
    path_descs.push_back(SGPixelPath(Point2<int16_t>(0, i),
                                     Point2<int16_t>(width - 1, i),
                                     Point2<int16_t>(1, 0), width));
  }

  /**
   * Vertical paths
   */
  for (auto i = 0; i < width; i++) {
    path_descs.push_back(SGPixelPath(Point2<int16_t>(i, 0),
                                     Point2<int16_t>(i, height - 1),
                                     Point2<int16_t>(0, 1), height));
  }
  /**
   * Diagonal left to right
   */
  const auto lastX = width - 1;
  const auto lastY = height - 1;

  for (size_t i = 0; i < width; i++) {
    const Point2<int16_t> start(width - 1 - i, 0);
    const Point2<int16_t> end(width - 1, std::min(i, height - 1));
    path_descs.push_back(
        SGPixelPath(start, end, Point2<int16_t>(1, 1), end.y - start.y + 1));
  }

  for (size_t i = 0; i < height - 1; i++) {
    const size_t offset = width + i;

    const Point2<int16_t> start(0, i + 1);
    const Point2<int16_t> end(std::min(height - 1 - (i + 1), width - 1),
                              height - 1);

    path_descs.push_back(
        SGPixelPath(start, end, Point2<int16_t>(1, 1), end.x - start.x + 1));
  }

  /**
   * Diagonal right to left
   */
  for (size_t i = 0; i < width; i++) {
    const int sX = i;
    const int sY = 0;
    const int diagSize = std::min((int)i + 1, (int)height);

    path_descs.push_back(SGPixelPath(Point2<int16_t>(sX, sY),
                                     Point2<int16_t>(0, diagSize - 1),
                                     Point2<int16_t>(-1, 1), diagSize));
  }

  for (size_t i = 0; i < height - 1; i++) {
    const size_t offset = width + i;

    const int sX = lastX;
    const int sY = i + 1;
    const int diagSize = std::min(static_cast<int>(height - int(i + 1)),
                                  static_cast<int>(width));

    path_descs.push_back(
        SGPixelPath(Point2<int16_t>(sX, sY),
                    Point2<int16_t>(sX - diagSize + 1, sY + diagSize - 1),
                    Point2<int16_t>(-1, 1), diagSize));
  }

  return path_descs;
}

template <typename scalar_t>
struct SGMCostOperator {
  const torch::TensorAccessor<scalar_t, 3> cost_volume;
  const torch::TensorAccessor<scalar_t, 2> left_image;
  torch::TensorAccessor<scalar_t, 3> output_cost_vol;
  const scalar_t penalty1, penalty2;

  BorderedBuffer<scalar_t, 1> prev_cost, prev_cost_cache;

  SGMCostOperator(const torch::TensorAccessor<scalar_t, 3> cost_volume,
                  const torch::TensorAccessor<scalar_t, 2> left_image,
                  torch::TensorAccessor<scalar_t, 3> output_cost_vol,
                  scalar_t penalty1, scalar_t penalty2)
      : cost_volume(cost_volume),
        left_image(left_image),
        output_cost_vol(output_cost_vol),
        penalty1{penalty1},
        penalty2{penalty2},
        prev_cost(cost_volume.size(2),
                  std::numeric_limits<scalar_t>::infinity()),
        prev_cost_cache(cost_volume.size(2),
                        std::numeric_limits<scalar_t>::infinity()) {}

  void operator()(const SGPixelPath &path_desc) noexcept {
    const auto max_disparity = cost_volume.size(2);

    auto current_pixel = path_desc.start;
    const auto cost_volume_acc = cost_volume[current_pixel.y][current_pixel.x];
    auto output_cost_acc = output_cost_vol[current_pixel.y][current_pixel.x];

    for (auto disp = 0; disp < max_disparity; disp++) {
      const auto initial_cost = cost_volume_acc[disp];
      prev_cost[disp] = initial_cost;
      output_cost_acc[disp] += initial_cost;
    }

    scalar_t prev_intensity = left_image[current_pixel.y][current_pixel.x];
    for (auto i = 1; i < path_desc.size; ++i) {
      const auto prev_min_cost =
          *std::min_element(prev_cost.begin(), prev_cost.end());
      current_pixel += path_desc.direction;

      const auto intensity = left_image[current_pixel.y][current_pixel.x];

      const auto p2_adjusted =
          std::max(penalty1, penalty2 / std::abs(intensity - prev_intensity));

      prev_intensity = intensity;

      const auto cost_volume_acc =
          cost_volume[current_pixel.y][current_pixel.x];
      auto output_cost_acc = output_cost_vol[current_pixel.y][current_pixel.x];
      for (size_t disp = 0; disp < max_disparity; disp++) {
        const auto match_cost = cost_volume_acc[disp];
        const auto sgm_cost =
            match_cost +
            get_min(prev_cost[disp], prev_cost[disp - 1] + penalty1,
                    prev_cost[disp + 1] + penalty1,
                    prev_min_cost + p2_adjusted) -
            prev_min_cost;
        output_cost_acc[disp] += sgm_cost;
        prev_cost_cache[disp] = sgm_cost;
      }

      std::swap(prev_cost, prev_cost_cache);
    }
  }
};

void RunSemiglobalAggregationGPU(const torch::Tensor &cost_volume,
                                 const torch::Tensor &left_image,
                                 float penalty1, float penalty2,
                                 torch::Tensor &output_cost_volume);

void AggregationOps::RunSemiglobal(const torch::Tensor &cost_volume,
                                   const torch::Tensor &left_image,
                                   float penalty1, float penalty2,
                                   torch::Tensor &output_cost_volume) {
  const auto ref_device = cost_volume.device();
  STM_CHECK_DEVICE(ref_device, left_image);
  STM_CHECK_DEVICE(ref_device, output_cost_volume);

  if (ref_device.is_cuda()) {
    RunSemiglobalAggregationGPU(cost_volume, left_image, penalty1, penalty2,
                                output_cost_volume);
  } else {
    auto aggregation_paths(
        SGPixelPath::GeneratePaths(left_image.size(1), left_image.size(0)));

    AT_DISPATCH_FLOATING_TYPES(cost_volume.scalar_type(), "SemiglobalCPU", [&] {
      const auto max_disp = cost_volume.size(2);

      SGMCostOperator<scalar_t> sgm_cost_op(
          cost_volume.accessor<scalar_t, 3>(),
          left_image.accessor<scalar_t, 2>(),
          output_cost_volume.accessor<scalar_t, 3>(), scalar_t(penalty1),
          scalar_t(penalty2));

      for (const auto sg_path_desc : aggregation_paths) {
        sgm_cost_op(sg_path_desc);
        sgm_cost_op(sg_path_desc.inverse());
      }
    });
  }
}
}  // namespace stereomatch
