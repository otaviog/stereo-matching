#include "semiglobal.hpp"
#include "aggregation.hpp"

#include <cassert>
#include <vector>

namespace stereomatch {

template <typename H, typename... T>
constexpr H get_min(H head, T... tail) {
  auto min = head;
  ((tail < min ? min = tail, 0 : 0), ...);
  return min;
}

template <typename T, int border_size>
struct BorderedBuffer {
 public:
  typedef std::vector<T> ArrayType;
  typedef typename ArrayType::const_iterator const_iterator;

  BorderedBuffer(int size, T border_value) noexcept
      : array(size + border_size) {
    for (auto i = 0; i < border_size; ++i) {
      array[i] = array[size - 1 + i] = border_value;
    }
  }

  const T operator[](int idx) const noexcept {
    return array[idx + border_size];
  }

  T &operator[](int idx) noexcept { return array[idx + border_size]; }

  const_iterator begin() const { return array.begin() + border_size; }
  const_iterator end() const { return array.end() - border_size; }

 private:
  std::vector<T> array;
};

std::vector<SGPixelPath> SGPixelPath::GeneratePaths(size_t width,
                                                    size_t height) {
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
 public:
  SGMCostOperator(const torch::TensorAccessor<scalar_t, 3> cost_volume,
                  const torch::TensorAccessor<scalar_t, 2> intensity_image,
                  torch::TensorAccessor<scalar_t, 3> output_cost_vol,
                  scalar_t penalty1, scalar_t penalty2)
      : cost_volume(cost_volume),
        intensity_image(intensity_image),
        output_cost_vol(output_cost_vol),
        penalty1{penalty1},
        penalty2{penalty2},
        prev_cost(cost_volume.size(2),
                  std::numeric_limits<scalar_t>::infinity()),
        prev_cost_cache(cost_volume.size(2),
                        std::numeric_limits<scalar_t>::infinity()) {}

  void operator()(const SGPixelPath &path) noexcept {
    scalar_t prev_intensity = 0;
    const auto max_disparity = cost_volume.size(2);

    const auto X = path.start.x;
    const auto Y = path.start.y;
    const auto disparities = cost_volume[Y][X];
    auto output_cost = output_cost_vol[Y][X];

    for (auto disp = 0; disp < max_disparity; disp++) {
      const auto cost = disparities[disp];

      output_cost[disp] = cost;
      prev_cost[disp] = cost;
    }

    prev_intensity = intensity_image[Y][X];

    auto current_pixel = path.start + path.direction;
    for (auto i = 0; i < path.size - 1; ++i, current_pixel += path.direction) {
      const auto prev_min_cost =
          *std::min_element(prev_cost.begin(), prev_cost.end());

      const auto intensity = intensity_image[current_pixel.y][current_pixel.x];
      const auto p2_adjusted = std::max(penalty1, penalty2 / std::abs(intensity - prev_intensity));

      const auto disparities = cost_volume[current_pixel.y][current_pixel.x];
      auto output_cost = output_cost_vol[current_pixel.y][current_pixel.x];
      for (size_t disp = 0; disp < max_disparity; disp++) {
        const auto match_cost =
            disparities[disp];

        const auto sgm_cost =
            match_cost +
            get_min(prev_cost[disp], prev_cost[disp - 1] + penalty1,
                    prev_cost[disp + 1] + penalty1, prev_min_cost + p2_adjusted) - prev_min_cost;
        output_cost[disp] += sgm_cost;
        prev_cost_cache[disp] = sgm_cost;
      }

      prev_intensity = intensity;

      std::swap(prev_cost, prev_cost_cache);
    }
  }

 private:
  const torch::TensorAccessor<scalar_t, 3> cost_volume;
  const torch::TensorAccessor<scalar_t, 2> intensity_image;
  torch::TensorAccessor<scalar_t, 3> output_cost_vol;
  const scalar_t penalty1, penalty2;

  BorderedBuffer<scalar_t, 1> prev_cost, prev_cost_cache;
};

void AggregationModule::RunSemiglobal(const torch::Tensor &cost_volume,
                                      const torch::Tensor &left_image,
                                      float penalty1, float penalty2,
                                      torch::Tensor &output_cost_volume) {
  auto aggregation_paths(
      SGPixelPath::GeneratePaths(left_image.size(1), left_image.size(0)));

  AT_DISPATCH_FLOATING_TYPES(cost_volume.scalar_type(), "SemiglobalCPU", [&] {
    const auto max_disp = cost_volume.size(2);

    SGMCostOperator<scalar_t> sgm_cost_op(
        cost_volume.accessor<scalar_t, 3>(), left_image.accessor<scalar_t, 2>(),
        output_cost_volume.accessor<scalar_t, 3>(), scalar_t(penalty1),
        scalar_t(penalty2));

    for (const auto sg_path : aggregation_paths) {
      sgm_cost_op(sg_path);
      sgm_cost_op(sg_path.inverse());
    }
  });
}
}  // namespace stereomatch
