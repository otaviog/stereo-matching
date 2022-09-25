#pragma once

#include <vector>

namespace stereomatch {

template <typename T, int border_size>
struct BorderedBuffer {
 public:
  typedef std::vector<T> ArrayType;
  typedef typename ArrayType::const_iterator const_iterator;

  BorderedBuffer(int size, T border_value) noexcept
      : array(size + border_size) {
    for (auto i = 0; i < border_size; ++i) {
      array[i] = array[size + border_size + i] = border_value;
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

}  // namespace stereomatch
