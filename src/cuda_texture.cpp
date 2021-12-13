#include "cuda_texture.hpp"

namespace {
bool ispower2(int n) {
  // https://stackoverflow.com/a/108360
  return (n & (n - 1)) == 0;
}
}  // namespace

namespace stereomatch {

void CUDATexture::Release() {
  if (is_empty()) {
    return;
  }

  std::cout << "Release" << std::endl;
  CudaSafeCall(cudaDestroyTextureObject(texture_object_));
  CudaSafeCall(cudaFreeArray(array_));
  texture_object_ = 0;
  array_ = nullptr;
}

void CUDATexture::CopyFromTensor(const torch::Tensor &tensor,
                                 bool normalized_coords) {
  if (!is_empty()) {
    throw std::runtime_error(
        "CUDATExture::CopyFromTensor: Loading into a non-empty instance.");
  }

  if (normalized_coords &&
      (!ispower2(tensor.size(0)) || !ispower2(tensor.size(1)))) {
    throw std::runtime_error(
        "CUDATExture::CopyFromTensor: Only power of 2 texture can use "
        "normalized coordinates");
  }

  CudaSafeCall(cudaDeviceSynchronize());
  const int channels = tensor.size(2);
  if (channels == 3) {
    throw std::runtime_error(
        "CUDATExture::CopyFromTensor: Tensor with 3 or more than 4 channels "
        "aren't "
        "supported");
  }
  if (channels > 4) {
    throw std::runtime_error(
        "CUDATExture::CopyFromTensor: Tensor with more than 4 channels aren't "
        "supported");
  }
  const int height = tensor.size(0);
  const int width = tensor.size(1);

  cudaChannelFormatDesc channel_desc;
  size_t type_size;
  switch (tensor.scalar_type()) {
    case torch::ScalarType::Float:
      type_size = sizeof(float);
      switch (channels) {
        case 1:
          channel_desc = cudaCreateChannelDesc<float1>();
          break;
        case 2:
          channel_desc = cudaCreateChannelDesc<float2>();
          break;
        case 4:
          channel_desc = cudaCreateChannelDesc<float4>();
          break;
      };
      break;
    case torch::ScalarType::Byte:
      type_size = sizeof(uint8_t);
      switch (channels) {
        case 1:
          channel_desc = cudaCreateChannelDesc<uchar1>();
          break;
        case 2:
          channel_desc = cudaCreateChannelDesc<uchar2>();
          break;
        default:
          channel_desc = cudaCreateChannelDesc<uchar4>();
      }
      break;
    default:
      throw std::runtime_error(
          "CUDATExture::CopyFromTensor: Input tensor's dtype not supported.");
  }

  auto memcpy_copy_kind = cudaMemcpyHostToDevice;
  if (tensor.is_cuda()) {
    memcpy_copy_kind = cudaMemcpyDeviceToDevice;
  }

  volatile auto stride = tensor.strides();

#if 1
  cudaArray *array = nullptr;
  CudaSafeCall(cudaMallocArray(&array, &channel_desc, width, height, 0));
  CudaSafeCall(cudaMemcpyToArray(array, 0, 0, tensor.data_ptr(),
                                 width * height * type_size * channels,
                                 memcpy_copy_kind));
#else
  cudaArray *array = nullptr;
  cudaMalloc3DArray(&array, &channel_desc,
                    make_cudaExtent(width, height, channels));
  cudaMemcpy3DParms copy_params;
  memset(&copy_params, 0, sizeof(cudaMemcpy3DParms));
  copy_params.dstArray = array;
  copy_params.extent = make_cudaExtent(width, height, channels);
  copy_params.kind = memcpy_copy_kind;
  copy_params.srcPtr = make_cudaPitchedPtr(
      tensor.data_ptr(), type_size * width * channels, width, height);
#endif

  CudaSafeCall(cudaDeviceSynchronize());
  cudaResourceDesc tex_resource;
  memset(&tex_resource, 0, sizeof(cudaResourceDesc));
  tex_resource.resType = cudaResourceTypeArray;
  tex_resource.res.array.array = array;

  cudaTextureDesc tex_description;
  memset(&tex_description, 0, sizeof(cudaTextureDesc));
  tex_description.addressMode[0] = tex_description.addressMode[1] =
      cudaAddressModeClamp;
  tex_description.filterMode = cudaFilterModePoint;
  tex_description.readMode = cudaReadModeElementType;
  tex_description.normalizedCoords = normalized_coords;

  cudaTextureObject_t texture_object;
  CudaSafeCall(cudaCreateTextureObject(&texture_object, &tex_resource,
                                       &tex_description, NULL));

  texture_object_ = texture_object;
  array_ = array;
  width_ = width;
  height_ = height;
  channels_ = channels;
}  // namespace stereomatch

torch::Tensor CUDATexture::ToTensor() const {
  torch::Tensor tensor = torch::empty({height_, width_, channels_});
  // torch::Device(torch::DeviceType::CUDA));
  size_t type_size;
  switch (tensor.scalar_type()) {
    case torch::ScalarType::Float:
      type_size = sizeof(float);
      break;
    case torch::ScalarType::Byte:
      type_size = sizeof(uint8_t);
      break;
    default:
      throw std::runtime_error(
          "CUDATexture::ToTensor: [BUG!] Texture created with non-supported "
          "format");
  }

  cudaMemcpy3DParms copy_params;
  memset(&copy_params, 0, sizeof(copy_params));
  copy_params.srcArray = array_;

  copy_params.dstPtr = make_cudaPitchedPtr(
      tensor.data_ptr(), type_size * width_ * height_, width_, height_);
  copy_params.extent = make_cudaExtent(width_, height_, channels_);
  // copy_params.kind = cudaMemcpyDeviceToDevice;
  copy_params.kind = cudaMemcpyDeviceToHost;
  CudaSafeCall(cudaMemcpy3D(&copy_params));
  /*
    CudaSafeCall(cudaMemcpyFromArray(tensor.data_ptr(), array_, 0, 0,
    type_size * width_ * height_ * channels_,
  cudaMemcpyDeviceToDevice));
  */
  return tensor;
}

void CUDATexture::RegisterPybind(pybind11::module &m) {
  pybind11::class_<CUDATexture>(m, "CUDATexture")
      .def(py::init<>())
      .def("copy_from_tensor", &CUDATexture::CopyFromTensor)
      .def("to_tensor", &CUDATexture::ToTensor)
      .def("release", &CUDATexture::Release)
      .def_readonly("width", &CUDATexture::width_)
      .def_readonly("height", &CUDATexture::height_)
      .def_readonly("channels", &CUDATexture::channels_)
      .def_readonly("dtype", &CUDATexture::scalar_type_)
      .def_readonly("cuda_texture_object", &CUDATexture::texture_object_)
      .def_static("_run_test_kernel", &CUDATexture::RunTestKernel)
      .def_static("_run_test_kernel2", &CUDATexture::RunTestKernel2);
}
}  // namespace stereomatch
