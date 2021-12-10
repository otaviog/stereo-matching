#include "cuda_texture.hpp"

namespace stereomatch {

CUDATexture CUDATexture::FromTensor(const torch::Tensor &tensor) {
  // torch::Tensor tensor =
  //_tensor.permute({0, 1, 2}).to(torch::kFloat32).contiguous();

  if (!tensor.device().is_cuda())
    throw std::runtime_error("CUDATexture::FromTensor: passing a GPU tensor");

  CudaSafeCall(cudaDeviceSynchronize());
  cudaArray *array;
  int height = tensor.size(0);
  int width = tensor.size(1);
  int channels = tensor.size(2);

  {
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    CudaSafeCall(cudaMalloc3DArray(
        &array, &channel_desc, make_cudaExtent(width, height, channels), 0));
  }

  {
    cudaMemcpy3DParms copy_params = {0};
    memset(&copy_params, 0, sizeof(copy_params));
    copy_params.srcPtr = make_cudaPitchedPtr(tensor.data_ptr<float>(),
                                             sizeof(float) * width
                                             //* channels
                                             ,
                                             width, height);
    copy_params.dstArray = array;
    copy_params.extent = make_cudaExtent(width, height, channels);
    copy_params.kind = cudaMemcpyDeviceToDevice;
    CudaSafeCall(cudaMemcpy3D(&copy_params));
  }

  cudaTextureObject_t texture_object;
  {
    cudaResourceDesc tex_resource;
    memset(&tex_resource, 0, sizeof(cudaResourceDesc));
    tex_resource.resType = cudaResourceTypeArray;
    tex_resource.res.array.array = array;

    cudaTextureDesc tex_description;
    memset(&tex_description, 0, sizeof(cudaTextureDesc));

#if 1
    tex_description.addressMode[0] = tex_description.addressMode[1] =
        tex_description.addressMode[2] = cudaAddressModeClamp;
#else
    tex_description.addressMode[0] = tex_description.addressMode[1] =
        tex_description.addressMode[2] = cudaAddressModeBorder;
    tex_description.borderColor[0] = tex_description.borderColor[1] =
        tex_description.borderColor[2] = tex_description.borderColor[3] = 0.0f;
#endif
    tex_description.filterMode = cudaFilterModePoint;
    tex_description.readMode = cudaReadModeElementType;

    CudaSafeCall(cudaCreateTextureObject(&texture_object, &tex_resource,
                                         &tex_description, NULL));
  }

  return CUDATexture(texture_object, array, width, height, channels);
}

torch::Tensor CUDATexture::ToTensor() const {
  torch::Tensor tensor = torch::empty({height_, width_, channels_},
                                      torch::Device(torch::DeviceType::CUDA));
  cudaMemcpy3DParms copy_params = {0};
  memset(&copy_params, 0, sizeof(copy_params));
  copy_params.srcArray = array_;

  copy_params.dstPtr = make_cudaPitchedPtr(tensor.data_ptr<float>(),
                                           sizeof(float) * width_
                                           //* channels
                                           ,
                                           width_, height_);
  copy_params.extent = make_cudaExtent(width_, height_, channels_);
  copy_params.kind = cudaMemcpyDeviceToDevice;
  CudaSafeCall(cudaMemcpy3D(&copy_params));

  return tensor;
}

void CUDATexture::RegisterPybind(pybind11::module &m) {
  pybind11::class_<CUDATexture>(m, "CUDATexture")
      .def(py::init<>())
      .def_static("from_tensor", &CUDATexture::FromTensor)
      .def("to_tensor", &CUDATexture::ToTensor)
      .def_readonly("width", &CUDATexture::width_)
      .def_readonly("height", &CUDATexture::height_)
      .def_readonly("channels", &CUDATexture::channels_);
}
}  // namespace stereomatch
