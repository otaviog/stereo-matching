#include "cuda_texture.hpp"

#include "accessor.hpp"
#include "check.hpp"
#include "kernel.hpp"

namespace stereomatch {

template <typename scalar_t>
struct CopyTextureTestKernel {
  cudaTextureObject_t input_texture;
  typename Accessor<kCUDA, scalar_t, 2>::T output_tensor;

  CopyTextureTestKernel(const CUDATexture& input_texture,
                        torch::Tensor output_tensor)
      : input_texture(input_texture),
        output_tensor(Accessor<kCUDA, scalar_t, 2>::Get(output_tensor)) {
    std::cout << input_texture << std::endl;
  }

  __device__ void operator()(int row, int col) {
    float width = output_tensor.size(1);
    float height = output_tensor.size(0);
    output_tensor[row][col] =
        tex2D<scalar_t>(input_texture, float(col), float(row));
  }
};

void CUDATexture::RunTransferTestKernel(const CUDATexture& input_texture,
                                torch::Tensor output_tensor) {
  if (!output_tensor.device().is_cuda()) {
    throw std::runtime_error("Cost volume must be a CUDA tensor.");
  }

  if (output_tensor.scalar_type() == torch::ScalarType::Float) {
    CopyTextureTestKernel<float> kernel(input_texture, output_tensor);
    KernelLauncher<kCUDA>::Launch2D(kernel, input_texture.get_width(),
                                    input_texture.get_height());
  } else if (output_tensor.scalar_type() == torch::ScalarType::Byte) {
    CopyTextureTestKernel<uint8_t> kernel(input_texture, output_tensor);
    KernelLauncher<kCUDA>::Launch2D(kernel, input_texture.get_width(),
                                    input_texture.get_height());
  } else {
    throw std::runtime_error("Cannot test this type");
  }

  cudaDeviceSynchronize();
}

__global__ void copyKernel(cudaTextureObject_t input,
                           torch::PackedTensorAccessor32<float, 2> output) {
  int width = output.size(1);
  int height = output.size(0);

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width) return;

  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= height) return;
  float u = x / (float)width;
  float v = y / (float)height;
  // Transform coordinates
  // u += 0.5f;
  // v += 0.5f;
  // Read from texture and write to global memory
  output[y][x] = tex2D<float>(input, u, v);
}

void CUDATexture::RunBindingTestKernel(const torch::Tensor& input_tensor,
									   torch::Tensor output_tensor) {
  const int width = input_tensor.size(1);
  const int height = input_tensor.size(0);
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray* cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  // Copy to device memory some data located at address h_data
  // in host memory
  size_t size = width * height * sizeof(float);
  cudaMemcpyToArray(cuArray, 0, 0, input_tensor.data_ptr(), size,
                    cudaMemcpyHostToDevice);
  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  // Allocate result of transformation in device memory
  // Invoke kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);
  copyKernel<<<dimGrid, dimBlock>>>(
      texObj, output_tensor.packed_accessor32<float, 2>());
  // Destroy texture object
  cudaDestroyTextureObject(texObj);
  // Free device memory
  cudaFreeArray(cuArray);
}
}  // namespace stereomatch
