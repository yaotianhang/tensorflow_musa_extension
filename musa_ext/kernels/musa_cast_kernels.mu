#include <musa_runtime.h>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace musa {

// 1. Define kernel function
template <typename DstT>
__global__ void BoolCastKernel(const bool* src, DstT* dst, int n) {
  // Manually compute global index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = src[i] ? static_cast<DstT>(1) : static_cast<DstT>(0);
  }
}

// 2. Define launch function
template <typename DstT>
void LaunchBoolCast(const bool* src, DstT* dst, int n, musaStream_t stream) {
  if (n <= 0) return;

  // Manual configuration: 256 threads per block
  int threads_per_block = 256;
  // Calculate number of blocks needed, ensure all data is covered
  int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

  // Launch MUSA Kernel
  BoolCastKernel<DstT>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(src, dst, n);
}

// Explicit instantiation of supported types
template void LaunchBoolCast<float>(const bool*, float*, int, musaStream_t);
template void LaunchBoolCast<int32_t>(const bool*, int32_t*, int, musaStream_t);

}  // namespace musa
}  // namespace tensorflow
