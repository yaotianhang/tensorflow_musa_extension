#include <musa_runtime.h>

#include <cstdint>

// Must enter tensorflow namespace to match the call signature in .cc files completely
namespace tensorflow {

// ==========================================
// 1. EmbeddingLookup Kernel Function
// ==========================================
template <typename T, typename Tidx>
__global__ void EmbeddingLookupKernel(const T* params, const Tidx* ids,
                                      T* output, int vocab_size,
                                      int embedding_dim, int total_count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < total_count) {
    int id_idx = index / embedding_dim;
    int dim_idx = index % embedding_dim;

    Tidx lookup_id = ids[id_idx];

    // Boundary check
    if (lookup_id >= 0 && lookup_id < vocab_size) {
      int src_idx = lookup_id * embedding_dim + dim_idx;
      output[index] = params[src_idx];
    } else {
      // Fill with 0 if out of bounds (static_cast ensures type safety)
      output[index] = static_cast<T>(0);
    }
  }
}

// ==========================================
// 2. Launcher Function (for calling from .cc files)
// ==========================================
template <typename T, typename Tidx>
void LaunchEmbeddingLookup(const T* params, const Tidx* ids, T* output,
                           int vocab_size, int embedding_dim, int total_count,
                           musaStream_t stream) {
  if (total_count <= 0) return;

  int block_size = 256;
  int grid_size = (total_count + block_size - 1) / block_size;

  // Note: No spaces allowed inside <<< >>>
  EmbeddingLookupKernel<T, Tidx><<<grid_size, block_size, 0, stream>>>(
      params, ids, output, vocab_size, embedding_dim, total_count);
}

// ==========================================
// 3. Explicit Instantiation
// ==========================================
// Only when explicitly written here will the compiler generate the corresponding binary symbols.
// We define a macro to simplify the code, ensuring each data type T automatically supports all index types Tidx.

#define INSTANTIATE_LAUNCH(T, Tidx)                                            \
  template void LaunchEmbeddingLookup<T, Tidx>(const T*, const Tidx*, T*, int, \
                                               int, int, musaStream_t);

// Define a macro: register all possible index types (int32, long, long long) for a data type T
#define REGISTER_ALL_INDICES(T)                                   \
  INSTANTIATE_LAUNCH(T, int32_t); /* int32 index */              \
  INSTANTIATE_LAUNCH(T, long);    /* int64 index (Linux long) */ \
  INSTANTIATE_LAUNCH(                                             \
      T,                                                          \
      long long); /* int64 index (Linux long long) - solve undefined symbol */

// -----------------------------------------------------
// A. Floating Point Types (Float, Double)
// -----------------------------------------------------
REGISTER_ALL_INDICES(float);
REGISTER_ALL_INDICES(double);

// -----------------------------------------------------
// B. Integer Types (Int32, Int64)
// -----------------------------------------------------
// This is the missing part that caused your previous error "Could not find device... T=DT_INT32"!
REGISTER_ALL_INDICES(int32_t);
REGISTER_ALL_INDICES(int64_t);

// -----------------------------------------------------
// C. (Optional) Half Precision Float (Half)
// -----------------------------------------------------
// If your MUSA environment supports Eigen::half or __half, uncomment the following
// #include <Eigen/Core>
// REGISTER_ALL_INDICES(Eigen::half);

}  // namespace tensorflow
