#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "utils_op.h"
#include "mu/device/musa_device.h"
#include <vector>

// 声明kernel函数 - 修改参数类型为 const T* const*
namespace tensorflow {
namespace musa {
template <typename T>
void LaunchAddN(const T* const* inputs, T* output, int num_inputs, int n, musaStream_t stream);
} // namespace musa
} // namespace tensorflow

namespace tensorflow {
namespace musa {

template <typename T>
class MusaAddNOp : public MusaOpKernel {
 public:
  explicit MusaAddNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const int num_inputs = ctx->num_inputs();
    OP_REQUIRES(ctx, num_inputs >= 1,
                errors::InvalidArgument("AddN requires at least one input."));

    // 获取第一个输入作为参考
    const Tensor& first_input = ctx->input(0);
    TensorShape output_shape = first_input.shape();
    
    // 验证所有输入具有相同的形状
    for (int i = 1; i < num_inputs; ++i) {
      const Tensor& input = ctx->input(i);
      OP_REQUIRES(ctx, input.shape() == output_shape,
                  errors::InvalidArgument("Input ", i, " has shape ",
                                         input.shape().DebugString(),
                                         " which is incompatible with ",
                                         output_shape.DebugString()));
    }

    // 分配输出张量
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    
    if (output->NumElements() == 0) {
      return;
    }

    // 单输入情况：直接内存拷贝
    if (num_inputs == 1) {
      auto* device = GetDeviceByCtx(ctx);
      auto stream = device->GetStream();
      size_t total_bytes = first_input.TotalBytes();
      musaError_t err = musaMemcpyAsync(output->data(), first_input.data(), 
                                        total_bytes, musaMemcpyDeviceToDevice, stream);
      OP_REQUIRES(ctx, err == musaSuccess,
                  errors::Internal("MUSA AddN single input copy failed: ", 
                                 musaGetErrorString(err)));
      return;
    }

    // 多输入情况：使用自定义kernel
    const int64_t total_elements = output->NumElements();
    
    // 分配设备指针数组 - 使用 const T* 类型
    const T** d_input_ptrs = nullptr;
    size_t ptr_array_size = num_inputs * sizeof(const T*);
    musaError_t err = musaMalloc(&d_input_ptrs, ptr_array_size);
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA AddN failed to allocate pointer array: ", 
                               musaGetErrorString(err)));

    // 在主机端准备指针数组
    std::vector<const T*> h_input_ptrs(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      h_input_ptrs[i] = ctx->input(i).flat<T>().data(); // 直接获取const指针
    }

    // 将指针数组拷贝到设备
    err = musaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(), ptr_array_size,
                         musaMemcpyHostToDevice, GetDeviceByCtx(ctx)->GetStream());
    OP_REQUIRES(ctx, err == musaSuccess,
                errors::Internal("MUSA AddN failed to copy pointer array: ", 
                               musaGetErrorString(err)));

    // 启动自定义kernel
    T* output_ptr = output->flat<T>().data();
    LaunchAddN<T>(d_input_ptrs, output_ptr, num_inputs, total_elements, 
                  GetDeviceByCtx(ctx)->GetStream());

    // 清理设备内存
    musaFree(d_input_ptrs);
  }
};

// 注册 AddN 算子
#define REGISTER_MUSA_ADDN(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(Name("AddN")                                \
                              .Device(DEVICE_MTGPU)                   \
                              .TypeConstraint<TYPE>("T"),             \
                          MusaAddNOp<TYPE>);

REGISTER_MUSA_ADDN(float);
REGISTER_MUSA_ADDN(double);
REGISTER_MUSA_ADDN(Eigen::half);
REGISTER_MUSA_ADDN(bfloat16);
REGISTER_MUSA_ADDN(int32);
// 使用 tensorflow::int64 而不是 int64_t
REGISTER_MUSA_ADDN(tensorflow::int64);

#undef REGISTER_MUSA_ADDN

}  // namespace musa
}  // namespace tensorflow