/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace musa {

class MusaSqueezeOp : public OpKernel {
 public:
  explicit MusaSqueezeOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("squeeze_dims", &squeeze_dims_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& input = c->input(0);

    // Calculate squeezed shape
    // We can directly use TF's utility functions, or manual logic:
    // Iterate through input.shape(), remove all dim_size == 1 that are in
    // squeeze_dims_
    TensorShape output_shape;
    for (int i = 0; i < input.dims(); ++i) {
      bool should_squeeze = false;
      if (input.dim_size(i) == 1) {
        if (squeeze_dims_.empty()) {
          should_squeeze = true;  // Default to squeeze all dimensions of size 1
        } else {
          for (int d : squeeze_dims_) {
            // Handle negative indices
            int positive_d = d < 0 ? d + input.dims() : d;
            if (i == positive_d) {
              should_squeeze = true;
              break;
            }
          }
        }
      }
      if (!should_squeeze) {
        output_shape.AddDim(input.dim_size(i));
      }
    }

    // Key: Zero-copy implementation
    // We don't allocate new memory, directly point output to input memory, only
    // change Shape descriptor
    Tensor output;
    if (!output.CopyFrom(input, output_shape)) {
      c->CtxFailure(errors::Internal("Failed to squeeze tensor shape"));
      return;
    }

    // Set result to output
    c->set_output(0, output);

    // std::cerr << ">>> [MUSA_DEBUG] Squeeze: " << input.shape().DebugString()
    //           << " -> " << output_shape.DebugString() << std::endl;
  }

 private:
  std::vector<int32> squeeze_dims_;
};

// 注册支持所有类型，因为 Squeeze 不看数据内容，只看形状
#define REGISTER_MUSA_SQUEEZE(type)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Squeeze").Device("MUSA").TypeConstraint<type>("T"), \
      MusaSqueezeOp);

REGISTER_MUSA_SQUEEZE(float);
REGISTER_MUSA_SQUEEZE(Eigen::half);
REGISTER_MUSA_SQUEEZE(bfloat16);
REGISTER_MUSA_SQUEEZE(int32);
REGISTER_MUSA_SQUEEZE(int64);
REGISTER_MUSA_SQUEEZE(bool);
REGISTER_MUSA_SQUEEZE(double);
REGISTER_MUSA_SQUEEZE(uint8);

}  // namespace musa
}  // namespace tensorflow
