/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */

#include <mudnn.h>
#include <mudnn_tensor.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {
namespace {

template <typename T>
class MusaZerosLikeOp : public MusaOpKernel {
 public:
  explicit MusaZerosLikeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    // 1. Create output Tensor with consistent shape
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    // Return early if number of elements is 0
    if (output->NumElements() == 0) return;

    // 2. Get MUSA handle and Tensor descriptor
    auto& h = GetHandleByCtx(ctx);
    auto out_mt = CreateMTensor(*output);

    // 3. Call the underlying Fill operator
    // You can find class Fill in mudnn_tensor.h, its usage is as follows:
    ::musa::dnn::Fill op;

    // Set fill value to 0
    // Note: Fill operator's SetValue usually receives double, internally it will
    // convert according to Tensor type automatically
    MTOP_CHECK_OK(op.SetValue(0.0), "Fill SetValue to 0", ctx);

    // 4. Execute filling
    MTOP_CHECK_OK_RUN(op.Run(h, out_mt), "Fill Run for ZerosLike", ctx);
  }
};

// =====================================================================
// 5. Register operator (supports 6 common types)
// =====================================================================
#define REGISTER_MUSA_ZEROS_LIKE(type)                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaZerosLikeOp<type>);

REGISTER_MUSA_ZEROS_LIKE(float);
REGISTER_MUSA_ZEROS_LIKE(Eigen::half);
REGISTER_MUSA_ZEROS_LIKE(double);
REGISTER_MUSA_ZEROS_LIKE(int32);
REGISTER_MUSA_ZEROS_LIKE(int64);
REGISTER_MUSA_ZEROS_LIKE(bool);

#undef REGISTER_MUSA_ZEROS_LIKE

}  // namespace
}  // namespace musa
}  // namespace tensorflow
