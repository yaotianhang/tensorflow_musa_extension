/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

// Rsqrt forward (using Unary class)
template <typename T>
class MusaRsqrtOp : public MusaOpKernel {
 public:
  explicit MusaRsqrtOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    if (input.NumElements() > 0) {
      auto in_mt = CreateMTensor(input, format_);
      auto out_mt = CreateMTensor(*output, format_);
      auto& h = GetHandleByCtx(context);
      mUnary op;
      op.SetMode(::musa::dnn::Unary::Mode::RSQRT);
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, in_mt), "RunRsqrt", context);
    }
  }
};

// RsqrtGrad backward (according to detection results, using Binary class)
template <typename T>
class MusaRsqrtGradOp : public MusaOpKernel {
 public:
  explicit MusaRsqrtGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}
  void Compute(OpKernelContext* context) override {
    // RsqrtGrad: y = input(0), dy = input(1)
    const Tensor& y = context->input(0);
    const Tensor& dy = context->input(1);
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, y.shape(), &dx));

    if (y.NumElements() > 0) {
      auto y_mt = CreateMTensor(y, format_);
      auto dy_mt = CreateMTensor(dy, format_);
      auto dx_mt = CreateMTensor(*dx, format_);
      auto& h = GetHandleByCtx(context);

      // Key: Detection shows RSQRT_BW belongs to Binary class
      ::musa::dnn::Binary op;
      op.SetMode(::musa::dnn::Binary::Mode::RSQRT_BW);

      // Binary::Run accepts (handle, output, input0, input1)
      // Corresponds to dx = BinaryOp(dy, y)
      MTOP_CHECK_OK_RUN(op.Run(h, dx_mt, dy_mt, y_mt), "RunRsqrtGrad", context);
    }
  }
};

// Register 6 data types
#define REGISTER_MUSA_RSQRT_KERNELS(type)                         \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Rsqrt").Device("MUSA").TypeConstraint<type>("T"),     \
      MusaRsqrtOp<type>);                                         \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("RsqrtGrad").Device("MUSA").TypeConstraint<type>("T"), \
      MusaRsqrtGradOp<type>);

REGISTER_MUSA_RSQRT_KERNELS(float);
REGISTER_MUSA_RSQRT_KERNELS(Eigen::half);
REGISTER_MUSA_RSQRT_KERNELS(bfloat16);
REGISTER_MUSA_RSQRT_KERNELS(double);
REGISTER_MUSA_RSQRT_KERNELS(int32);
REGISTER_MUSA_RSQRT_KERNELS(int64);

#undef REGISTER_MUSA_RSQRT_KERNELS

}  // namespace musa
}  // namespace tensorflow
