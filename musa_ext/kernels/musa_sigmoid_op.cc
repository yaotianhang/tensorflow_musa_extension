#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

/**
 * Forward operator (using Unary SIGMOID)
 */
template <typename T>
class MusaSigmoidOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Unary::Mode::SIGMOID), "Set Sigmoid",
                  ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt), "Sigmoid Forward Run",
                      ctx);
  }
};

/**
 * Backward operator (using native Binary SIGMOID_BW)
 */
template <typename T>
class MusaSigmoidGradOp : public MusaOpKernel {
 public:
  explicit MusaSigmoidGradOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Print trace information for debugging
    // fprintf(stderr, ">>> [MUSA_NATIVE_BW] %s\n", name().c_str());

    const Tensor& y = ctx->input(0);   // Forward output Sigmoid(x)
    const Tensor& dy = ctx->input(1);  // Gradient passed back

    Tensor* dz = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &dz));

    if (y.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    auto y_mt = CreateMTensor(y, format_);
    auto dy_mt = CreateMTensor(dy, format_);
    auto dz_mt = CreateMTensor(*dz, format_);

    // Core: According to grep, it belongs to Binary class
    ::musa::dnn::Binary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::SIGMOID_BW),
                  "Set Sigmoid_BW", ctx);

    // Execute native backward: dz = sigmoid_bw(y, dy)
    // muDNN Binary typical parameter order: (handle, output, input1, input2)
    // For backward operator, input1 is forward output y, input2 is gradient dy
    // Try swapping y_mt and dy_mt order
    MTOP_CHECK_OK_RUN(op.Run(handle, dz_mt, dy_mt, y_mt),
                      "Sigmoid_BW Native Run", ctx);
  }
};
// 注册
REGISTER_KERNEL_BUILDER(
    Name("Sigmoid").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("SigmoidGrad").Device("MUSA").TypeConstraint<float>("T"),
    MusaSigmoidGradOp<float>);

}  // namespace musa
}  // namespace tensorflow
