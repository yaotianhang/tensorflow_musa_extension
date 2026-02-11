#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace musa {

// 1. Register Op definition (reuse TF standard definition, usually no need to rewrite REGISTER_OP, but for completeness)
// Note: TF core usually has already registered "Placeholder", we mainly responsible for registering Kernel.
// If it's standalone plugin development, register Kernel directly.

template <typename T>
class MusaPlaceholderOp : public OpKernel {
 public:
  explicit MusaPlaceholderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    // Placeholder may have some shape attributes, but usually no need to do heavy work in constructor
  }

  void Compute(OpKernelContext* ctx) override {
    // ！！！ Core Logic ！！！

    // If the program really enters this Compute function, something serious happened:
    // User called session.run(), but forgot to feed data to this placeholder
    // in feed_dict.

    if (ctx->output_required(0)) {
      ctx->CtxFailure(errors::InvalidArgument(
          "You must feed a value for placeholder tensor '", name(),
          "' with dtype ", DataTypeString(output_type(0))));
    }

    // Indeed, no need to call mudnn here, no need to allocate_output
    // Because if data comes in, TF framework layer will replace the output before this Op runs.
  }
};

// 2. Register Kernel
// We tell TF: If someone puts Placeholder on MUSA device, use this Kernel
// (Although it's just an error repeater)
#define REGISTER_PLACEHOLDER(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("Placeholder").Device("MUSA").TypeConstraint<TYPE>("dtype"), \
      MusaPlaceholderOp<TYPE>);

REGISTER_PLACEHOLDER(float);
REGISTER_PLACEHOLDER(double);
REGISTER_PLACEHOLDER(Eigen::half);
REGISTER_PLACEHOLDER(bfloat16);
REGISTER_PLACEHOLDER(int32);
REGISTER_PLACEHOLDER(int64);
REGISTER_PLACEHOLDER(bool);

#undef REGISTER_PLACEHOLDER

}  // namespace musa
}  // namespace tensorflow
