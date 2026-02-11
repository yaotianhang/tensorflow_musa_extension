/* Copyright @2020-2026 Moore Threads Technology Co., Ltd. All rights reserved.
 */
#include "tensorflow/core/util/bcast.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

// 使用模板类，通过 Mode 参数区分 And 和 Or
template <::musa::dnn::Binary::Mode mode>
class MusaLogicalBinaryOp : public MusaOpKernel {
 public:
  explicit MusaLogicalBinaryOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    // 1. 输入数据类型校验 - 确保输入为布尔类型
    OP_REQUIRES(
        ctx, in0.dtype() == DT_BOOL,
        errors::InvalidArgument("Input 0 must be boolean (DT_BOOL), got ",
                                DataTypeString(in0.dtype())));
    OP_REQUIRES(
        ctx, in1.dtype() == DT_BOOL,
        errors::InvalidArgument("Input 1 must be boolean (DT_BOOL), got ",
                                DataTypeString(in1.dtype())));

    // 2. 处理广播逻辑
    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for logical op: ",
                                        in0.shape().DebugString(), " vs ",
                                        in1.shape().DebugString()));

    // 3. 分配输出张量 (DT_BOOL)
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &out));

    if (out->NumElements() == 0) return;

    // 4. 准备 muDNN 句柄和张量描述符
    auto& handle = GetHandleByCtx(ctx);
    mTensor t0 = CreateMTensor(in0);
    mTensor t1 = CreateMTensor(in1);
    mTensor t_out = CreateMTensor(*out);

    // 5. 配置 muDNN Binary 算子
    ::musa::dnn::Binary op;

    // 使用模板参数设置模式 (LOGICAL_AND 或 LOGICAL_OR)
    auto status = op.SetMode(mode);
    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("muDNN Binary SetMode failed for logical op"));

    // 6. 执行算子
    status = op.Run(handle, t_out, t0, t1);

    if (status != mStatus::SUCCESS) {
      // 提供更详细的错误信息，包括操作模式
      const char* mode_str = (mode == ::musa::dnn::Binary::Mode::LOGICAL_OR)
                                 ? "LOGICAL_OR"
                                 : "LOGICAL_AND";
      LOG(ERROR) << "muDNN Logical binary op (" << mode_str
                 << ") Run failed with status: " << static_cast<int>(status);

      // 对于 NOT_SUPPORTED 状态，提供具体建议
      if (status == mStatus::NOT_SUPPORTED) {
        OP_REQUIRES(
            ctx, false,
            errors::Unimplemented(
                "muDNN does not support BOOL logical operations on this "
                "hardware/device. "
                "Consider using CPU fallback or check MUSA driver version."));
      }
    }

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal(
                    "muDNN Logical Run failed. "
                    "Check if muDNN supports BOOL kernels for this mode."));
  }
};

// --- 算子注册 ---

// 注册 LogicalOr
REGISTER_KERNEL_BUILDER(
    Name("LogicalOr").Device(DEVICE_MTGPU),
    MusaLogicalBinaryOp<::musa::dnn::Binary::Mode::LOGICAL_OR>);

// 注册 LogicalAnd
REGISTER_KERNEL_BUILDER(
    Name("LogicalAnd").Device(DEVICE_MTGPU),
    MusaLogicalBinaryOp<::musa::dnn::Binary::Mode::LOGICAL_AND>);

}  // namespace musa
}  // namespace tensorflow