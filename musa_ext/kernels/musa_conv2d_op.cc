#include <type_traits>
#include <vector>

#include "mudnn_xmma.h"  // ::musa::dnn::Convolution
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "utils_op.h"

namespace tensorflow {
namespace musa {

namespace {

// Parse attr vector (strides / dilations) by data_format
inline int GetDimFromAttr(const std::vector<int32>& v, TensorFormat fmt,
                          char dim) {
  const int idx = GetTensorDimIndex(fmt, dim);
  return (idx >= 0) ? v[idx] : -1;
}

Status PermuteTensorOnMUSA(OpKernelContext* ctx, const Tensor& input,
                           Tensor* output, const std::vector<int64_t>& perm,
                           mFormat in_format, mFormat out_format) {
  if (input.dims() != static_cast<int>(perm.size())) {
    return errors::InvalidArgument("Permute rank mismatch: input rank=",
                                   input.dims(), ", perm size=", perm.size());
  }

  mHandle& h = GetHandleByCtx(ctx);

  mTensor in_mt = CreateMTensor(input, in_format);
  mTensor out_mt = CreateMTensor(*output, out_format);

  ::musa::dnn::Permute pop;
  auto s = pop.ConfigDimStride(out_mt, in_mt, static_cast<int>(perm.size()),
                               perm.data());
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("muDNN Permute ConfigDimStride failed.");
  }

  s = pop.Run(h, out_mt, in_mt);
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("muDNN Permute Run failed.");
  }

  return Status::OK();
}

Status ReorderConv2DFilterHWIOToOIHW(OpKernelContext* ctx,
                                     const Tensor& filter_hwio,
                                     Tensor* filter_oihw) {
  // HWIO -> OIHW : [H, W, I, O] -> [O, I, H, W]
  static const std::vector<int64_t> kPerm = {3, 2, 0, 1};

  // 这里 in/out format 只是给 CreateMTensor 打标签。
  // 由于 CreateMTensor 是通用 helper，没有 filter 专用格式时，
  // 用 NCHW 承载 4D tensor 是一个工程折中方案（重点是维度顺序已经通过 permute
  // 正确重排）。
  return PermuteTensorOnMUSA(ctx, filter_hwio, filter_oihw, kPerm,
                             mFormat::NCHW, mFormat::NCHW);
}

// Compute Conv2D output shape and explicit padding (top/bottom/left/right).
// Supports SAME / VALID only.
Status ComputeOutputAndPadding2D(int in_h, int in_w, int filter_h, int filter_w,
                                 int stride_h, int stride_w, int dilation_h,
                                 int dilation_w, Padding padding, int* out_h,
                                 int* out_w, int* pad_top, int* pad_bottom,
                                 int* pad_left, int* pad_right) {
  const int eff_kh = (filter_h - 1) * dilation_h + 1;
  const int eff_kw = (filter_w - 1) * dilation_w + 1;

  if (padding == Padding::VALID) {
    *out_h = (in_h - eff_kh + stride_h) / stride_h;
    *out_w = (in_w - eff_kw + stride_w) / stride_w;
    *pad_top = *pad_bottom = *pad_left = *pad_right = 0;
    return Status::OK();
  }

  if (padding == Padding::SAME) {
    *out_h = (in_h + stride_h - 1) / stride_h;  // ceil(in_h / stride_h)
    *out_w = (in_w + stride_w - 1) / stride_w;  // ceil(in_w / stride_w)

    const int pad_needed_h =
        std::max(0, (*out_h - 1) * stride_h + eff_kh - in_h);
    const int pad_needed_w =
        std::max(0, (*out_w - 1) * stride_w + eff_kw - in_w);

    *pad_top = pad_needed_h / 2;
    *pad_bottom = pad_needed_h - *pad_top;
    *pad_left = pad_needed_w / 2;
    *pad_right = pad_needed_w - *pad_left;
    return Status::OK();
  }

  return errors::InvalidArgument(
      "MUSA Conv2D only supports SAME/VALID padding.");
}

template <typename T>
Status RunMuDNNConv2D(OpKernelContext* ctx, const Tensor& input,
                      const Tensor& filter, Tensor* output,
                      TensorFormat data_format, int stride_h, int stride_w,
                      int dilation_h, int dilation_w, int pad_top,
                      int pad_bottom, int pad_left, int pad_right) {
  // mudnn_xmma.h::Convolution::SetNdInfo(length, pad, stride, dilation)
  // appears to take one pad value per spatial dim (symmetric padding).
  if (!(pad_top == pad_bottom && pad_left == pad_right)) {
    return errors::Unimplemented(
        "Current muDNN Conv2D path expects symmetric padding in SetNdInfo. "
        "Got pad_top=",
        pad_top, ", pad_bottom=", pad_bottom, ", pad_left=", pad_left,
        ", pad_right=", pad_right,
        ". Consider explicit pad + VALID conv for asymmetric SAME cases.");
  }

  auto& handle = GetHandleByCtx(ctx);

  // Input/output tensor format follows TF data_format
  //   mFormat io_format = format::getFormatByTF(data_format);
  mTensor x = CreateMTensor(input);
  mTensor y = CreateMTensor(*output);

  // filter: TF Conv2D official layout = HWIO [KH, KW, IC, OC]
  const int64 kh = filter.dim_size(0);
  const int64 kw = filter.dim_size(1);
  const int64 filter_ic = filter.dim_size(2);
  const int64 out_c = filter.dim_size(3);

  // Allocate temporary tensor for OIHW layout: [OC, IC, KH, KW]
  Tensor filter_oihw;
  Status alloc_st = ctx->allocate_temp(
      filter.dtype(), TensorShape({out_c, filter_ic, kh, kw}), &filter_oihw);
  if (!alloc_st.ok()) return alloc_st;

  // Reorder HWIO -> OIHW using muDNN Permute
  Status reorder_st = ReorderConv2DFilterHWIOToOIHW(ctx, filter, &filter_oihw);
  if (!reorder_st.ok()) return reorder_st;

  // Use NCHW-like format tag to describe OIHW tensor in current helper
  // convention
  mTensor w = CreateMTensor(filter_oihw, mFormat::NCHW);

  ::musa::dnn::Convolution conv;

  auto s = conv.SetGroups(1);
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("muDNN Convolution::SetGroups(1) failed.");
  }

  // Conv2D spatial dims: [H, W]
  const int pads[2] = {pad_top, pad_left};
  const int strides[2] = {stride_h, stride_w};
  const int dilations[2] = {dilation_h, dilation_w};

  s = conv.SetNdInfo(/*length=*/2, pads, strides, dilations);
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("muDNN Convolution::SetNdInfo failed.");
  }

  // Optional: set compute mode if your project needs mixed-precision policy.
  // s = conv.SetComputeMode(::musa::dnn::Convolution::ComputeMode::DEFAULT);
  // if (s != ::musa::dnn::Status::SUCCESS) {
  //   return errors::Internal("muDNN Convolution::SetComputeMode failed.");
  // }

  ::musa::dnn::Convolution::Algorithm algo;
  s = conv.GetRecommendForwardAlgorithm(handle, algo, y, x, w);
  LOG(INFO) << "[Conv2D] Before GetRecommendForwardAlgorithm";
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetRecommendForwardAlgorithm failed.");
  }

  size_t workspace_size = 0;
  s = conv.GetForwardWorkspaceSize(handle, workspace_size, y, x, w, algo);
  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal(
        "muDNN Convolution::GetForwardWorkspaceSize failed.");
  }

  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    musaError_t err = musaMalloc(&workspace_ptr, workspace_size);
    if (err != musaSuccess) {
      return errors::ResourceExhausted("musaMalloc workspace failed, size=",
                                       workspace_size);
    }
  }

  // NOTE:
  // MemoryMaintainer exact typedef is in mudnn_base.h and may vary by version.
  // If compilation fails here, adapt this block to your actual
  // typedef/signature.
  ::musa::dnn::MemoryMaintainer maintainer = nullptr;
  if (workspace_size > 0) {
    maintainer = [workspace_ptr,
                  workspace_size](size_t bytes) -> ::musa::dnn::MemoryHandler {
      if (workspace_ptr == nullptr || bytes > workspace_size) {
        return ::musa::dnn::MemoryHandler(nullptr, [](void*) {});
      }

      // Return a non-owning handle (no-op deleter), because workspace_ptr
      // is freed manually after conv.Run(...)
      return ::musa::dnn::MemoryHandler(workspace_ptr, [](void*) {});
    };
  }

  s = conv.Run(handle, y, x, w, algo, maintainer);

  if (workspace_ptr != nullptr) {
    musaFree(workspace_ptr);
  }

  if (s != ::musa::dnn::Status::SUCCESS) {
    return errors::Internal("muDNN Convolution::Run failed.");
  }

  return Status::OK();
}

}  // namespace

template <typename T>
class MusaConv2DOp : public MusaOpKernel {
 public:
  explicit MusaConv2DOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    // Official Conv2D attrs
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str_));

    // Some TF versions expose explicit_paddings attr on Conv2D
    if (ctx->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("explicit_paddings", &explicit_paddings_));
    }

    OP_REQUIRES(
        ctx, FormatFromString(data_format_str_, &data_format_),
        errors::InvalidArgument("Invalid data_format: ", data_format_str_));
    OP_REQUIRES(ctx, data_format_ == FORMAT_NHWC || data_format_ == FORMAT_NCHW,
                errors::InvalidArgument("Only NHWC/NCHW are supported, got: ",
                                        data_format_str_));

    OP_REQUIRES_OK(ctx, GetPaddingFromString(padding_str_, &padding_));
    OP_REQUIRES(
        ctx, padding_ == Padding::SAME || padding_ == Padding::VALID,
        errors::InvalidArgument(
            "Only SAME/VALID padding supported in this implementation."));

    OP_REQUIRES(
        ctx, strides_.size() == 4,
        errors::InvalidArgument("Conv2D strides attr must have 4 elements."));
    OP_REQUIRES(
        ctx, dilations_.size() == 4,
        errors::InvalidArgument("Conv2D dilations attr must have 4 elements."));

    stride_h_ = GetDimFromAttr(strides_, data_format_, 'H');
    stride_w_ = GetDimFromAttr(strides_, data_format_, 'W');
    dilation_h_ = GetDimFromAttr(dilations_, data_format_, 'H');
    dilation_w_ = GetDimFromAttr(dilations_, data_format_, 'W');

    const int stride_n = GetDimFromAttr(strides_, data_format_, 'N');
    const int stride_c = GetDimFromAttr(strides_, data_format_, 'C');
    const int dilation_n = GetDimFromAttr(dilations_, data_format_, 'N');
    const int dilation_c = GetDimFromAttr(dilations_, data_format_, 'C');

    OP_REQUIRES(ctx, stride_n == 1 && stride_c == 1,
                errors::InvalidArgument(
                    "Conv2D does not support strides in batch/channel dims."));
    OP_REQUIRES(
        ctx, dilation_n == 1 && dilation_c == 1,
        errors::InvalidArgument(
            "Conv2D does not support dilations in batch/channel dims."));
    OP_REQUIRES(ctx, stride_h_ > 0 && stride_w_ > 0,
                errors::InvalidArgument("Spatial strides must be > 0."));
    OP_REQUIRES(ctx, dilation_h_ > 0 && dilation_w_ > 0,
                errors::InvalidArgument("Spatial dilations must be > 0."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    const Tensor& filter = ctx->input(1);

    OP_REQUIRES(ctx, input.dims() == 4,
                errors::InvalidArgument("Conv2D input must be 4-D, got: ",
                                        input.shape().DebugString()));
    OP_REQUIRES(ctx, filter.dims() == 4,
                errors::InvalidArgument("Conv2D filter must be 4-D, got: ",
                                        filter.shape().DebugString()));

    // TF Conv2D filter layout: [KH, KW, IC, OC] (HWIO)
    const int64 kh = filter.dim_size(0);
    const int64 kw = filter.dim_size(1);
    const int64 filter_ic = filter.dim_size(2);
    const int64 out_c = filter.dim_size(3);

    const int n_idx = GetTensorDimIndex(data_format_, 'N');
    const int h_idx = GetTensorDimIndex(data_format_, 'H');
    const int w_idx = GetTensorDimIndex(data_format_, 'W');
    const int c_idx = GetTensorDimIndex(data_format_, 'C');

    const int64 N = input.dim_size(n_idx);
    const int64 H = input.dim_size(h_idx);
    const int64 W = input.dim_size(w_idx);
    const int64 C = input.dim_size(c_idx);

    OP_REQUIRES(ctx, C == filter_ic,
                errors::InvalidArgument(
                    "Input channels and filter in_channels mismatch: ", C,
                    " vs ", filter_ic));

    int out_h = 0, out_w = 0;
    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    OP_REQUIRES_OK(
        ctx, ComputeOutputAndPadding2D(
                 static_cast<int>(H), static_cast<int>(W), static_cast<int>(kh),
                 static_cast<int>(kw), stride_h_, stride_w_, dilation_h_,
                 dilation_w_, padding_, &out_h, &out_w, &pad_top, &pad_bottom,
                 &pad_left, &pad_right));

    OP_REQUIRES(ctx, out_h >= 0 && out_w >= 0,
                errors::InvalidArgument("Computed negative output size: [",
                                        out_h, ", ", out_w, "]"));

    TensorShape output_shape;
    if (data_format_ == FORMAT_NHWC) {
      output_shape = TensorShape({N, out_h, out_w, out_c});
    } else {
      output_shape = TensorShape({N, out_c, out_h, out_w});
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    OP_REQUIRES_OK(
        ctx, RunMuDNNConv2D<T>(ctx, input, filter, output, data_format_,
                               stride_h_, stride_w_, dilation_h_, dilation_w_,
                               pad_top, pad_bottom, pad_left, pad_right));
  }

 private:
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  std::string padding_str_;
  std::string data_format_str_;
  std::vector<int64>
      explicit_paddings_;  // parsed for compatibility, not used yet

  TensorFormat data_format_ = FORMAT_NHWC;
  Padding padding_ = Padding::SAME;

  int stride_h_ = 1;
  int stride_w_ = 1;
  int dilation_h_ = 1;
  int dilation_w_ = 1;
};

// Kernel registration (TF2-only common dtypes)
#define REGISTER_MUSA_CONV2D(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("Conv2D").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaConv2DOp<TYPE>)

REGISTER_MUSA_CONV2D(float);
REGISTER_MUSA_CONV2D(double);
REGISTER_MUSA_CONV2D(Eigen::half);
REGISTER_MUSA_CONV2D(bfloat16);

#undef REGISTER_MUSA_CONV2D

}  // namespace musa
}  // namespace tensorflow