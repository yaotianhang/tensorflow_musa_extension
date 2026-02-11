#include "utils_op.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
namespace musa {

namespace {
mType GetType(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_UINT8:
      return mType::UINT8;
    case DataType::DT_INT16:
      return mType::INT16;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_UINT16:
      return mType::UINT16;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_UINT32:
      return mType::UINT32;
    case DataType::DT_UINT64:
      return mType::UINT64;
    case DataType::DT_BOOL:
      return mType::BOOL;
    default:
      CHECK(false);
      throw;
  }
}
}  // namespace

mStatus MusaFree(void* ptr) {
  if (ptr) musaFree(ptr);
  return mStatus::SUCCESS;
}
mStatus MusaAllocate(size_t size, void** ptr) {
  musaMalloc(ptr, size);
  return mStatus::SUCCESS;
}

// ============================================================
// CreateMTensor: Minimal Strategy (Low-Dim -> Force NCHW)
// ============================================================
mTensor CreateMTensor(const Tensor& t, mFormat format) {
  mTensor rst;
  rst.SetAddr(
      const_cast<void*>(static_cast<const void*>(t.tensor_data().data())));
  rst.SetType(GetType(t.dtype()));

  // 1. Extract dimensions as-is (without padding, fully trust low-dimensional support from underlying)
  auto dims_raw = t.shape().dim_sizes();
  std::vector<int64_t> dims;
  for (auto d : dims_raw) {
    dims.push_back(d);
  }

  // 2. Format strategy (your core idea)
  // If it's real 4D data (Conv/BN input), must respect the incoming format (NHWC/NCHW),
  // so that Layout Optimization can be supported.
  if (dims.size() >= 4) {
    rst.SetFormat(format);
  }
  // If it's low dimensional data (1D parameters, 2D/3D intermediate variables),
  // force NCHW.
  // NCHW here represents "Linear Contiguous Memory".
  // - For Sub ([8]): linear memory -> won't crash.
  // - For BN Param ([32]): linear memory -> parameter read correctly.
  else {
    rst.SetFormat(mFormat::NCHW);
  }

  rst.SetNdInfo(static_cast<int>(dims.size()), dims.data());
  return rst;
}

// For operators that are not 4D
mTensor CreateMTensor(const Tensor& t) {
  mTensor rst;
  MTOP_CHECK_LOG(rst.SetAddr(t.data()), "SetAddr");
  MTOP_CHECK_LOG(rst.SetType(GetType(t.dtype())), "SetType");
  auto dims_int = t.shape().dim_sizes();
  MTOP_CHECK_LOG(
      rst.SetNdInfo(static_cast<int>(dims_int.size()),
                    reinterpret_cast<const int64_t*>(dims_int.data())),
      "SetNdInfo");
  return rst;
}

// ============================================================
// GetMusaFormat: Standard Logic
// ============================================================
mFormat GetMusaFormat(OpKernelConstruction* ctx) {
  string df;
  // Read attribute normally, don't do extra work
  if (ctx->HasAttr("data_format")) {
    if (ctx->GetAttr("data_format", &df).ok()) {
      return (df == "NCHW") ? mFormat::NCHW : mFormat::NHWC;
    }
  }
  // Default NHWC (for 4D operators)
  return mFormat::NHWC;
}

MusaDevice* GetDeviceByCtx(tensorflow::OpKernelContext* context) {
  DeviceBase* device_base = context->device();
  MusaDevice* musa_device = reinterpret_cast<MusaDevice*>(device_base);
  if (!musa_device) return nullptr;
  musaSetDevice(musa_device->get_device_id());
  return musa_device;
}

}  // namespace musa
}  // namespace tensorflow
