#include <musa_runtime.h>
#include <stdio.h>

#include <vector>

#include "device/musa_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session_options.h"

// Must include this header file to get StreamExecutor
#include "tensorflow/stream_executor/multi_platform_manager.h"

// Define the device name constant
extern "C" const char DEVICE_MTGPU[];

namespace tensorflow {
void ForceMusaOptimizationPassRegistration();
}

// Define the device name constant
const char DEVICE_MTGPU[] = "MUSA";

namespace tensorflow {
namespace musa {

class MusaDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      fprintf(stderr, ">>>> [MUSA] ERROR: musaGetDeviceCount failed: %d\n",
              err);
      return Status::OK();
    }

    fprintf(stderr,
            ">>>> [MUSA] DeviceFactory detected %d physical devices <<<<\n",
            count);

    for (int i = 0; i < count; ++i) {
      devices->push_back(strings::StrCat("/physical_device:MUSA:", i));
    }
    return Status::OK();
  }

  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    int count = 0;
    musaError_t err = musaGetDeviceCount(&count);
    if (err != musaSuccess) {
      return errors::Internal("Failed to get MUSA device count");
    }

    fprintf(stderr,
            ">>>> [MUSA] DeviceFactory creating %d logical instances <<<<\n",
            count);

    // Get "MUSA" platform manager
    auto platform_status =
        ::stream_executor::MultiPlatformManager::PlatformWithName("MUSA");
    if (!platform_status.ok()) {
      return platform_status.status();
    }
    auto* platform = platform_status.ValueOrDie();

    for (int i = 0; i < count; ++i) {
      DeviceAttributes attr;
      string name = strings::StrCat(name_prefix, "/device:MUSA:", i);
      attr.set_name(name);
      attr.set_device_type("MUSA");
      attr.set_memory_limit(16ULL * 1024 * 1024 * 1024);
      attr.mutable_locality()->set_bus_id(i);
      attr.set_physical_device_desc(strings::StrCat("device: MUSA device ", i));

      // Get Executor for current device
      auto executor_status = platform->ExecutorForDevice(i);
      if (!executor_status.ok()) {
        return executor_status.status();
      }
      auto* executor = executor_status.ValueOrDie();

      // Dependency injection: pass executor to MusaDevice
      devices->push_back(std::unique_ptr<Device>(
          new MusaDevice(Env::Default(), attr, i, executor)));

      fprintf(stderr,
              ">>>> [MUSA] Logical Device /device:MUSA:%d created. <<<<\n", i);
    }
    return Status::OK();
  }
};

REGISTER_LOCAL_DEVICE_FACTORY("MUSA", MusaDeviceFactory, 210);

}  // namespace musa
}  // namespace tensorflow

extern "C" {
void __attribute__((constructor)) OnMusaPluginLoad() {
  fprintf(stderr,
          "\n>>>> [MUSA] SUCCESS: MUSA Factory Object Registered via Global "
          "Constructor! <<<<\n");

  tensorflow::ForceMusaOptimizationPassRegistration();

  fprintf(stderr,
          ">>>> [MUSA] SUCCESS: MUSA Optimization Passes Activated! <<<<\n\n");
}
}
extern "C" void ForceLinkMusaAmpOptimizer();
