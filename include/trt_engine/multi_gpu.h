#pragma once

#include <trt_engine/cuda_utils.h>
#include <trt_engine/engine.h>
#include <trt_engine/types.h>

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace trt_engine {

// ── Multi-GPU inference engine ──────────────────────────────────────────
// Maintains one InferenceEngine per GPU device and distributes inference
// requests across them using round-robin load balancing.
class MultiGPUEngine {
public:
    /// @param engine_path  Path to the serialized TensorRT engine
    /// @param device_ids   List of CUDA device IDs to use
    /// @param config       Base engine configuration (device_id will be overridden per device)
    MultiGPUEngine(const std::string& engine_path,
                   const std::vector<int>& device_ids,
                   const EngineConfig& config = {});

    ~MultiGPUEngine() = default;

    // Not copyable or movable
    MultiGPUEngine(const MultiGPUEngine&) = delete;
    MultiGPUEngine& operator=(const MultiGPUEngine&) = delete;

    /// Run inference on the next available device (round-robin).
    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);

    /// Async inference on the next available device.
    std::future<InferenceResult> infer_async(
        const std::vector<std::vector<float>>& input_buffers);

    /// Number of devices this engine spans.
    int get_device_count() const;

    /// Query properties of a specific device by index (0-based into device_ids).
    DeviceProperties get_device_info(int index) const;

    /// Get the device IDs in use.
    const std::vector<int>& get_device_ids() const { return device_ids_; }

private:
    int select_device();

    std::string                                 engine_path_;
    std::vector<int>                            device_ids_;
    std::vector<std::unique_ptr<InferenceEngine>> engines_;
    std::atomic<uint64_t>                       rr_counter_{0};
};

}  // namespace trt_engine
