#include <trt_engine/multi_gpu.h>

#include <stdexcept>

namespace trt_engine {

MultiGPUEngine::MultiGPUEngine(const std::string& engine_path,
                               const std::vector<int>& device_ids,
                               const EngineConfig& config)
    : engine_path_(engine_path), device_ids_(device_ids) {

    if (device_ids_.empty()) {
        throw EngineException("MultiGPUEngine: device_ids list is empty");
    }

    int available = get_device_count();

    engines_.reserve(device_ids_.size());
    for (int dev_id : device_ids_) {
        if (dev_id < 0 || dev_id >= available) {
            throw EngineException("MultiGPUEngine: invalid device ID " +
                                  std::to_string(dev_id) +
                                  " (available: 0.." +
                                  std::to_string(available - 1) + ")");
        }

        EngineConfig dev_config = config;
        dev_config.device_id = dev_id;

        auto engine = InferenceEngine::create(engine_path_, dev_config);
        engines_.push_back(std::move(engine));

        get_logger().info("MultiGPUEngine: initialized device " +
                          std::to_string(dev_id));
    }

    get_logger().info("MultiGPUEngine: ready with " +
                      std::to_string(engines_.size()) + " devices");
}

int MultiGPUEngine::select_device() {
    uint64_t idx = rr_counter_.fetch_add(1, std::memory_order_relaxed);
    return static_cast<int>(idx % engines_.size());
}

InferenceResult MultiGPUEngine::infer(
    const std::vector<std::vector<float>>& input_buffers) {
    int idx = select_device();
    return engines_[idx]->infer(input_buffers);
}

std::future<InferenceResult> MultiGPUEngine::infer_async(
    const std::vector<std::vector<float>>& input_buffers) {
    int idx = select_device();
    return engines_[idx]->infer_async(input_buffers);
}

int MultiGPUEngine::get_device_count() const {
    return static_cast<int>(device_ids_.size());
}

DeviceProperties MultiGPUEngine::get_device_info(int index) const {
    if (index < 0 || index >= static_cast<int>(device_ids_.size())) {
        throw EngineException("MultiGPUEngine: device index out of range: " +
                              std::to_string(index));
    }
    return trt_engine::get_device_properties(device_ids_[index]);
}

}  // namespace trt_engine
