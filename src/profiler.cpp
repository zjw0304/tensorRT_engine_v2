#include <trt_engine/profiler.h>
#include <trt_engine/logger.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

#ifdef TRT_ENGINE_HAS_NVML
#include <nvml.h>
#endif

namespace trt_engine {

// ═══════════════════════════════════════════════════════════════════════════
//  TRTProfiler
// ═══════════════════════════════════════════════════════════════════════════

void TRTProfiler::reportLayerTime(const char* layer_name, float ms) noexcept {
    if (!layer_name) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& ls = layers_[layer_name];
    ls.total_ms += ms;
    ls.call_count++;
}

std::map<std::string, float> TRTProfiler::get_layer_timings() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, float> result;
    for (const auto& [name, stats] : layers_) {
        result[name] = stats.total_ms;
    }
    return result;
}

std::string TRTProfiler::report_text() const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (layers_.empty()) {
        return "No layer timing data recorded.\n";
    }

    // Sort layers by total time descending
    std::vector<std::pair<std::string, LayerStats>> sorted_layers(
        layers_.begin(), layers_.end());
    std::sort(sorted_layers.begin(), sorted_layers.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_ms > b.second.total_ms;
              });

    float total_ms = 0.0f;
    for (const auto& [name, stats] : sorted_layers) {
        total_ms += stats.total_ms;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "=== TensorRT Layer Timing Report ===\n";
    oss << std::setw(60) << std::left << "Layer"
        << std::setw(12) << std::right << "Time(ms)"
        << std::setw(8) << "Calls"
        << std::setw(12) << "Avg(ms)"
        << std::setw(10) << "Pct(%)"
        << "\n";
    oss << std::string(102, '-') << "\n";

    for (const auto& [name, stats] : sorted_layers) {
        float avg = (stats.call_count > 0) ? stats.total_ms / stats.call_count : 0.0f;
        float pct = (total_ms > 0.0f) ? (stats.total_ms / total_ms) * 100.0f : 0.0f;
        oss << std::setw(60) << std::left << name
            << std::setw(12) << std::right << stats.total_ms
            << std::setw(8) << stats.call_count
            << std::setw(12) << avg
            << std::setw(9) << pct << "%"
            << "\n";
    }

    oss << std::string(102, '-') << "\n";
    oss << "Total: " << total_ms << " ms\n";

    return oss.str();
}

void TRTProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    layers_.clear();
}

// ═══════════════════════════════════════════════════════════════════════════
//  PerformanceProfiler
// ═══════════════════════════════════════════════════════════════════════════

void PerformanceProfiler::record_inference(double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.push_back(latency_ms);
}

size_t PerformanceProfiler::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latencies_.size();
}

double PerformanceProfiler::percentile(const std::vector<double>& sorted_data,
                                        double p) {
    if (sorted_data.empty()) return 0.0;
    if (sorted_data.size() == 1) return sorted_data[0];

    double rank = (p / 100.0) * static_cast<double>(sorted_data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(rank));
    size_t upper = static_cast<size_t>(std::ceil(rank));

    if (lower == upper || upper >= sorted_data.size()) {
        return sorted_data[lower];
    }

    double frac = rank - static_cast<double>(lower);
    return sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac;
}

PerformanceStats PerformanceProfiler::get_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PerformanceStats stats;
    stats.total_inferences = latencies_.size();

    if (latencies_.empty()) {
        return stats;
    }

    // Make a sorted copy
    std::vector<double> sorted = latencies_;
    std::sort(sorted.begin(), sorted.end());

    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();

    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.mean_ms = sum / static_cast<double>(sorted.size());

    stats.p50_ms = percentile(sorted, 50.0);
    stats.p95_ms = percentile(sorted, 95.0);
    stats.p99_ms = percentile(sorted, 99.0);

    // Throughput: inferences per second based on mean latency
    if (stats.mean_ms > 0.0) {
        stats.throughput_fps = 1000.0 / stats.mean_ms;
    }

    return stats;
}

void PerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.clear();
}

std::string PerformanceProfiler::report_text() const {
    auto stats = get_statistics();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "=== Performance Report ===\n";
    oss << "Total inferences: " << stats.total_inferences << "\n";
    oss << "Latency (ms):\n";
    oss << "  Min:  " << stats.min_ms << "\n";
    oss << "  Max:  " << stats.max_ms << "\n";
    oss << "  Mean: " << stats.mean_ms << "\n";
    oss << "  P50:  " << stats.p50_ms << "\n";
    oss << "  P95:  " << stats.p95_ms << "\n";
    oss << "  P99:  " << stats.p99_ms << "\n";
    oss << "Throughput: " << stats.throughput_fps << " inferences/sec\n";

    return oss.str();
}

std::string PerformanceProfiler::report_json() const {
    auto stats = get_statistics();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\n";
    oss << "  \"total_inferences\": " << stats.total_inferences << ",\n";
    oss << "  \"latency_ms\": {\n";
    oss << "    \"min\": " << stats.min_ms << ",\n";
    oss << "    \"max\": " << stats.max_ms << ",\n";
    oss << "    \"mean\": " << stats.mean_ms << ",\n";
    oss << "    \"p50\": " << stats.p50_ms << ",\n";
    oss << "    \"p95\": " << stats.p95_ms << ",\n";
    oss << "    \"p99\": " << stats.p99_ms << "\n";
    oss << "  },\n";
    oss << "  \"throughput_fps\": " << stats.throughput_fps << "\n";
    oss << "}\n";

    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
//  GPU Metrics (NVML-based)
// ═══════════════════════════════════════════════════════════════════════════

#ifdef TRT_ENGINE_HAS_NVML

// RAII helper to ensure nvmlInit/nvmlShutdown pairing
namespace {
struct NvmlGuard {
    bool initialized = false;
    NvmlGuard() {
        nvmlReturn_t ret = nvmlInit_v2();
        initialized = (ret == NVML_SUCCESS);
        if (!initialized) {
            get_logger().warning("NVML initialization failed");
        }
    }
    ~NvmlGuard() {
        if (initialized) {
            nvmlShutdown();
        }
    }
};

static nvmlDevice_t get_nvml_device(int device_id) {
    nvmlDevice_t device;
    nvmlReturn_t ret = nvmlDeviceGetHandleByIndex_v2(
        static_cast<unsigned int>(device_id), &device);
    if (ret != NVML_SUCCESS) {
        return nullptr;
    }
    return device;
}
}  // namespace

unsigned int PerformanceProfiler::gpu_utilization(int device_id) {
    NvmlGuard guard;
    if (!guard.initialized) return 0;

    nvmlDevice_t device = get_nvml_device(device_id);
    if (!device) return 0;

    nvmlUtilization_t util;
    nvmlReturn_t ret = nvmlDeviceGetUtilizationRates(device, &util);
    if (ret != NVML_SUCCESS) return 0;

    return util.gpu;
}

size_t PerformanceProfiler::memory_used(int device_id) {
    NvmlGuard guard;
    if (!guard.initialized) return 0;

    nvmlDevice_t device = get_nvml_device(device_id);
    if (!device) return 0;

    nvmlMemory_t mem;
    nvmlReturn_t ret = nvmlDeviceGetMemoryInfo(device, &mem);
    if (ret != NVML_SUCCESS) return 0;

    return static_cast<size_t>(mem.used);
}

unsigned int PerformanceProfiler::temperature(int device_id) {
    NvmlGuard guard;
    if (!guard.initialized) return 0;

    nvmlDevice_t device = get_nvml_device(device_id);
    if (!device) return 0;

    unsigned int temp = 0;
    nvmlReturn_t ret = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (ret != NVML_SUCCESS) return 0;

    return temp;
}

unsigned int PerformanceProfiler::power_usage(int device_id) {
    NvmlGuard guard;
    if (!guard.initialized) return 0;

    nvmlDevice_t device = get_nvml_device(device_id);
    if (!device) return 0;

    unsigned int power = 0;
    nvmlReturn_t ret = nvmlDeviceGetPowerUsage(device, &power);
    if (ret != NVML_SUCCESS) return 0;

    return power;
}

GpuMetrics PerformanceProfiler::get_gpu_metrics(int device_id) {
    GpuMetrics metrics;

    NvmlGuard guard;
    if (!guard.initialized) return metrics;

    nvmlDevice_t device = get_nvml_device(device_id);
    if (!device) return metrics;

    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
        metrics.gpu_utilization_percent = util.gpu;
    }

    nvmlMemory_t mem;
    if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) {
        metrics.memory_used_bytes = static_cast<size_t>(mem.used);
        metrics.memory_total_bytes = static_cast<size_t>(mem.total);
    }

    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        metrics.temperature_celsius = temp;
    }

    unsigned int power = 0;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
        metrics.power_usage_milliwatts = power;
    }

    return metrics;
}

#else  // !TRT_ENGINE_HAS_NVML

unsigned int PerformanceProfiler::gpu_utilization(int /*device_id*/) {
    return 0;
}

size_t PerformanceProfiler::memory_used(int /*device_id*/) {
    return 0;
}

unsigned int PerformanceProfiler::temperature(int /*device_id*/) {
    return 0;
}

unsigned int PerformanceProfiler::power_usage(int /*device_id*/) {
    return 0;
}

GpuMetrics PerformanceProfiler::get_gpu_metrics(int /*device_id*/) {
    return GpuMetrics{};
}

#endif  // TRT_ENGINE_HAS_NVML

}  // namespace trt_engine
