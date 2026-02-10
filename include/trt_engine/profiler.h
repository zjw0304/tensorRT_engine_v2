#pragma once

#include <NvInfer.h>
#include <trt_engine/types.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace trt_engine {

// ── Performance statistics ──────────────────────────────────────────────────
struct PerformanceStats {
    double min_ms           = 0.0;
    double max_ms           = 0.0;
    double mean_ms          = 0.0;
    double p50_ms           = 0.0;
    double p95_ms           = 0.0;
    double p99_ms           = 0.0;
    double throughput_fps   = 0.0;
    size_t total_inferences = 0;
};

// ── GPU metrics ─────────────────────────────────────────────────────────────
struct GpuMetrics {
    unsigned int gpu_utilization_percent = 0;
    size_t       memory_used_bytes      = 0;
    size_t       memory_total_bytes     = 0;
    unsigned int temperature_celsius    = 0;
    unsigned int power_usage_milliwatts = 0;
};

// ── TRT layer-level profiler ────────────────────────────────────────────────
// Implements nvinfer1::IProfiler to collect per-layer timing from TensorRT.
class TRTProfiler : public nvinfer1::IProfiler {
public:
    TRTProfiler() = default;
    ~TRTProfiler() override = default;

    // nvinfer1::IProfiler callback
    void reportLayerTime(const char* layer_name, float ms) noexcept override;

    // Retrieve accumulated layer timings (layer_name -> total_ms)
    std::map<std::string, float> get_layer_timings() const;

    // Get a sorted report (slowest layers first)
    std::string report_text() const;

    // Reset all layer data
    void reset();

private:
    mutable std::mutex mutex_;

    struct LayerStats {
        float  total_ms = 0.0f;
        int    call_count = 0;
    };
    std::map<std::string, LayerStats> layers_;
};

// ── High-level performance profiler ─────────────────────────────────────────
class PerformanceProfiler {
public:
    PerformanceProfiler() = default;
    ~PerformanceProfiler() = default;

    // Record a single inference latency measurement
    void record_inference(double latency_ms);

    // Compute statistics from all recorded measurements
    PerformanceStats get_statistics() const;

    // Clear all recorded data
    void reset();

    // Generate a formatted text report
    std::string report_text() const;

    // Generate a JSON string report
    std::string report_json() const;

    // Number of recorded inferences
    size_t count() const;

    // ── GPU metrics via NVML ────────────────────────────────────────────
    // These require NVML to be available at build time (TRT_ENGINE_HAS_NVML).
    // If NVML is not available, they return zeroed-out metrics.
    static unsigned int gpu_utilization(int device_id = 0);
    static size_t       memory_used(int device_id = 0);
    static unsigned int temperature(int device_id = 0);
    static unsigned int power_usage(int device_id = 0);
    static GpuMetrics   get_gpu_metrics(int device_id = 0);

private:
    static double percentile(const std::vector<double>& sorted_data, double p);

    mutable std::mutex   mutex_;
    std::vector<double>  latencies_;
};

}  // namespace trt_engine
