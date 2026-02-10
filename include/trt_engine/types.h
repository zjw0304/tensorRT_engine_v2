#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_engine {

// ── Precision modes ────────────────────────────────────────────────────────
enum class Precision {
    FP32,
    FP16,
    INT8,
    FP8
};

// ── Log severity levels (mirrors TensorRT) ─────────────────────────────────
enum class LogSeverity {
    INTERNAL_ERROR = 0,
    ERROR          = 1,
    WARNING        = 2,
    INFO           = 3,
    VERBOSE        = 4
};

// ── Dynamic shape profile ──────────────────────────────────────────────────
struct DynamicShapeProfile {
    std::string      name;
    std::vector<int> min_dims;
    std::vector<int> opt_dims;
    std::vector<int> max_dims;
};

// ── Builder configuration ──────────────────────────────────────────────────
struct BuilderConfig {
    Precision   precision           = Precision::FP32;
    size_t      max_workspace_size  = 1ULL << 30;  // 1 GB
    bool        enable_cuda_graph   = false;
    bool        enable_dla          = false;
    int         dla_core            = 0;
    std::string timing_cache_path;
    int         max_aux_streams     = 0;
    bool        strongly_typed      = false;

    std::vector<DynamicShapeProfile> dynamic_shapes;
};

// ── Device configuration ───────────────────────────────────────────────────
struct DeviceConfig {
    int    device_id     = 0;
    size_t workspace_size = 1ULL << 30;  // 1 GB
};

// ── Inference result ───────────────────────────────────────────────────────
struct InferenceResult {
    std::vector<std::vector<float>> outputs;
    float       latency_ms = 0.0f;
    bool        success    = false;
    std::string error_msg;
};

// ── Tensor info ────────────────────────────────────────────────────────────
struct TensorInfo {
    std::string      name;
    std::vector<int> shape;
    Precision        dtype;
    size_t           size_bytes = 0;
};

// ── TensorRT object deleter for use with unique_ptr ────────────────────────
struct TRTDeleter {
    template <typename T>
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

// Convenience type aliases for RAII ownership of TensorRT objects
using UniqueRuntime  = std::unique_ptr<nvinfer1::IRuntime,          TRTDeleter>;
using UniqueEngine   = std::unique_ptr<nvinfer1::ICudaEngine,       TRTDeleter>;
using UniqueContext  = std::unique_ptr<nvinfer1::IExecutionContext,  TRTDeleter>;
using UniqueBuilder  = std::unique_ptr<nvinfer1::IBuilder,           TRTDeleter>;
using UniqueNetwork  = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDeleter>;
using UniqueParser   = std::unique_ptr<nvonnxparser::IParser,        TRTDeleter>;
// Note: IOptimizationProfile is owned by the IBuilder and must not be deleted
// independently.  Do not wrap it in a unique_ptr with TRTDeleter.

// ── Device properties ─────────────────────────────────────────────────────
struct DeviceProperties {
    std::string name;
    int         compute_capability_major = 0;
    int         compute_capability_minor = 0;
    size_t      total_global_memory      = 0;
    int         multi_processor_count    = 0;
    int         max_threads_per_block    = 0;
    size_t      shared_memory_per_block  = 0;
    int         warp_size                = 0;
    int         clock_rate_khz           = 0;
    int         memory_clock_rate_khz    = 0;
    int         memory_bus_width_bits    = 0;
};

// ── Utility functions ──────────────────────────────────────────────────────

inline std::string precision_to_string(Precision p) {
    switch (p) {
        case Precision::FP32: return "FP32";
        case Precision::FP16: return "FP16";
        case Precision::INT8: return "INT8";
        case Precision::FP8:  return "FP8";
    }
    return "UNKNOWN";
}

inline Precision string_to_precision(const std::string& s) {
    if (s == "FP32" || s == "fp32") return Precision::FP32;
    if (s == "FP16" || s == "fp16") return Precision::FP16;
    if (s == "INT8" || s == "int8") return Precision::INT8;
    if (s == "FP8"  || s == "fp8")  return Precision::FP8;
    throw std::invalid_argument("Unknown precision string: " + s);
}

inline nvinfer1::DataType precision_to_trt(Precision p) {
    switch (p) {
        case Precision::FP32: return nvinfer1::DataType::kFLOAT;
        case Precision::FP16: return nvinfer1::DataType::kHALF;
        case Precision::INT8: return nvinfer1::DataType::kINT8;
        case Precision::FP8:  return nvinfer1::DataType::kFP8;
    }
    return nvinfer1::DataType::kFLOAT;
}

inline size_t datatype_size(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kFP8:   return 1;
        default: return 0;
    }
}

inline std::string severity_to_string(LogSeverity s) {
    switch (s) {
        case LogSeverity::INTERNAL_ERROR: return "INTERNAL_ERROR";
        case LogSeverity::ERROR:          return "ERROR";
        case LogSeverity::WARNING:        return "WARNING";
        case LogSeverity::INFO:           return "INFO";
        case LogSeverity::VERBOSE:        return "VERBOSE";
    }
    return "UNKNOWN";
}

inline LogSeverity trt_severity_to_log(nvinfer1::ILogger::Severity s) {
    switch (s) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return LogSeverity::INTERNAL_ERROR;
        case nvinfer1::ILogger::Severity::kERROR:          return LogSeverity::ERROR;
        case nvinfer1::ILogger::Severity::kWARNING:        return LogSeverity::WARNING;
        case nvinfer1::ILogger::Severity::kINFO:           return LogSeverity::INFO;
        case nvinfer1::ILogger::Severity::kVERBOSE:        return LogSeverity::VERBOSE;
    }
    return LogSeverity::INFO;
}

inline nvinfer1::ILogger::Severity log_to_trt_severity(LogSeverity s) {
    switch (s) {
        case LogSeverity::INTERNAL_ERROR: return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
        case LogSeverity::ERROR:          return nvinfer1::ILogger::Severity::kERROR;
        case LogSeverity::WARNING:        return nvinfer1::ILogger::Severity::kWARNING;
        case LogSeverity::INFO:           return nvinfer1::ILogger::Severity::kINFO;
        case LogSeverity::VERBOSE:        return nvinfer1::ILogger::Severity::kVERBOSE;
    }
    return nvinfer1::ILogger::Severity::kINFO;
}

}  // namespace trt_engine
