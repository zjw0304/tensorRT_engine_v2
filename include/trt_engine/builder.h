#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>
#include <trt_engine/types.h>

#include <string>
#include <vector>

namespace trt_engine {

// ── Engine builder ──────────────────────────────────────────────────────────
//
// Builds TensorRT engines from ONNX models.  Supports FP32, FP16, and INT8
// precision, dynamic shapes via optimization profiles, timing-cache
// persistence, DLA offloading, and auxiliary-stream configuration.
//
class EngineBuilder {
public:
    explicit EngineBuilder(Logger& logger);
    ~EngineBuilder() = default;

    // Non-copyable, non-movable
    EngineBuilder(const EngineBuilder&) = delete;
    EngineBuilder& operator=(const EngineBuilder&) = delete;

    // ── Build ────────────────────────────────────────────────────────────
    // Build a TRT engine from an ONNX model and return the serialized
    // engine plan as a byte vector.
    std::vector<char> build_engine(const std::string& onnx_path,
                                   const BuilderConfig& config);

    // ── Serialization ────────────────────────────────────────────────────
    // Save a serialized engine plan to disk.
    static bool save_engine(const std::vector<char>& engine_data,
                            const std::string& path);

    // Load a serialized engine plan from disk.
    static std::vector<char> load_engine(const std::string& path);

    // ── INT8 calibrator ──────────────────────────────────────────────────
    // Set an external calibrator for INT8 quantization.
    void set_calibrator(nvinfer1::IInt8Calibrator* calibrator);

private:
    // Internal build implementation.
    std::vector<char> build_engine_from_onnx(const std::string& onnx_path,
                                             const BuilderConfig& config);

    // Configure precision flags on the builder config.
    void configure_precision(nvinfer1::IBuilderConfig* trt_config,
                             const BuilderConfig& config);

    // Add optimization profiles for dynamic shapes.
    void configure_dynamic_shapes(nvinfer1::IBuilder* builder,
                                  nvinfer1::IBuilderConfig* trt_config,
                                  nvinfer1::INetworkDefinition* network,
                                  const BuilderConfig& config);

    // Load / save timing cache.
    std::vector<char> load_timing_cache(const std::string& path);
    void save_timing_cache(nvinfer1::IBuilderConfig* trt_config,
                           const std::string& path);

    Logger&                     logger_;
    nvinfer1::IInt8Calibrator*  calibrator_ = nullptr;
};

}  // namespace trt_engine
