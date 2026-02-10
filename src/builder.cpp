#include <trt_engine/builder.h>

#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace trt_engine {

// ── Constructor ─────────────────────────────────────────────────────────────

EngineBuilder::EngineBuilder(Logger& logger)
    : logger_(logger) {}

// ── Public: build_engine ────────────────────────────────────────────────────

std::vector<char> EngineBuilder::build_engine(const std::string& onnx_path,
                                               const BuilderConfig& config) {
    return build_engine_from_onnx(onnx_path, config);
}

// ── Public: set_calibrator ──────────────────────────────────────────────────

void EngineBuilder::set_calibrator(nvinfer1::IInt8Calibrator* calibrator) {
    calibrator_ = calibrator;
}

// ── Public: save_engine ─────────────────────────────────────────────────────

bool EngineBuilder::save_engine(const std::vector<char>& engine_data,
                                 const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        get_logger().error("Failed to open output file for engine: " + path);
        return false;
    }
    file.write(engine_data.data(),
               static_cast<std::streamsize>(engine_data.size()));
    if (!file.good()) {
        get_logger().error("Failed to write engine data to file: " + path);
        return false;
    }
    get_logger().info("Engine saved to " + path + " (" +
                      std::to_string(engine_data.size()) + " bytes)");
    return true;
}

// ── Public: load_engine ─────────────────────────────────────────────────────

std::vector<char> EngineBuilder::load_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        get_logger().error("Failed to open engine file: " + path);
        return {};
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        get_logger().error("Engine file is empty or unreadable: " + path);
        return {};
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    if (!file.read(data.data(), size)) {
        get_logger().error("Failed to read engine file: " + path);
        return {};
    }

    get_logger().info("Engine loaded from " + path + " (" +
                      std::to_string(size) + " bytes)");
    return data;
}

// ── Internal: build_engine_from_onnx ────────────────────────────────────────

std::vector<char> EngineBuilder::build_engine_from_onnx(
        const std::string& onnx_path, const BuilderConfig& config) {

    auto t0 = std::chrono::steady_clock::now();

    // --- 1. verify the ONNX file exists ---
    if (!fs::exists(onnx_path)) {
        logger_.error("ONNX file not found: " + onnx_path);
        return {};
    }
    logger_.info("Building TensorRT engine from: " + onnx_path);
    logger_.info("Precision: " + precision_to_string(config.precision));

    // --- 2. create builder ---
    UniqueBuilder builder(nvinfer1::createInferBuilder(logger_));
    TRT_CHECK(builder != nullptr);
    logger_.info("Created TensorRT builder");

    // --- 3. create network (explicit batch) ---
    uint32_t network_flags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if (config.strongly_typed) {
        network_flags |= 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    }
    UniqueNetwork network(builder->createNetworkV2(network_flags));
    TRT_CHECK(network != nullptr);
    logger_.info("Created network definition (explicit batch" +
                 std::string(config.strongly_typed ? ", strongly typed" : "") + ")");

    // --- 4. create ONNX parser and parse ---
    UniqueParser parser(
        nvonnxparser::createParser(*network, logger_));
    TRT_CHECK(parser != nullptr);

    bool parsed = parser->parseFromFile(
        onnx_path.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if (!parsed) {
        // Log parser errors
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            logger_.error(std::string("ONNX parse error: ") +
                          parser->getError(i)->desc());
        }
        logger_.error("Failed to parse ONNX file: " + onnx_path);
        return {};
    }
    logger_.info("ONNX model parsed successfully (" +
                 std::to_string(network->getNbLayers()) + " layers, " +
                 std::to_string(network->getNbInputs()) + " inputs, " +
                 std::to_string(network->getNbOutputs()) + " outputs)");

    // --- 5. create builder config ---
    std::unique_ptr<nvinfer1::IBuilderConfig, TRTDeleter> trt_config(
        builder->createBuilderConfig());
    TRT_CHECK(trt_config != nullptr);

    // Workspace
    trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   config.max_workspace_size);
    logger_.info("Max workspace size: " +
                 std::to_string(config.max_workspace_size / (1ULL << 20)) + " MB");

    // --- 6. builder optimization level ---
    trt_config->setBuilderOptimizationLevel(config.builder_optimization_level);
    logger_.info("Builder optimization level: " +
                 std::to_string(config.builder_optimization_level));

    // --- 7. precision ---
    configure_precision(trt_config.get(), config);

    // --- 8. DLA ---
    if (config.enable_dla) {
        if (builder->getNbDLACores() > 0) {
            trt_config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            trt_config->setDLACore(config.dla_core);
            trt_config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
            logger_.info("DLA enabled on core " + std::to_string(config.dla_core));
        } else {
            logger_.warning("DLA requested but no DLA cores available. "
                            "Falling back to GPU.");
        }
    }

    // --- 9. auxiliary streams ---
    if (config.max_aux_streams > 0) {
        trt_config->setMaxAuxStreams(config.max_aux_streams);
        logger_.info("Max auxiliary streams: " +
                     std::to_string(config.max_aux_streams));
    }

    // --- 10. dynamic shapes / optimization profiles ---
    configure_dynamic_shapes(builder.get(), trt_config.get(), network.get(), config);

    // --- 11. timing cache ---
    std::string effective_timing_cache_path = config.timing_cache_path;
    if (effective_timing_cache_path.empty() && config.auto_timing_cache) {
        effective_timing_cache_path =
            (fs::path(onnx_path).parent_path() / "timing_cache.bin").string();
        logger_.info("Auto timing cache path: " + effective_timing_cache_path);
    }

    std::vector<char> timing_cache_data;
    if (!effective_timing_cache_path.empty()) {
        timing_cache_data = load_timing_cache(effective_timing_cache_path);
    }
    nvinfer1::ITimingCache* timing_cache = trt_config->createTimingCache(
        timing_cache_data.data(), timing_cache_data.size());
    if (timing_cache) {
        trt_config->setTimingCache(*timing_cache, false);
        logger_.info("Timing cache " +
                     std::string(timing_cache_data.empty() ? "created" : "loaded"));
    }

    // --- 12. build the serialized network ---
    logger_.info("Starting engine build ...");
    std::unique_ptr<nvinfer1::IHostMemory, TRTDeleter> serialized(
        builder->buildSerializedNetwork(*network, *trt_config));
    if (!serialized || serialized->size() == 0) {
        logger_.error("Engine build failed (buildSerializedNetwork returned "
                      "null or empty)");
        return {};
    }

    // --- 13. copy to vector ---
    std::vector<char> engine_data(
        static_cast<const char*>(serialized->data()),
        static_cast<const char*>(serialized->data()) + serialized->size());

    // --- 14. save timing cache ---
    if (!effective_timing_cache_path.empty()) {
        save_timing_cache(trt_config.get(), effective_timing_cache_path);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    logger_.info("Engine built successfully in " +
                 std::to_string(elapsed_ms) + " ms (" +
                 std::to_string(engine_data.size()) + " bytes)");

    return engine_data;
}

// ── Internal: configure_precision ───────────────────────────────────────────

void EngineBuilder::configure_precision(nvinfer1::IBuilderConfig* trt_config,
                                         const BuilderConfig& config) {
    switch (config.precision) {
        case Precision::FP16:
            trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            logger_.info("FP16 precision enabled");
            break;

        case Precision::INT8:
            trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
            // Also enable FP16 as a fallback for layers without INT8 support.
            trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            if (calibrator_) {
                trt_config->setInt8Calibrator(calibrator_);
                logger_.info("INT8 precision enabled with external calibrator");
            } else {
                logger_.warning("INT8 precision enabled but no calibrator set. "
                                "The ONNX model must contain Q/DQ nodes.");
            }
            break;

        case Precision::FP8:
            trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
            logger_.info("FP8 precision requested; enabling FP16 flag. "
                         "The ONNX model should contain FP8 Q/DQ nodes.");
            break;

        case Precision::FP32:
        default:
            logger_.info("FP32 (default) precision");
            break;
    }
}

// ── Internal: configure_dynamic_shapes ──────────────────────────────────────

void EngineBuilder::configure_dynamic_shapes(
        nvinfer1::IBuilder* builder,
        nvinfer1::IBuilderConfig* trt_config,
        nvinfer1::INetworkDefinition* network,
        const BuilderConfig& config) {

    if (config.dynamic_shapes.empty()) {
        return;
    }

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    TRT_CHECK(profile != nullptr);

    for (const auto& shape : config.dynamic_shapes) {
        // Validate that the input exists in the network.
        bool found = false;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            if (network->getInput(i)->getName() == shape.name) {
                found = true;
                break;
            }
        }
        if (!found) {
            logger_.warning("Dynamic shape profile references unknown input: " +
                            shape.name + " (skipping)");
            continue;
        }

        // Convert std::vector<int> to nvinfer1::Dims.
        auto to_dims = [](const std::vector<int>& v) -> nvinfer1::Dims {
            nvinfer1::Dims dims{};
            dims.nbDims = static_cast<int>(v.size());
            for (int i = 0; i < dims.nbDims; ++i) {
                dims.d[i] = v[static_cast<size_t>(i)];
            }
            return dims;
        };

        nvinfer1::Dims min_d = to_dims(shape.min_dims);
        nvinfer1::Dims opt_d = to_dims(shape.opt_dims);
        nvinfer1::Dims max_d = to_dims(shape.max_dims);

        profile->setDimensions(shape.name.c_str(),
                               nvinfer1::OptProfileSelector::kMIN, min_d);
        profile->setDimensions(shape.name.c_str(),
                               nvinfer1::OptProfileSelector::kOPT, opt_d);
        profile->setDimensions(shape.name.c_str(),
                               nvinfer1::OptProfileSelector::kMAX, max_d);

        logger_.info("Optimization profile for '" + shape.name + "': "
                     "min=[" + [&]() {
                         std::string s;
                         for (size_t i = 0; i < shape.min_dims.size(); ++i) {
                             if (i) s += ",";
                             s += std::to_string(shape.min_dims[i]);
                         }
                         return s;
                     }() + "] opt=[" + [&]() {
                         std::string s;
                         for (size_t i = 0; i < shape.opt_dims.size(); ++i) {
                             if (i) s += ",";
                             s += std::to_string(shape.opt_dims[i]);
                         }
                         return s;
                     }() + "] max=[" + [&]() {
                         std::string s;
                         for (size_t i = 0; i < shape.max_dims.size(); ++i) {
                             if (i) s += ",";
                             s += std::to_string(shape.max_dims[i]);
                         }
                         return s;
                     }() + "]");
    }

    trt_config->addOptimizationProfile(profile);
    logger_.info("Added " + std::to_string(config.dynamic_shapes.size()) +
                 " dynamic shape profile(s)");
}

// ── Internal: timing cache I/O ──────────────────────────────────────────────

std::vector<char> EngineBuilder::load_timing_cache(const std::string& path) {
    if (!fs::exists(path)) {
        logger_.info("Timing cache file not found, creating new: " + path);
        return {};
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        logger_.warning("Failed to open timing cache file: " + path);
        return {};
    }
    auto size = file.tellg();
    if (size <= 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<char> data(static_cast<size_t>(size));
    file.read(data.data(), size);
    logger_.info("Timing cache loaded from " + path + " (" +
                 std::to_string(size) + " bytes)");
    return data;
}

void EngineBuilder::save_timing_cache(nvinfer1::IBuilderConfig* trt_config,
                                       const std::string& path) {
    const nvinfer1::ITimingCache* cache = trt_config->getTimingCache();
    if (!cache) {
        logger_.warning("No timing cache to save");
        return;
    }
    std::unique_ptr<nvinfer1::IHostMemory, TRTDeleter> serialized(
        cache->serialize());
    if (!serialized || serialized->size() == 0) {
        logger_.warning("Failed to serialize timing cache");
        return;
    }
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        logger_.warning("Failed to open timing cache output file: " + path);
        return;
    }
    file.write(static_cast<const char*>(serialized->data()),
               static_cast<std::streamsize>(serialized->size()));
    logger_.info("Timing cache saved to " + path + " (" +
                 std::to_string(serialized->size()) + " bytes)");
}

}  // namespace trt_engine
