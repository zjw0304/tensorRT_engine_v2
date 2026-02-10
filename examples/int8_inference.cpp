// int8_inference.cpp
// Demonstrates INT8 quantized inference using the EntropyCalibratorV2
// with a calibration data directory.

#include <trt_engine/trt_engine.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <calibration_data_dir>" << std::endl;
        std::cerr << "  calibration_data_dir should contain .bin or .raw files"
                  << std::endl;
        return 1;
    }

    const std::string onnx_path = argv[1];
    const std::string calib_dir = argv[2];
    const std::string engine_fp32_path = "model_fp32.engine";
    const std::string engine_int8_path = "model_int8.engine";

    try {
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        trt_engine::EngineBuilder builder(logger);

        // Step 1: Build FP32 engine for comparison
        std::cout << "Building FP32 engine..." << std::endl;
        trt_engine::BuilderConfig fp32_config;
        fp32_config.precision = trt_engine::Precision::FP32;
        auto fp32_data = builder.build_engine(onnx_path, fp32_config);
        trt_engine::EngineBuilder::save_engine(fp32_data, engine_fp32_path);

        // Step 2: Set up INT8 calibrator
        // Input dimensions for a single sample (e.g., 3x224x224 for ResNet)
        std::vector<int> input_dims = {3, 224, 224};

        trt_engine::EntropyCalibratorV2 calibrator(
            calib_dir,               // directory with calibration .bin files
            32,                      // calibration batch size
            "input",                 // input tensor name
            input_dims,              // per-sample dimensions
            "calibration.cache"      // cache file for reuse
        );

        // Step 3: Build INT8 engine
        std::cout << "Building INT8 engine..." << std::endl;
        builder.set_calibrator(&calibrator);

        trt_engine::BuilderConfig int8_config;
        int8_config.precision = trt_engine::Precision::INT8;
        auto int8_data = builder.build_engine(onnx_path, int8_config);
        trt_engine::EngineBuilder::save_engine(int8_data, engine_int8_path);

        // Step 4: Compare FP32 and INT8 inference
        auto engine_fp32 = trt_engine::InferenceEngine::create(engine_fp32_path);
        auto engine_int8 = trt_engine::InferenceEngine::create(engine_int8_path);

        engine_fp32->warmup(5);
        engine_int8->warmup(5);

        // Prepare input data
        size_t input_size = 1;
        for (int d : input_dims) input_size *= d;
        std::vector<std::vector<float>> inputs = {
            std::vector<float>(input_size, 0.5f)
        };

        // Run inference on both
        auto result_fp32 = engine_fp32->infer(inputs);
        auto result_int8 = engine_int8->infer(inputs);

        if (result_fp32.success && result_int8.success) {
            std::cout << "\n=== Comparison ===" << std::endl;
            std::cout << "FP32 latency: " << result_fp32.latency_ms << " ms"
                      << std::endl;
            std::cout << "INT8 latency: " << result_int8.latency_ms << " ms"
                      << std::endl;

            float speedup = result_fp32.latency_ms / result_int8.latency_ms;
            std::cout << "Speedup:      " << speedup << "x" << std::endl;

            // Compare first few output values
            size_t n = std::min<size_t>(5, result_fp32.outputs[0].size());
            std::cout << "\nFirst " << n << " output values:" << std::endl;
            std::cout << "FP32: ";
            for (size_t i = 0; i < n; ++i)
                std::cout << result_fp32.outputs[0][i] << " ";
            std::cout << std::endl;
            std::cout << "INT8: ";
            for (size_t i = 0; i < n; ++i)
                std::cout << result_int8.outputs[0][i] << " ";
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
