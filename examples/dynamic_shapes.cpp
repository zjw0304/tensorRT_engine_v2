// dynamic_shapes.cpp
// Demonstrates configuring optimization profiles for dynamic input shapes
// and running inference at different input sizes.

#include <trt_engine/trt_engine.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    const std::string onnx_path = argv[1];
    const std::string engine_path = "model_dynamic.engine";

    try {
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        trt_engine::EngineBuilder builder(logger);

        // Configure dynamic shapes with optimization profiles.
        // min/opt/max define the range of valid input dimensions.
        trt_engine::BuilderConfig build_config;
        build_config.precision = trt_engine::Precision::FP16;
        build_config.max_workspace_size = 1ULL << 30;
        build_config.dynamic_shapes = {{
            "input",                   // tensor name
            {1, 3, 224, 224},          // min dimensions
            {4, 3, 224, 224},          // opt dimensions (most common)
            {16, 3, 224, 224}          // max dimensions
        }};

        std::cout << "Building engine with dynamic shapes..." << std::endl;
        auto engine_data = builder.build_engine(onnx_path, build_config);
        if (engine_data.empty()) {
            std::cerr << "Engine build failed." << std::endl;
            return 1;
        }
        trt_engine::EngineBuilder::save_engine(engine_data, engine_path);

        // Create the inference engine
        auto engine = trt_engine::InferenceEngine::create(engine_path);
        engine->warmup(3);

        // Run inference at different batch sizes
        std::vector<int> batch_sizes = {1, 4, 8, 16};
        const int C = 3, H = 224, W = 224;

        for (int batch : batch_sizes) {
            // Set the input shape for this batch size
            engine->set_input_shape("input", {batch, C, H, W});

            // Create input data for this batch
            size_t num_elements = static_cast<size_t>(batch * C * H * W);
            std::vector<std::vector<float>> inputs = {
                std::vector<float>(num_elements, 0.5f)
            };

            auto result = engine->infer(inputs);

            if (result.success) {
                std::cout << "Batch " << batch
                          << ": latency=" << result.latency_ms << " ms"
                          << ", output_elements=" << result.outputs[0].size()
                          << std::endl;
            } else {
                std::cerr << "Batch " << batch
                          << ": failed - " << result.error_msg << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
