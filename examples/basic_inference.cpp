// basic_inference.cpp
// Demonstrates loading an ONNX model, building a TensorRT engine,
// and running inference using the trt_engine library.

#include <trt_engine/trt_engine.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    const std::string onnx_path = argv[1];
    const std::string engine_path = "model.engine";

    try {
        // Step 1: Configure the logger
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        // Step 2: Build the TensorRT engine from an ONNX model
        trt_engine::EngineBuilder builder(logger);

        trt_engine::BuilderConfig build_config;
        build_config.precision = trt_engine::Precision::FP16;
        build_config.max_workspace_size = 1ULL << 30; // 1 GB

        std::cout << "Building engine from: " << onnx_path << std::endl;
        auto engine_data = builder.build_engine(onnx_path, build_config);
        if (engine_data.empty()) {
            std::cerr << "Engine build failed." << std::endl;
            return 1;
        }

        // Step 3: Save the engine for future reuse
        trt_engine::EngineBuilder::save_engine(engine_data, engine_path);

        // Step 4: Create the inference engine
        auto engine = trt_engine::InferenceEngine::create(engine_path);

        // Step 5: Query input/output tensor information
        auto input_info = engine->get_input_info();
        auto output_info = engine->get_output_info();

        std::cout << "\nInputs:" << std::endl;
        for (const auto& ti : input_info) {
            std::cout << "  " << ti.name << " shape=[";
            for (size_t i = 0; i < ti.shape.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << ti.shape[i];
            }
            std::cout << "] " << trt_engine::precision_to_string(ti.dtype) << std::endl;
        }

        std::cout << "Outputs:" << std::endl;
        for (const auto& ti : output_info) {
            std::cout << "  " << ti.name << " shape=[";
            for (size_t i = 0; i < ti.shape.size(); ++i) {
                if (i > 0) std::cout << ",";
                std::cout << ti.shape[i];
            }
            std::cout << "] " << trt_engine::precision_to_string(ti.dtype) << std::endl;
        }

        // Step 6: Warm up the engine
        engine->warmup(5);

        // Step 7: Prepare dummy input data
        size_t input_elems = 1;
        for (int d : input_info[0].shape) {
            input_elems *= (d > 0) ? d : 1;
        }

        std::vector<std::vector<float>> inputs = {
            std::vector<float>(input_elems, 0.5f)
        };

        // Step 8: Run inference
        auto result = engine->infer(inputs);

        if (result.success) {
            std::cout << "\nInference succeeded!" << std::endl;
            std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
            std::cout << "Output elements: " << result.outputs[0].size() << std::endl;

            // Print first 10 output values
            std::cout << "First 10 values: ";
            for (size_t i = 0; i < std::min<size_t>(10, result.outputs[0].size()); ++i) {
                std::cout << result.outputs[0][i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Inference failed: " << result.error_msg << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
