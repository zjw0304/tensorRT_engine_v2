// multi_gpu.cpp
// Demonstrates using MultiGPUEngine for data-parallel inference
// across multiple GPUs with round-robin load balancing.

#include <trt_engine/trt_engine.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.engine>" << std::endl;
        return 1;
    }

    const std::string engine_path = argv[1];

    try {
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        // Enumerate available GPUs
        int gpu_count = trt_engine::get_device_count();
        std::cout << "Available GPUs: " << gpu_count << std::endl;

        std::vector<int> device_ids;
        for (int i = 0; i < gpu_count; ++i) {
            auto props = trt_engine::get_device_properties(i);
            std::cout << "  GPU " << i << ": " << props.name
                      << " (SM " << props.compute_capability_major
                      << "." << props.compute_capability_minor
                      << ", " << (props.total_global_memory / (1ULL << 30))
                      << " GB)" << std::endl;
            device_ids.push_back(i);
        }

        if (device_ids.empty()) {
            std::cerr << "No GPUs found." << std::endl;
            return 1;
        }

        // Create multi-GPU engine spanning all devices
        trt_engine::MultiGPUEngine engine(engine_path, device_ids);

        // Prepare dummy input
        auto single = trt_engine::InferenceEngine::create(engine_path);
        auto input_info = single->get_input_info();
        size_t input_elems = 1;
        for (int d : input_info[0].shape)
            input_elems *= (d > 0) ? d : 1;
        single.reset();

        std::vector<std::vector<float>> inputs = {
            std::vector<float>(input_elems, 0.5f)
        };

        // Run inference across GPUs (round-robin)
        const int num_requests = 20;
        std::cout << "\nRunning " << num_requests
                  << " requests across " << engine.get_device_count()
                  << " GPUs..." << std::endl;

        for (int i = 0; i < num_requests; ++i) {
            auto result = engine.infer(inputs);
            if (result.success) {
                std::cout << "  Request " << i
                          << ": latency=" << result.latency_ms << " ms"
                          << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
