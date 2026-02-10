// multi_stream.cpp
// Demonstrates using MultiStreamEngine for concurrent inference
// across multiple CUDA streams.

#include <trt_engine/trt_engine.h>

#include <chrono>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.engine>" << std::endl;
        return 1;
    }

    const std::string engine_path = argv[1];
    const int num_streams = 4;
    const int num_requests = 50;

    try {
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        // Create a multi-stream engine with 4 parallel workers
        std::cout << "Creating MultiStreamEngine with "
                  << num_streams << " streams..." << std::endl;

        trt_engine::MultiStreamEngine engine(engine_path, num_streams);

        // Determine input size from single-stream engine
        auto single = trt_engine::InferenceEngine::create(engine_path);
        auto input_info = single->get_input_info();
        size_t input_elems = 1;
        for (int d : input_info[0].shape) {
            input_elems *= (d > 0) ? d : 1;
        }
        single.reset(); // Free the single engine

        // Prepare input data
        std::vector<std::vector<float>> input_data = {
            std::vector<float>(input_elems, 0.5f)
        };

        // Submit all requests concurrently
        std::cout << "Submitting " << num_requests
                  << " concurrent requests..." << std::endl;

        auto wall_start = std::chrono::steady_clock::now();

        std::vector<std::future<trt_engine::InferenceResult>> futures;
        futures.reserve(num_requests);

        for (int i = 0; i < num_requests; ++i) {
            futures.push_back(engine.submit(input_data));
        }

        // Collect results
        int success_count = 0;
        float total_latency = 0.0f;

        for (auto& f : futures) {
            auto result = f.get();
            if (result.success) {
                ++success_count;
                total_latency += result.latency_ms;
            }
        }

        auto wall_end = std::chrono::steady_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(
                             wall_end - wall_start).count();

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Successful: " << success_count
                  << "/" << num_requests << std::endl;
        std::cout << "Wall time:  " << wall_ms << " ms" << std::endl;
        std::cout << "Avg GPU latency: " << total_latency / success_count
                  << " ms" << std::endl;
        std::cout << "Throughput: "
                  << (success_count * 1000.0 / wall_ms)
                  << " infer/sec" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
