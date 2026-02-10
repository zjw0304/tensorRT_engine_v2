// benchmark_throughput.cpp - Throughput benchmark for TensorRT inference engine
//
// Measures maximum inference throughput (inferences/sec, images/sec)
// across multiple batch sizes and with/without CUDA graphs.

#include <trt_engine/trt_engine.h>
#include <trt_engine/profiler.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct BenchmarkConfig {
    std::string engine_path;
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    std::string precision_str = "FP16";
    int num_iterations = 100;
    int warmup_iterations = 10;
    std::string output_json;
    bool test_cuda_graph = true;
};

struct ThroughputResult {
    int batch_size;
    double total_time_ms;
    int num_iterations;
    double throughput_ips;     // inferences per second
    double images_per_sec;     // batch_size * ips
    double mean_latency_ms;
    bool cuda_graph;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --engine <path>         Path to serialized TRT engine (required)\n"
              << "  --batch-sizes <list>    Comma-separated batch sizes (default: 1,2,4,8,16,32)\n"
              << "  --precision <str>       Precision label for reporting (default: FP16)\n"
              << "  --iterations <n>        Number of inference iterations (default: 100)\n"
              << "  --warmup <n>            Number of warmup iterations (default: 10)\n"
              << "  --output <path>         Output JSON file path\n"
              << "  --no-cuda-graph         Skip CUDA graph tests\n";
}

static std::vector<int> parse_batch_sizes(const std::string& s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        int val = std::stoi(token);
        if (val > 0) result.push_back(val);
    }
    return result;
}

static BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            cfg.engine_path = argv[++i];
        } else if (arg == "--batch-sizes" && i + 1 < argc) {
            cfg.batch_sizes = parse_batch_sizes(argv[++i]);
        } else if (arg == "--precision" && i + 1 < argc) {
            cfg.precision_str = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_json = argv[++i];
        } else if (arg == "--no-cuda-graph") {
            cfg.test_cuda_graph = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

static ThroughputResult run_throughput_test(
        const std::string& engine_path,
        int batch_size,
        int num_iterations,
        int warmup_iterations,
        bool use_cuda_graph) {

    trt_engine::EngineConfig config;
    config.enable_cuda_graph = use_cuda_graph;

    auto engine = trt_engine::InferenceEngine::create(engine_path, config);

    // Get input info and create dummy input
    auto inputs_info = engine->get_input_info();
    std::vector<std::vector<float>> dummy_inputs;
    dummy_inputs.reserve(inputs_info.size());

    for (auto& ti : inputs_info) {
        int64_t vol = 1;
        for (size_t d = 0; d < ti.shape.size(); ++d) {
            if (d == 0) {
                // Replace batch dim
                vol *= batch_size;
                engine->set_input_shape(ti.name,
                    [&]() {
                        auto dims = ti.shape;
                        if (!dims.empty()) dims[0] = batch_size;
                        return dims;
                    }());
            } else {
                vol *= (ti.shape[d] > 0) ? ti.shape[d] : 1;
            }
        }
        dummy_inputs.emplace_back(static_cast<size_t>(vol), 0.5f);
    }

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        engine->infer(dummy_inputs);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto result = engine->infer(dummy_inputs);
        if (!result.success) {
            std::cerr << "Inference failed at iteration " << i
                      << ": " << result.error_msg << "\n";
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    ThroughputResult res;
    res.batch_size = batch_size;
    res.total_time_ms = elapsed_ms;
    res.num_iterations = num_iterations;
    res.throughput_ips = (num_iterations / elapsed_ms) * 1000.0;
    res.images_per_sec = res.throughput_ips * batch_size;
    res.mean_latency_ms = elapsed_ms / num_iterations;
    res.cuda_graph = use_cuda_graph;

    return res;
}

static std::string results_to_json(const std::vector<ThroughputResult>& results,
                                    const BenchmarkConfig& cfg) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\n";
    oss << "  \"benchmark\": \"throughput\",\n";
    oss << "  \"engine\": \"" << cfg.engine_path << "\",\n";
    oss << "  \"precision\": \"" << cfg.precision_str << "\",\n";
    oss << "  \"iterations\": " << cfg.num_iterations << ",\n";
    oss << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "    {\n";
        oss << "      \"batch_size\": " << r.batch_size << ",\n";
        oss << "      \"cuda_graph\": " << (r.cuda_graph ? "true" : "false") << ",\n";
        oss << "      \"total_time_ms\": " << r.total_time_ms << ",\n";
        oss << "      \"throughput_ips\": " << r.throughput_ips << ",\n";
        oss << "      \"images_per_sec\": " << r.images_per_sec << ",\n";
        oss << "      \"mean_latency_ms\": " << r.mean_latency_ms << "\n";
        oss << "    }";
        if (i + 1 < results.size()) oss << ",";
        oss << "\n";
    }

    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

int main(int argc, char** argv) {
    auto cfg = parse_args(argc, argv);

    if (cfg.engine_path.empty()) {
        std::cerr << "Error: --engine is required\n";
        print_usage(argv[0]);
        return 1;
    }

    trt_engine::get_logger().set_severity(trt_engine::LogSeverity::WARNING);

    std::cout << "=== Throughput Benchmark ===\n";
    std::cout << "Engine: " << cfg.engine_path << "\n";
    std::cout << "Precision: " << cfg.precision_str << "\n";
    std::cout << "Iterations: " << cfg.num_iterations << "\n";
    std::cout << "Warmup: " << cfg.warmup_iterations << "\n\n";

    std::vector<ThroughputResult> all_results;

    // Header
    std::cout << std::setw(8) << "Batch"
              << std::setw(12) << "CUDA Graph"
              << std::setw(14) << "Throughput"
              << std::setw(16) << "Images/sec"
              << std::setw(14) << "Latency(ms)"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (int bs : cfg.batch_sizes) {
        // Without CUDA graph
        try {
            auto res = run_throughput_test(
                cfg.engine_path, bs, cfg.num_iterations,
                cfg.warmup_iterations, false);
            all_results.push_back(res);

            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(8) << bs
                      << std::setw(12) << "No"
                      << std::setw(14) << res.throughput_ips
                      << std::setw(16) << res.images_per_sec
                      << std::setw(14) << res.mean_latency_ms
                      << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error (batch=" << bs << ", no graph): " << e.what() << "\n";
        }

        // With CUDA graph
        if (cfg.test_cuda_graph) {
            try {
                auto res = run_throughput_test(
                    cfg.engine_path, bs, cfg.num_iterations,
                    cfg.warmup_iterations, true);
                all_results.push_back(res);

                std::cout << std::setw(8) << bs
                          << std::setw(12) << "Yes"
                          << std::setw(14) << res.throughput_ips
                          << std::setw(16) << res.images_per_sec
                          << std::setw(14) << res.mean_latency_ms
                          << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error (batch=" << bs << ", graph): " << e.what() << "\n";
            }
        }
    }

    // Write JSON output
    if (!cfg.output_json.empty()) {
        std::string json = results_to_json(all_results, cfg);
        std::ofstream out(cfg.output_json);
        if (out.is_open()) {
            out << json;
            std::cout << "\nResults written to " << cfg.output_json << "\n";
        } else {
            std::cerr << "Failed to write output file: " << cfg.output_json << "\n";
        }
    }

    return 0;
}
