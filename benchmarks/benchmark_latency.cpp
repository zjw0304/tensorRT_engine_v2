// benchmark_latency.cpp - Latency benchmark for TensorRT inference engine
//
// Measures per-inference latency and reports p50, p95, p99, min, max, mean.
// Tests both synchronous and asynchronous inference paths.

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

struct LatencyConfig {
    std::string engine_path;
    int batch_size = 1;
    int num_iterations = 200;
    int warmup_iterations = 20;
    std::string output_json;
};

struct LatencyResult {
    std::string mode;  // "sync" or "async"
    int batch_size;
    int num_iterations;
    double min_ms;
    double max_ms;
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --engine <path>       Path to serialized TRT engine (required)\n"
              << "  --batch-size <n>      Batch size (default: 1)\n"
              << "  --iterations <n>      Number of inference iterations (default: 200)\n"
              << "  --warmup <n>          Number of warmup iterations (default: 20)\n"
              << "  --output <path>       Output JSON file path\n";
}

static LatencyConfig parse_args(int argc, char** argv) {
    LatencyConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            cfg.engine_path = argv[++i];
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_json = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

static LatencyResult run_latency_test(
        trt_engine::InferenceEngine& engine,
        const std::vector<std::vector<float>>& dummy_inputs,
        int batch_size,
        int num_iterations,
        int warmup_iterations,
        bool async_mode) {

    trt_engine::PerformanceProfiler profiler;

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        if (async_mode) {
            auto f = engine.infer_async(dummy_inputs);
            f.get();
        } else {
            engine.infer(dummy_inputs);
        }
    }

    // Measure
    for (int i = 0; i < num_iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (async_mode) {
            auto f = engine.infer_async(dummy_inputs);
            auto result = f.get();
            if (!result.success) {
                std::cerr << "Async inference failed at iteration " << i
                          << ": " << result.error_msg << "\n";
                continue;
            }
        } else {
            auto result = engine.infer(dummy_inputs);
            if (!result.success) {
                std::cerr << "Sync inference failed at iteration " << i
                          << ": " << result.error_msg << "\n";
                continue;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        profiler.record_inference(latency_ms);
    }

    auto stats = profiler.get_statistics();

    LatencyResult res;
    res.mode = async_mode ? "async" : "sync";
    res.batch_size = batch_size;
    res.num_iterations = num_iterations;
    res.min_ms = stats.min_ms;
    res.max_ms = stats.max_ms;
    res.mean_ms = stats.mean_ms;
    res.p50_ms = stats.p50_ms;
    res.p95_ms = stats.p95_ms;
    res.p99_ms = stats.p99_ms;

    return res;
}

static void print_result(const LatencyResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mode:       " << r.mode << "\n";
    std::cout << "  Batch size: " << r.batch_size << "\n";
    std::cout << "  Iterations: " << r.num_iterations << "\n";
    std::cout << "  Min (ms):   " << r.min_ms << "\n";
    std::cout << "  Max (ms):   " << r.max_ms << "\n";
    std::cout << "  Mean (ms):  " << r.mean_ms << "\n";
    std::cout << "  P50 (ms):   " << r.p50_ms << "\n";
    std::cout << "  P95 (ms):   " << r.p95_ms << "\n";
    std::cout << "  P99 (ms):   " << r.p99_ms << "\n\n";
}

static std::string results_to_json(const std::vector<LatencyResult>& results,
                                    const LatencyConfig& cfg) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "{\n";
    oss << "  \"benchmark\": \"latency\",\n";
    oss << "  \"engine\": \"" << cfg.engine_path << "\",\n";
    oss << "  \"batch_size\": " << cfg.batch_size << ",\n";
    oss << "  \"iterations\": " << cfg.num_iterations << ",\n";
    oss << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "    {\n";
        oss << "      \"mode\": \"" << r.mode << "\",\n";
        oss << "      \"batch_size\": " << r.batch_size << ",\n";
        oss << "      \"min_ms\": " << r.min_ms << ",\n";
        oss << "      \"max_ms\": " << r.max_ms << ",\n";
        oss << "      \"mean_ms\": " << r.mean_ms << ",\n";
        oss << "      \"p50_ms\": " << r.p50_ms << ",\n";
        oss << "      \"p95_ms\": " << r.p95_ms << ",\n";
        oss << "      \"p99_ms\": " << r.p99_ms << "\n";
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

    std::cout << "=== Latency Benchmark ===\n";
    std::cout << "Engine: " << cfg.engine_path << "\n";
    std::cout << "Batch size: " << cfg.batch_size << "\n";
    std::cout << "Iterations: " << cfg.num_iterations << "\n";
    std::cout << "Warmup: " << cfg.warmup_iterations << "\n\n";

    auto engine = trt_engine::InferenceEngine::create(cfg.engine_path);

    // Build dummy inputs
    auto inputs_info = engine->get_input_info();
    std::vector<std::vector<float>> dummy_inputs;
    dummy_inputs.reserve(inputs_info.size());

    for (auto& ti : inputs_info) {
        int64_t vol = 1;
        for (size_t d = 0; d < ti.shape.size(); ++d) {
            if (d == 0) {
                vol *= cfg.batch_size;
                auto dims = ti.shape;
                if (!dims.empty()) dims[0] = cfg.batch_size;
                engine->set_input_shape(ti.name, dims);
            } else {
                vol *= (ti.shape[d] > 0) ? ti.shape[d] : 1;
            }
        }
        dummy_inputs.emplace_back(static_cast<size_t>(vol), 0.5f);
    }

    // Pre-allocate device/pinned buffers for the fast path
    engine->prepare_buffers();

    std::vector<LatencyResult> all_results;

    // Synchronous test
    std::cout << "--- Synchronous Inference ---\n";
    try {
        auto res = run_latency_test(*engine, dummy_inputs, cfg.batch_size,
                                     cfg.num_iterations, cfg.warmup_iterations,
                                     false);
        print_result(res);
        all_results.push_back(res);
    } catch (const std::exception& e) {
        std::cerr << "Sync test failed: " << e.what() << "\n";
    }

    // Asynchronous test
    std::cout << "--- Asynchronous Inference ---\n";
    try {
        auto res = run_latency_test(*engine, dummy_inputs, cfg.batch_size,
                                     cfg.num_iterations, cfg.warmup_iterations,
                                     true);
        print_result(res);
        all_results.push_back(res);
    } catch (const std::exception& e) {
        std::cerr << "Async test failed: " << e.what() << "\n";
    }

    // Write JSON output
    if (!cfg.output_json.empty()) {
        std::string json = results_to_json(all_results, cfg);
        std::ofstream out(cfg.output_json);
        if (out.is_open()) {
            out << json;
            std::cout << "Results written to " << cfg.output_json << "\n";
        } else {
            std::cerr << "Failed to write output file: " << cfg.output_json << "\n";
        }
    }

    return 0;
}
