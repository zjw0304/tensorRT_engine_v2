// benchmark_nlp.cpp - NLP model benchmark for TensorRT inference engine
//
// Measures latency and throughput for NLP models (BERT, DistilBERT, GPT-2,
// T5-small) across configurable batch sizes and sequence lengths.

#include <trt_engine/trt_engine.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Configuration ───────────────────────────────────────────────────────────

static const std::vector<std::string> ALL_NLP_MODELS =
    {"bert-base", "distilbert", "gpt2", "t5-small"};

struct NLPBenchmarkConfig {
    std::string model_name;
    std::string models_dir = "models";
    std::string precision_str = "FP16";
    std::vector<int> batch_sizes = {1, 4, 8};
    std::vector<int> seq_lengths = {64, 128};
    int num_iterations = 100;
    int warmup_iterations = 20;
    std::string output_json;
    bool use_cuda_graph = false;
    bool compare_precision = false;
};

struct NLPBenchmarkResult {
    std::string model;
    int batch_size;
    int seq_length;
    std::string precision;
    bool cuda_graph;
    double mean_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;
    double throughput_ips;   // inferences per second
    double tokens_per_sec;   // batch * seq_len * ips
    int num_iterations;
};

// ── Model metadata ──────────────────────────────────────────────────────────

struct NLPModelInfo {
    std::string name;
    int vocab_size;
    bool has_token_type_ids;
    bool batch_dynamic;       // false = model only supports batch=1
};

static NLPModelInfo get_model_info(const std::string& model_name) {
    if (model_name == "bert-base")
        return {"bert-base", 30522, true, true};
    if (model_name == "distilbert")
        return {"distilbert", 30522, false, true};
    if (model_name == "gpt2")
        return {"gpt2", 50257, false, false};
    if (model_name == "t5-small")
        return {"t5-small", 32128, false, true};

    // Default fallback
    return {model_name, 30522, false, true};
}

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<float> create_int64_as_float(
        const std::vector<int64_t>& int_data) {
    size_t float_count =
        (int_data.size() * sizeof(int64_t)) / sizeof(float);
    std::vector<float> result(float_count);
    std::memcpy(result.data(), int_data.data(),
                int_data.size() * sizeof(int64_t));
    return result;
}

static std::vector<int64_t> create_input_ids(int batch, int seq_len,
                                              int vocab_size) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(1, vocab_size - 1);
    std::vector<int64_t> ids(batch * seq_len);
    for (auto& id : ids) id = dist(rng);
    return ids;
}

static std::vector<int64_t> create_ones(int count) {
    return std::vector<int64_t>(count, 1);
}

static std::vector<int64_t> create_zeros(int count) {
    return std::vector<int64_t>(count, 0);
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        int val = std::stoi(token);
        if (val > 0) result.push_back(val);
    }
    return result;
}

static double percentile(std::vector<double>& sorted_data, double pct) {
    if (sorted_data.empty()) return 0.0;
    double idx = pct / 100.0 * static_cast<double>(sorted_data.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = lo + 1;
    if (hi >= sorted_data.size()) return sorted_data.back();
    double frac = idx - static_cast<double>(lo);
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac;
}

// ── Engine building ─────────────────────────────────────────────────────────

static std::string build_or_load_engine(
        const std::string& onnx_path,
        const std::string& engine_path,
        bool has_token_type_ids,
        bool batch_dynamic,
        Precision precision) {
    if (fs::exists(engine_path)) {
        std::cout << "  Loading cached engine: " << engine_path << "\n";
        return engine_path;
    }

    std::cout << "  Building engine from " << onnx_path << " ...\n";

    auto& logger = get_logger();
    EngineBuilder builder(logger);

    BuilderConfig config;
    config.precision = precision;
    config.max_workspace_size = 1ULL << 30;

    DynamicShapeProfile p;
    if (batch_dynamic) {
        p.min_dims = {1, 32};
        p.opt_dims = {4, 128};
        p.max_dims = {32, 512};
    } else {
        p.min_dims = {1, 32};
        p.opt_dims = {1, 128};
        p.max_dims = {1, 512};
    }

    p.name = "input_ids";
    config.dynamic_shapes.push_back(p);
    p.name = "attention_mask";
    config.dynamic_shapes.push_back(p);
    if (has_token_type_ids) {
        p.name = "token_type_ids";
        config.dynamic_shapes.push_back(p);
    }

    auto engine_data = builder.build_engine(onnx_path, config);
    if (engine_data.empty()) return "";

    EngineBuilder::save_engine(engine_data, engine_path);
    std::cout << "  Engine saved to " << engine_path << "\n";
    return engine_path;
}

// ── Core benchmark runner ───────────────────────────────────────────────────

static NLPBenchmarkResult run_benchmark(
        const std::string& engine_path,
        const NLPModelInfo& model_info,
        int batch_size,
        int seq_length,
        const std::string& precision_str,
        int num_iterations,
        int warmup_iterations,
        bool use_cuda_graph) {

    EngineConfig ecfg;
    ecfg.enable_cuda_graph = use_cuda_graph;
    auto engine = InferenceEngine::create(engine_path, ecfg);

    // Set shapes
    engine->set_input_shape("input_ids",     {batch_size, seq_length});
    engine->set_input_shape("attention_mask", {batch_size, seq_length});
    if (model_info.has_token_type_ids) {
        engine->set_input_shape("token_type_ids", {batch_size, seq_length});
    }

    // Pre-allocate device/pinned buffers for the fast path
    engine->prepare_buffers();

    // Create inputs
    int n = batch_size * seq_length;
    auto ids  = create_int64_as_float(
        create_input_ids(batch_size, seq_length, model_info.vocab_size));
    auto mask = create_int64_as_float(create_ones(n));

    std::vector<std::vector<float>> inputs = {ids, mask};
    if (model_info.has_token_type_ids) {
        inputs.push_back(create_int64_as_float(create_zeros(n)));
    }

    // Warmup
    for (int i = 0; i < warmup_iterations; ++i) {
        engine->infer(inputs);
    }

    // Collect per-iteration latencies
    std::vector<double> latencies;
    latencies.reserve(num_iterations);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        auto result = engine->infer(inputs);
        if (!result.success) {
            std::cerr << "  Inference failed at iteration " << i
                      << ": " << result.error_msg << "\n";
            break;
        }
        latencies.push_back(result.latency_ms);
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();

    // Compute statistics
    std::sort(latencies.begin(), latencies.end());

    double sum = 0.0;
    for (double l : latencies) sum += l;

    NLPBenchmarkResult res;
    res.model          = model_info.name;
    res.batch_size     = batch_size;
    res.seq_length     = seq_length;
    res.precision      = precision_str;
    res.cuda_graph     = use_cuda_graph;
    res.num_iterations = static_cast<int>(latencies.size());
    res.mean_latency_ms = latencies.empty() ? 0.0 : sum / latencies.size();
    res.p50_latency_ms  = percentile(latencies, 50.0);
    res.p95_latency_ms  = percentile(latencies, 95.0);
    res.p99_latency_ms  = percentile(latencies, 99.0);
    res.throughput_ips  =
        latencies.empty() ? 0.0
                          : (static_cast<double>(latencies.size()) / wall_ms) * 1000.0;
    res.tokens_per_sec  =
        res.throughput_ips * batch_size * seq_length;

    return res;
}

// ── CLI ─────────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --model <name>          Model name: bert-base, distilbert, gpt2, t5-small, or \"all\"\n"
        << "  --models-dir <path>     Path to models directory (default: models)\n"
        << "  --precision <str>       FP32 | FP16 | INT8 (default: FP16)\n"
        << "  --batch-sizes <list>    Comma-separated batch sizes (default: 1,4,8)\n"
        << "  --seq-lengths <list>    Comma-separated sequence lengths (default: 64,128)\n"
        << "  --iterations <n>        Number of timed iterations (default: 100)\n"
        << "  --warmup <n>            Number of warmup iterations (default: 20)\n"
        << "  --output <path>         Output JSON file path\n"
        << "  --cuda-graph            Enable CUDA graph capture\n"
        << "  --compare-precision     Run both FP32 and FP16, show speedup comparison\n"
        << "  --help                  Show this help\n";
}

static NLPBenchmarkConfig parse_args(int argc, char** argv) {
    NLPBenchmarkConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            cfg.model_name = argv[++i];
        } else if (arg == "--models-dir" && i + 1 < argc) {
            cfg.models_dir = argv[++i];
        } else if (arg == "--precision" && i + 1 < argc) {
            cfg.precision_str = argv[++i];
        } else if (arg == "--batch-sizes" && i + 1 < argc) {
            cfg.batch_sizes = parse_int_list(argv[++i]);
        } else if (arg == "--seq-lengths" && i + 1 < argc) {
            cfg.seq_lengths = parse_int_list(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            cfg.num_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_json = argv[++i];
        } else if (arg == "--cuda-graph") {
            cfg.use_cuda_graph = true;
        } else if (arg == "--compare-precision") {
            cfg.compare_precision = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
    }
    return cfg;
}

// ── JSON output ─────────────────────────────────────────────────────────────

static std::string results_to_json(
        const std::vector<NLPBenchmarkResult>& results,
        const NLPBenchmarkConfig& cfg) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\n";
    oss << "  \"benchmark\": \"nlp\",\n";
    oss << "  \"model\": \"" << cfg.model_name << "\",\n";
    oss << "  \"precision\": \"" << cfg.precision_str << "\",\n";
    oss << "  \"iterations\": " << cfg.num_iterations << ",\n";
    oss << "  \"warmup\": " << cfg.warmup_iterations << ",\n";
    oss << "  \"cuda_graph\": " << (cfg.use_cuda_graph ? "true" : "false") << ",\n";
    oss << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        oss << "    {\n";
        oss << "      \"model\": \"" << r.model << "\",\n";
        oss << "      \"batch_size\": " << r.batch_size << ",\n";
        oss << "      \"seq_length\": " << r.seq_length << ",\n";
        oss << "      \"precision\": \"" << r.precision << "\",\n";
        oss << "      \"cuda_graph\": " << (r.cuda_graph ? "true" : "false") << ",\n";
        oss << "      \"mean_latency_ms\": " << r.mean_latency_ms << ",\n";
        oss << "      \"p50_latency_ms\": " << r.p50_latency_ms << ",\n";
        oss << "      \"p95_latency_ms\": " << r.p95_latency_ms << ",\n";
        oss << "      \"p99_latency_ms\": " << r.p99_latency_ms << ",\n";
        oss << "      \"throughput_ips\": " << r.throughput_ips << ",\n";
        oss << "      \"tokens_per_sec\": " << r.tokens_per_sec << ",\n";
        oss << "      \"iterations\": " << r.num_iterations << "\n";
        oss << "    }";
        if (i + 1 < results.size()) oss << ",";
        oss << "\n";
    }

    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    auto cfg = parse_args(argc, argv);

    if (cfg.model_name.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]);
        return 1;
    }

    get_logger().set_severity(LogSeverity::WARNING);

    auto info = get_model_info(cfg.model_name);
    Precision precision = string_to_precision(cfg.precision_str);

    std::string onnx_path =
        cfg.models_dir + "/" + cfg.model_name + "/model.onnx";
    std::string engine_path =
        cfg.models_dir + "/" + cfg.model_name + "/model_" +
        cfg.precision_str + ".engine";

    if (!fs::exists(onnx_path)) {
        std::cerr << "Error: ONNX model not found: " << onnx_path << "\n";
        return 1;
    }

    std::cout << "=== NLP Benchmark ===\n"
              << "Model:      " << cfg.model_name << "\n"
              << "Precision:  " << cfg.precision_str << "\n"
              << "Iterations: " << cfg.num_iterations << "\n"
              << "Warmup:     " << cfg.warmup_iterations << "\n"
              << "CUDA Graph: " << (cfg.use_cuda_graph ? "Yes" : "No") << "\n\n";

    // Build or load engine
    auto ep = build_or_load_engine(
        onnx_path, engine_path,
        info.has_token_type_ids, info.batch_dynamic, precision);
    if (ep.empty()) {
        std::cerr << "Error: failed to build engine\n";
        return 1;
    }

    // For models without dynamic batch, only batch=1 is valid.
    std::vector<int> effective_batch_sizes = cfg.batch_sizes;
    if (!info.batch_dynamic) {
        effective_batch_sizes = {1};
        std::cout << "  Note: " << cfg.model_name
                  << " only supports batch_size=1\n";
    }

    std::vector<NLPBenchmarkResult> all_results;

    // Print header
    std::cout << "\n"
              << std::setw(8)  << "Batch"
              << std::setw(8)  << "SeqLen"
              << std::setw(12) << "Mean(ms)"
              << std::setw(12) << "P50(ms)"
              << std::setw(12) << "P95(ms)"
              << std::setw(12) << "P99(ms)"
              << std::setw(14) << "Throughput"
              << std::setw(16) << "Tokens/sec"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (int bs : effective_batch_sizes) {
        for (int sl : cfg.seq_lengths) {
            try {
                auto res = run_benchmark(
                    ep, info, bs, sl,
                    cfg.precision_str,
                    cfg.num_iterations,
                    cfg.warmup_iterations,
                    cfg.use_cuda_graph);
                all_results.push_back(res);

                std::cout << std::fixed << std::setprecision(2)
                          << std::setw(8)  << bs
                          << std::setw(8)  << sl
                          << std::setw(12) << res.mean_latency_ms
                          << std::setw(12) << res.p50_latency_ms
                          << std::setw(12) << res.p95_latency_ms
                          << std::setw(12) << res.p99_latency_ms
                          << std::setw(14) << res.throughput_ips
                          << std::setw(16) << res.tokens_per_sec
                          << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error (batch=" << bs << ", seq=" << sl
                          << "): " << e.what() << "\n";
            }
        }
    }

    // Write JSON
    if (!cfg.output_json.empty()) {
        std::string json = results_to_json(all_results, cfg);
        std::ofstream out(cfg.output_json);
        if (out.is_open()) {
            out << json;
            std::cout << "\nResults written to " << cfg.output_json << "\n";
        } else {
            std::cerr << "Failed to write: " << cfg.output_json << "\n";
        }
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
