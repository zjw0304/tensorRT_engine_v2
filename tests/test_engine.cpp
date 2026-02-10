// test_engine.cpp - Unit tests for InferenceEngine
//
// Tests engine loading, input/output info queries, inference, async
// inference, and warmup. Tests requiring a real engine are skipped if
// no serialized engine file is found.

#include <gtest/gtest.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/engine.h>
#include <trt_engine/types.h>

#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Helper: check if CUDA is available ──────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// ── Helper: find a test engine file ─────────────────────────────────────────

static std::string find_test_engine() {
    std::vector<std::string> paths = {
        "/tmp/test_model.engine",
        "test_data/test_model.engine",
        "../test_data/test_model.engine",
    };
    for (const auto& p : paths) {
        if (fs::exists(p)) return p;
    }
    return "";
}

// ── EngineConfig tests ──────────────────────────────────────────────────────

TEST(EngineConfigTest, DefaultValues) {
    EngineConfig config;
    EXPECT_EQ(config.device_id, 0);
    EXPECT_EQ(config.context_pool_size, 2);
    EXPECT_FALSE(config.enable_cuda_graph);
    EXPECT_EQ(config.thread_pool_size, 2);
    EXPECT_EQ(config.num_pipeline_streams, 1);
}

TEST(EngineConfigTest, CustomValues) {
    EngineConfig config;
    config.device_id = 1;
    config.context_pool_size = 4;
    config.enable_cuda_graph = true;
    config.thread_pool_size = 8;

    EXPECT_EQ(config.device_id, 1);
    EXPECT_EQ(config.context_pool_size, 4);
    EXPECT_TRUE(config.enable_cuda_graph);
    EXPECT_EQ(config.thread_pool_size, 8);
}

// ── EngineException tests ───────────────────────────────────────────────────

TEST(EngineExceptionTest, ThrowAndCatch) {
    EXPECT_THROW({
        throw EngineException("test error");
    }, EngineException);
}

TEST(EngineExceptionTest, MessageContent) {
    try {
        throw EngineException("engine load failed");
    } catch (const EngineException& e) {
        EXPECT_STREQ(e.what(), "engine load failed");
    }
}

TEST(EngineExceptionTest, InheritsFromRuntimeError) {
    EXPECT_THROW({
        throw EngineException("test");
    }, std::runtime_error);
}

// ── Engine creation from invalid path ───────────────────────────────────────

TEST(InferenceEngineTest, CreateFromInvalidPath) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    EXPECT_THROW({
        auto engine = InferenceEngine::create("/nonexistent/path.engine");
    }, EngineException);
}

TEST(InferenceEngineTest, CreateFromEmptyData) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<char> empty_data;
    EXPECT_THROW({
        auto engine = InferenceEngine::create(empty_data);
    }, std::exception);
}

TEST(InferenceEngineTest, CreateFromInvalidData) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    std::vector<char> bad_data = {'N', 'O', 'T', '_', 'T', 'R', 'T'};
    EXPECT_THROW({
        auto engine = InferenceEngine::create(bad_data);
    }, std::exception);
}

// ── Tests requiring a real engine file ──────────────────────────────────────

class InferenceEngineWithModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) {
            GTEST_SKIP() << "No CUDA device";
        }
        engine_path_ = find_test_engine();
        if (engine_path_.empty()) {
            GTEST_SKIP() << "No test engine file found";
        }
    }

    std::string engine_path_;
};

TEST_F(InferenceEngineWithModelTest, CreateFromFile) {
    auto engine = InferenceEngine::create(engine_path_);
    ASSERT_NE(engine, nullptr);
}

TEST_F(InferenceEngineWithModelTest, GetInputInfo) {
    auto engine = InferenceEngine::create(engine_path_);
    auto inputs = engine->get_input_info();
    EXPECT_GT(inputs.size(), 0u);

    for (const auto& ti : inputs) {
        EXPECT_FALSE(ti.name.empty());
        EXPECT_FALSE(ti.shape.empty());
    }
}

TEST_F(InferenceEngineWithModelTest, GetOutputInfo) {
    auto engine = InferenceEngine::create(engine_path_);
    auto outputs = engine->get_output_info();
    EXPECT_GT(outputs.size(), 0u);

    for (const auto& ti : outputs) {
        EXPECT_FALSE(ti.name.empty());
        EXPECT_FALSE(ti.shape.empty());
    }
}

TEST_F(InferenceEngineWithModelTest, Warmup) {
    auto engine = InferenceEngine::create(engine_path_);
    EXPECT_NO_THROW(engine->warmup(2));
}

TEST_F(InferenceEngineWithModelTest, SyncInference) {
    auto engine = InferenceEngine::create(engine_path_);
    engine->warmup(1);

    auto inputs_info = engine->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) {
            vol *= (d > 0) ? d : 1;
        }
        inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
    }

    auto result = engine->infer(inputs);
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
    EXPECT_GT(result.latency_ms, 0.0f);
}

TEST_F(InferenceEngineWithModelTest, AsyncInference) {
    auto engine = InferenceEngine::create(engine_path_);
    engine->warmup(1);

    auto inputs_info = engine->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) {
            vol *= (d > 0) ? d : 1;
        }
        inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
    }

    auto future = engine->infer_async(inputs);
    auto result = future.get();
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(InferenceEngineWithModelTest, SetInputShape) {
    auto engine = InferenceEngine::create(engine_path_);
    auto inputs_info = engine->get_input_info();
    if (inputs_info.empty()) {
        GTEST_SKIP() << "No inputs in engine";
    }

    // This should not throw for a valid input name
    EXPECT_NO_THROW({
        engine->set_input_shape(inputs_info[0].name, inputs_info[0].shape);
    });
}

TEST_F(InferenceEngineWithModelTest, InputCountMismatch) {
    auto engine = InferenceEngine::create(engine_path_);
    engine->warmup(1);

    // Pass wrong number of inputs
    std::vector<std::vector<float>> bad_inputs;
    auto result = engine->infer(bad_inputs);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_msg.empty());
}

// ── TensorInfo tests ────────────────────────────────────────────────────────

TEST(TensorInfoTest, DefaultValues) {
    TensorInfo ti;
    EXPECT_TRUE(ti.name.empty());
    EXPECT_TRUE(ti.shape.empty());
    EXPECT_EQ(ti.size_bytes, 0u);
}

// ── InferenceResult tests ───────────────────────────────────────────────────

TEST(InferenceResultTest, DefaultValues) {
    InferenceResult result;
    EXPECT_TRUE(result.outputs.empty());
    EXPECT_FLOAT_EQ(result.latency_ms, 0.0f);
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error_msg.empty());
}

// ── Pipeline tests ──────────────────────────────────────────────────────────

static std::string find_nlp_engine() {
    // Prefer smaller models first to avoid OOM on constrained GPUs
    std::vector<std::string> paths = {
        "models/resnet18/model_fp16.engine",
        "../models/resnet18/model_fp16.engine",
        "/mnt/data2/tensorRT_engine_v2/models/resnet18/model_fp16.engine",
        "models/bert-base/model_FP16.engine",
        "../models/bert-base/model_FP16.engine",
        "/mnt/data2/tensorRT_engine_v2/models/bert-base/model_FP16.engine",
    };
    for (const auto& p : paths) {
        if (fs::exists(p)) return p;
    }
    // Fall back to the general test engine
    return find_test_engine();
}

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) {
            GTEST_SKIP() << "No CUDA device";
        }
        engine_path_ = find_nlp_engine();
        if (engine_path_.empty()) {
            GTEST_SKIP() << "No test engine file found";
        }
    }

    // Build dummy inputs from the engine's input specs.
    // Also sets input shapes for dynamic-dim models (replacing -1 with 1).
    std::vector<std::vector<float>> make_dummy_inputs(InferenceEngine& engine) {
        auto inputs_info = engine.get_input_info();
        std::vector<std::vector<float>> inputs;
        for (const auto& ti : inputs_info) {
            std::vector<int> resolved_shape;
            int64_t vol = 1;
            for (int d : ti.shape) {
                int dim = (d > 0) ? d : 1;
                resolved_shape.push_back(dim);
                vol *= dim;
            }
            // Set input shape to resolve dynamic dims
            bool has_dynamic = false;
            for (int d : ti.shape) {
                if (d <= 0) { has_dynamic = true; break; }
            }
            if (has_dynamic) {
                engine.set_input_shape(ti.name, resolved_shape);
            }
            inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
        }
        return inputs;
    }

    std::string engine_path_;
};

TEST_F(PipelineTest, SingleStream) {
    EngineConfig config;
    config.num_pipeline_streams = 1;
    auto engine = InferenceEngine::create(engine_path_, config);

    auto dummy = make_dummy_inputs(*engine);

    // Run sequential inference for reference; skip if OOM
    auto seq_result = engine->infer(dummy);
    if (!seq_result.success &&
        seq_result.error_msg.find("out of memory") != std::string::npos) {
        GTEST_SKIP() << "Insufficient GPU memory for model";
    }
    ASSERT_TRUE(seq_result.success) << seq_result.error_msg;

    // Run pipelined with 1 stream (should be equivalent to sequential)
    engine->prepare_pipeline();
    std::vector<std::vector<std::vector<float>>> batch = { dummy };
    auto pipeline_results = engine->infer_pipelined(batch);

    ASSERT_EQ(pipeline_results.size(), 1u);
    ASSERT_TRUE(pipeline_results[0].success) << pipeline_results[0].error_msg;
    EXPECT_GT(pipeline_results[0].latency_ms, 0.0f);
    ASSERT_EQ(pipeline_results[0].outputs.size(), seq_result.outputs.size());

    // Verify output shapes match
    for (size_t i = 0; i < seq_result.outputs.size(); ++i) {
        EXPECT_EQ(pipeline_results[0].outputs[i].size(),
                  seq_result.outputs[i].size());
    }
}

TEST_F(PipelineTest, MultipleStreams) {
    EngineConfig config;
    config.num_pipeline_streams = 2;
    config.context_pool_size = 4;  // Need enough contexts for pipeline
    auto engine = InferenceEngine::create(engine_path_, config);

    auto dummy = make_dummy_inputs(*engine);

    // Quick check: can we even run a single inference? Skip if OOM.
    auto check = engine->infer(dummy);
    if (!check.success &&
        check.error_msg.find("out of memory") != std::string::npos) {
        GTEST_SKIP() << "Insufficient GPU memory for model";
    }
    ASSERT_TRUE(check.success) << check.error_msg;

    // Build a batch of 4 requests
    std::vector<std::vector<std::vector<float>>> batch;
    for (int i = 0; i < 4; ++i) {
        batch.push_back(dummy);
    }

    engine->prepare_pipeline();
    auto results = engine->infer_pipelined(batch);

    ASSERT_EQ(results.size(), 4u);
    for (size_t r = 0; r < results.size(); ++r) {
        ASSERT_TRUE(results[r].success)
            << "Request " << r << " failed: " << results[r].error_msg;
        EXPECT_GT(results[r].latency_ms, 0.0f);
        EXPECT_GT(results[r].outputs.size(), 0u);
    }
}

// ── SyncMode tests ──────────────────────────────────────────────────────────

TEST(SyncModeTest, SpinWaitEvent) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaStream stream;
    CudaEvent event;

    // Record an event on the stream (no work queued, completes immediately)
    event.record(stream.get());

    // spin_wait_event should return without error
    EXPECT_NO_THROW(spin_wait_event(event));
}

TEST(SyncModeTest, HybridWaitEvent) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaStream stream;
    CudaEvent event;

    event.record(stream.get());

    // hybrid_wait_event should return without error
    EXPECT_NO_THROW(hybrid_wait_event(event, stream.get(), 100000));
}

TEST(SyncModeTest, SyncStreamBlocking) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaStream stream;
    CudaEvent event;

    event.record(stream.get());

    // Test BLOCKING mode via sync_stream
    EXPECT_NO_THROW(sync_stream(stream, event, SyncMode::BLOCKING));
}

TEST(SyncModeTest, SyncStreamSpinWait) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaStream stream;
    CudaEvent event;

    event.record(stream.get());

    EXPECT_NO_THROW(sync_stream(stream, event, SyncMode::SPIN_WAIT));
}

TEST(SyncModeTest, SyncStreamHybrid) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaStream stream;
    CudaEvent event;

    event.record(stream.get());

    EXPECT_NO_THROW(sync_stream(stream, event, SyncMode::HYBRID, 50000));
}

TEST(SyncModeTest, EngineConfigDefaults) {
    EngineConfig config;
    EXPECT_EQ(config.sync_mode, SyncMode::BLOCKING);
    EXPECT_EQ(config.hybrid_spin_ns, 100000u);
}
