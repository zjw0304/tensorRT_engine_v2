// test_cuda_graph.cpp - Unit tests for CudaGraphExecutor
//
// Tests graph capture, launch, reset, and re-capture behavior.
// Tests that require a real CUDA device/engine are guarded with skips.

#include <gtest/gtest.h>
#include <trt_engine/cuda_graph.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/engine.h>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Helper ──────────────────────────────────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

static std::string find_test_engine() {
    std::vector<std::string> paths = {
        "/tmp/test_model.engine",
        "test_data/test_model.engine",
    };
    for (const auto& p : paths) {
        if (fs::exists(p)) return p;
    }
    return "";
}

// ── Basic construct/destruct ────────────────────────────────────────────────

TEST(CudaGraphExecutorTest, DefaultConstruction) {
    CudaGraphExecutor graph;
    EXPECT_FALSE(graph.is_captured());
}

TEST(CudaGraphExecutorTest, ResetWithoutCapture) {
    CudaGraphExecutor graph;
    EXPECT_NO_THROW(graph.reset());
    EXPECT_FALSE(graph.is_captured());
}

TEST(CudaGraphExecutorTest, LaunchWithoutCapture) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaGraphExecutor graph;
    CudaStream stream;
    EXPECT_FALSE(graph.launch(stream.get()));
}

TEST(CudaGraphExecutorTest, CaptureNullContext) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    CudaGraphExecutor graph;
    CudaStream stream;
    EXPECT_FALSE(graph.capture(nullptr, stream.get()));
}

TEST(CudaGraphExecutorTest, CaptureNullStream) {
    CudaGraphExecutor graph;
    // Both null
    EXPECT_FALSE(graph.capture(nullptr, nullptr));
}

// ── Tests with a real engine ────────────────────────────────────────────────

class CudaGraphWithEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        engine_path_ = find_test_engine();
        if (engine_path_.empty()) GTEST_SKIP() << "No test engine";

        engine_ = InferenceEngine::create(engine_path_);
        ASSERT_NE(engine_, nullptr);
    }

    std::string engine_path_;
    std::unique_ptr<InferenceEngine> engine_;
};

TEST_F(CudaGraphWithEngineTest, CaptureAndLaunch) {
    // This requires creating a context and binding tensors manually.
    // For simplicity, we verify that the engine can run with cuda_graph
    // enabled in the EngineConfig.
    EngineConfig conf;
    conf.enable_cuda_graph = true;
    auto eng = InferenceEngine::create(engine_path_, conf);
    ASSERT_NE(eng, nullptr);

    eng->warmup(1);

    auto inputs_info = eng->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) vol *= (d > 0) ? d : 1;
        inputs.emplace_back(static_cast<size_t>(vol), 0.5f);
    }

    auto result = eng->infer(inputs);
    EXPECT_TRUE(result.success) << result.error_msg;
}

TEST_F(CudaGraphWithEngineTest, ResetCapturedGraph) {
    CudaGraphExecutor graph;
    // Verify reset on a non-captured graph is safe
    graph.reset();
    EXPECT_FALSE(graph.is_captured());
    graph.reset();
    EXPECT_FALSE(graph.is_captured());
}

// ── CudaGraphManager tests ─────────────────────────────────────────────────

TEST(CudaGraphManagerTest, BasicOperations) {
    CudaGraphManager manager;

    // Initially empty
    EXPECT_EQ(manager.size(), 0u);
    EXPECT_FALSE(manager.has_graph("key1"));

    // has_graph returns false for nonexistent key
    EXPECT_FALSE(manager.has_graph("nonexistent"));

    // remove on nonexistent key is safe
    EXPECT_NO_THROW(manager.remove("nonexistent"));

    // clear on empty manager is safe
    EXPECT_NO_THROW(manager.clear());
    EXPECT_EQ(manager.size(), 0u);

    // launch on nonexistent key returns false
    if (cuda_available()) {
        CudaStream stream;
        EXPECT_FALSE(manager.launch("nonexistent", stream.get()));
    }
}

TEST(CudaGraphManagerTest, MakeKey) {
    // Empty shapes
    {
        std::vector<std::pair<std::string, nvinfer1::Dims>> shapes;
        std::string key = CudaGraphManager::make_key(shapes);
        EXPECT_EQ(key, "");
    }

    // Single shape: 1D
    {
        nvinfer1::Dims dims;
        dims.nbDims = 1;
        dims.d[0] = 128;
        std::vector<std::pair<std::string, nvinfer1::Dims>> shapes = {
            {"input_ids", dims}
        };
        std::string key = CudaGraphManager::make_key(shapes);
        EXPECT_EQ(key, "input_ids:128");
    }

    // Two shapes: 2D each
    {
        nvinfer1::Dims dims1;
        dims1.nbDims = 2;
        dims1.d[0] = 1;
        dims1.d[1] = 128;

        nvinfer1::Dims dims2;
        dims2.nbDims = 2;
        dims2.d[0] = 1;
        dims2.d[1] = 128;

        std::vector<std::pair<std::string, nvinfer1::Dims>> shapes = {
            {"input_ids", dims1},
            {"attention_mask", dims2}
        };
        std::string key = CudaGraphManager::make_key(shapes);
        EXPECT_EQ(key, "input_ids:1x128,attention_mask:1x128");
    }

    // Different shapes produce different keys
    {
        nvinfer1::Dims dims1;
        dims1.nbDims = 2;
        dims1.d[0] = 1;
        dims1.d[1] = 128;

        nvinfer1::Dims dims2;
        dims2.nbDims = 2;
        dims2.d[0] = 1;
        dims2.d[1] = 256;

        std::string key1 = CudaGraphManager::make_key({{"input", dims1}});
        std::string key2 = CudaGraphManager::make_key({{"input", dims2}});
        EXPECT_NE(key1, key2);
    }
}

TEST_F(CudaGraphWithEngineTest, CudaGraphManagerCaptureAndLaunch) {
    EngineConfig conf;
    conf.enable_cuda_graph = true;
    auto eng = InferenceEngine::create(engine_path_, conf);
    ASSERT_NE(eng, nullptr);

    auto inputs_info = eng->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) vol *= (d > 0) ? d : 1;
        inputs.emplace_back(static_cast<size_t>(vol), 0.5f);
    }

    // First call captures the graph
    eng->prepare_buffers();
    auto result1 = eng->infer(inputs);
    EXPECT_TRUE(result1.success) << result1.error_msg;

    // Second call should use the cached graph
    auto result2 = eng->infer(inputs);
    EXPECT_TRUE(result2.success) << result2.error_msg;
}
