// test_multi_gpu.cpp - Unit tests for MultiGPUEngine
//
// Tests device enumeration, round-robin distribution, and basic multi-GPU
// inference. Tests are skipped when fewer than the required number of GPUs
// are available.

#include <gtest/gtest.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/multi_gpu.h>
#include <trt_engine/types.h>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Helpers ─────────────────────────────────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

static int get_num_gpus() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
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

// ── Device enumeration tests ────────────────────────────────────────────────

TEST(MultiGPUTest, DeviceCount) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    int count = get_device_count();
    EXPECT_GE(count, 1);
}

TEST(MultiGPUTest, DeviceProperties) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    auto props = get_device_properties(0);
    EXPECT_FALSE(props.name.empty());
    EXPECT_GT(props.total_global_memory, 0u);
    EXPECT_GT(props.multi_processor_count, 0);
    EXPECT_GE(props.compute_capability_major, 1);
}

// ── Construction tests ──────────────────────────────────────────────────────

TEST(MultiGPUEngineTest, EmptyDeviceList) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
    auto engine_path = find_test_engine();
    if (engine_path.empty()) GTEST_SKIP() << "No test engine";

    EXPECT_THROW({
        MultiGPUEngine engine(engine_path, {});
    }, EngineException);
}

TEST(MultiGPUEngineTest, InvalidDeviceID) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
    auto engine_path = find_test_engine();
    if (engine_path.empty()) GTEST_SKIP() << "No test engine";

    EXPECT_THROW({
        MultiGPUEngine engine(engine_path, {999});
    }, EngineException);
}

// ── Single GPU tests ────────────────────────────────────────────────────────

class MultiGPUSingleDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        engine_path_ = find_test_engine();
        if (engine_path_.empty()) GTEST_SKIP() << "No test engine";
    }

    std::string engine_path_;
};

TEST_F(MultiGPUSingleDeviceTest, CreateWithSingleDevice) {
    MultiGPUEngine engine(engine_path_, {0});
    EXPECT_EQ(engine.get_device_count(), 1);
    EXPECT_EQ(engine.get_device_ids().size(), 1u);
    EXPECT_EQ(engine.get_device_ids()[0], 0);
}

TEST_F(MultiGPUSingleDeviceTest, GetDeviceInfo) {
    MultiGPUEngine engine(engine_path_, {0});
    auto info = engine.get_device_info(0);
    EXPECT_FALSE(info.name.empty());
}

TEST_F(MultiGPUSingleDeviceTest, GetDeviceInfoOutOfRange) {
    MultiGPUEngine engine(engine_path_, {0});
    EXPECT_THROW({
        engine.get_device_info(1);
    }, EngineException);
}

TEST_F(MultiGPUSingleDeviceTest, RoundRobinSingleDevice) {
    // With a single device, all requests go to device 0
    MultiGPUEngine engine(engine_path_, {0});

    // Create input
    auto eng = InferenceEngine::create(engine_path_);
    auto inputs_info = eng->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) vol *= (d > 0) ? d : 1;
        inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
    }

    // Run multiple inferences
    for (int i = 0; i < 5; ++i) {
        auto result = engine.infer(inputs);
        EXPECT_TRUE(result.success) << result.error_msg;
    }
}

// ── Multi-GPU tests (require 2+ GPUs) ──────────────────────────────────────

class MultiGPUMultiDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        if (get_num_gpus() < 2) GTEST_SKIP() << "Need 2+ GPUs";
        engine_path_ = find_test_engine();
        if (engine_path_.empty()) GTEST_SKIP() << "No test engine";
    }

    std::string engine_path_;
};

TEST_F(MultiGPUMultiDeviceTest, CreateWithMultipleDevices) {
    MultiGPUEngine engine(engine_path_, {0, 1});
    EXPECT_EQ(engine.get_device_count(), 2);
}

TEST_F(MultiGPUMultiDeviceTest, RoundRobinDistribution) {
    MultiGPUEngine engine(engine_path_, {0, 1});

    auto eng = InferenceEngine::create(engine_path_);
    auto inputs_info = eng->get_input_info();
    std::vector<std::vector<float>> inputs;
    for (const auto& ti : inputs_info) {
        int64_t vol = 1;
        for (int d : ti.shape) vol *= (d > 0) ? d : 1;
        inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
    }

    // Run enough inferences to test round-robin
    for (int i = 0; i < 6; ++i) {
        auto result = engine.infer(inputs);
        EXPECT_TRUE(result.success) << result.error_msg;
    }
}
