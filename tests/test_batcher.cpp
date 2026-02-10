// test_batcher.cpp - Unit tests for DynamicBatcher
//
// Tests single request, batch formation, timeout behavior, and concurrent
// submissions. Tests requiring a real engine are skipped when unavailable.

#include <gtest/gtest.h>
#include <trt_engine/batcher.h>
#include <trt_engine/engine.h>
#include <trt_engine/types.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Helpers ─────────────────────────────────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
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

// ── Construction tests ──────────────────────────────────────────────────────

TEST(DynamicBatcherTest, InvalidNullEngine) {
    EXPECT_THROW({
        DynamicBatcher batcher(nullptr, 4, 50);
    }, EngineException);
}

TEST(DynamicBatcherTest, InvalidBatchSize) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    auto engine_path = find_test_engine();
    if (engine_path.empty()) GTEST_SKIP() << "No test engine";

    auto engine = InferenceEngine::create(engine_path);
    auto shared_engine = std::shared_ptr<InferenceEngine>(engine.release());

    EXPECT_THROW({
        DynamicBatcher batcher(shared_engine, 0, 50);
    }, EngineException);
}

TEST(DynamicBatcherTest, InvalidWaitTime) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    auto engine_path = find_test_engine();
    if (engine_path.empty()) GTEST_SKIP() << "No test engine";

    auto engine = InferenceEngine::create(engine_path);
    auto shared_engine = std::shared_ptr<InferenceEngine>(engine.release());

    EXPECT_THROW({
        DynamicBatcher batcher(shared_engine, 4, -1);
    }, EngineException);
}

// ── Functional tests (require engine) ───────────────────────────────────────

class DynamicBatcherWithEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        engine_path_ = find_test_engine();
        if (engine_path_.empty()) GTEST_SKIP() << "No test engine";

        auto engine = InferenceEngine::create(engine_path_);
        engine_ = std::shared_ptr<InferenceEngine>(engine.release());
        engine_->warmup(1);
    }

    std::string engine_path_;
    std::shared_ptr<InferenceEngine> engine_;

    std::vector<std::vector<float>> make_dummy_inputs() {
        auto inputs_info = engine_->get_input_info();
        std::vector<std::vector<float>> inputs;
        for (const auto& ti : inputs_info) {
            int64_t vol = 1;
            for (int d : ti.shape) vol *= (d > 0) ? d : 1;
            inputs.emplace_back(static_cast<size_t>(vol), 0.5f);
        }
        return inputs;
    }
};

TEST_F(DynamicBatcherWithEngineTest, SingleRequest) {
    DynamicBatcher batcher(engine_, 4, 100);
    auto inputs = make_dummy_inputs();

    auto future = batcher.submit(inputs);
    auto result = future.get();
    EXPECT_TRUE(result.success) << result.error_msg;
}

TEST_F(DynamicBatcherWithEngineTest, MultipleRequests) {
    DynamicBatcher batcher(engine_, 4, 200);
    auto inputs = make_dummy_inputs();

    constexpr int num_requests = 8;
    std::vector<std::future<InferenceResult>> futures;
    futures.reserve(num_requests);

    for (int i = 0; i < num_requests; ++i) {
        futures.push_back(batcher.submit(inputs));
    }

    int success_count = 0;
    for (auto& f : futures) {
        auto result = f.get();
        if (result.success) ++success_count;
    }
    EXPECT_EQ(success_count, num_requests);
}

TEST_F(DynamicBatcherWithEngineTest, ConcurrentSubmissions) {
    DynamicBatcher batcher(engine_, 4, 200);
    auto inputs = make_dummy_inputs();

    constexpr int num_threads = 4;
    constexpr int requests_per_thread = 2;

    std::vector<std::future<InferenceResult>> all_futures;
    std::vector<std::thread> threads;
    std::mutex futures_mutex;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < requests_per_thread; ++i) {
                auto future = batcher.submit(inputs);
                std::lock_guard<std::mutex> lock(futures_mutex);
                all_futures.push_back(std::move(future));
            }
        });
    }

    for (auto& th : threads) th.join();

    int total_success = 0;
    for (auto& f : all_futures) {
        auto result = f.get();
        if (result.success) ++total_success;
    }
    EXPECT_EQ(total_success, num_threads * requests_per_thread);
}

TEST_F(DynamicBatcherWithEngineTest, TimeoutBehavior) {
    // With a short timeout, a single request should still complete
    DynamicBatcher batcher(engine_, 16, 10);  // 10ms timeout, batch=16
    auto inputs = make_dummy_inputs();

    auto future = batcher.submit(inputs);
    auto result = future.get();
    EXPECT_TRUE(result.success) << result.error_msg;
}

TEST_F(DynamicBatcherWithEngineTest, Properties) {
    DynamicBatcher batcher(engine_, 8, 100);
    EXPECT_EQ(batcher.max_batch_size(), 8);
    EXPECT_EQ(batcher.max_wait_time_ms(), 100);
}
