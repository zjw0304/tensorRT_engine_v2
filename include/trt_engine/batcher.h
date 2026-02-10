#pragma once

#include <trt_engine/engine.h>
#include <trt_engine/types.h>

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace trt_engine {

// ── Dynamic batcher ─────────────────────────────────────────────────────
// Collects individual inference requests and batches them together before
// sending to the underlying InferenceEngine.  The batch is formed when
// either max_batch_size requests accumulate or max_wait_time_ms elapses.
class DynamicBatcher {
public:
    /// @param engine          Shared pointer to a fully-initialized InferenceEngine
    /// @param max_batch_size  Maximum number of requests per batch
    /// @param max_wait_time_ms  Maximum milliseconds to wait for a full batch
    DynamicBatcher(std::shared_ptr<InferenceEngine> engine,
                   int max_batch_size,
                   int max_wait_time_ms);

    ~DynamicBatcher();

    // Not copyable or movable
    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    /// Submit a single-sample inference request.
    /// The input is a set of per-tensor float vectors for ONE sample.
    /// Returns a future that will contain the individual result.
    std::future<InferenceResult> submit(
        const std::vector<std::vector<float>>& single_input);

    int max_batch_size() const { return max_batch_size_; }
    int max_wait_time_ms() const { return max_wait_time_ms_; }

private:
    struct PendingRequest {
        std::vector<std::vector<float>>  inputs;   // per-tensor data for 1 sample
        std::promise<InferenceResult>    promise;
    };

    void batch_loop();

    void execute_batch(std::vector<std::shared_ptr<PendingRequest>>& batch);

    std::shared_ptr<InferenceEngine> engine_;
    int max_batch_size_;
    int max_wait_time_ms_;

    std::mutex                                      queue_mutex_;
    std::condition_variable                          queue_cv_;
    std::queue<std::shared_ptr<PendingRequest>>      pending_queue_;

    std::thread      batch_thread_;
    std::atomic<bool> shutdown_{false};
};

}  // namespace trt_engine
