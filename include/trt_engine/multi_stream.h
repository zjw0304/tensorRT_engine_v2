#pragma once

#include <trt_engine/cuda_utils.h>
#include <trt_engine/engine.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>
#include <trt_engine/types.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace trt_engine {

// ── Multi-stream inference engine ───────────────────────────────────────
// Each worker thread owns its own CudaStream and IExecutionContext.
// Incoming requests are dispatched to workers via a thread-safe queue.
class MultiStreamEngine {
public:
    /// Construct from an engine file on disk.
    /// @param engine_path    Path to the serialized TensorRT engine
    /// @param num_streams    Number of parallel worker streams/contexts
    /// @param config         Engine configuration
    MultiStreamEngine(const std::string& engine_path,
                      int num_streams,
                      const EngineConfig& config = {});

    ~MultiStreamEngine();

    // Not copyable or movable
    MultiStreamEngine(const MultiStreamEngine&) = delete;
    MultiStreamEngine& operator=(const MultiStreamEngine&) = delete;

    /// Synchronous inference -- dispatches to the next available worker
    /// and blocks until the result is ready.
    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);

    /// Submit an inference request and return a future for the result.
    std::future<InferenceResult> submit(
        const std::vector<std::vector<float>>& input_buffers);

    /// Gracefully shut down all worker threads.
    void shutdown();

    int num_streams() const { return num_streams_; }

private:
    struct InferRequest {
        std::vector<std::vector<float>>    inputs;
        std::promise<InferenceResult>      promise;
    };

    void worker_loop(int worker_id);

    // Shared state
    int            num_streams_;
    EngineConfig   config_;
    UniqueRuntime  runtime_;
    UniqueEngine   engine_;

    // Worker threads
    std::vector<std::thread> workers_;

    // Request queue
    std::mutex                        queue_mutex_;
    std::condition_variable           queue_cv_;
    std::queue<std::shared_ptr<InferRequest>> request_queue_;
    std::atomic<bool>                 shutdown_{false};

    // Helper to compute volume
    static int64_t volume(const nvinfer1::Dims& dims);
};

}  // namespace trt_engine
