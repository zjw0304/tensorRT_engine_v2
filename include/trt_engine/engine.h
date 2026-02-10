#pragma once

#include <NvInfer.h>
#include <trt_engine/cuda_graph.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>
#include <trt_engine/types.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trt_engine {

// ── Engine configuration ─────────────────────────────────────────────────
struct EngineConfig {
    int    device_id              = 0;
    int    context_pool_size      = 2;
    bool   enable_cuda_graph      = false;
    int    thread_pool_size       = 2;
    int    num_pipeline_streams   = 1;  // Number of streams for pipelining (1 = no pipelining)
    SyncMode sync_mode            = SyncMode::BLOCKING;
    uint64_t hybrid_spin_ns       = 100000;  // 100us default for HYBRID mode
};

// ── Exception type ───────────────────────────────────────────────────────
class EngineException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ── Core inference engine ────────────────────────────────────────────────
class InferenceEngine {
public:
    // Factory: create from serialized engine file on disk
    static std::unique_ptr<InferenceEngine> create(const std::string& engine_path,
                                                   const EngineConfig& config = {});

    // Factory: create from serialized engine data in memory
    static std::unique_ptr<InferenceEngine> create(const std::vector<char>& engine_data,
                                                   const EngineConfig& config = {});

    ~InferenceEngine();

    // Not copyable or movable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // Synchronous inference
    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);

    // Asynchronous inference via internal thread pool
    std::future<InferenceResult> infer_async(const std::vector<std::vector<float>>& input_buffers);

    // Set input shape for dynamic-shape models
    void set_input_shape(const std::string& name, const std::vector<int>& dims);

    // Pre-allocate device/pinned buffers for the current input shapes.
    // Call after set_input_shape() and before infer() in perf-critical loops.
    // This enables the zero-allocation fast path in run_inference().
    void prepare_buffers();

    // Pre-allocate pipeline resources for multi-stream pipelining.
    // Call after set_input_shape() to create N sets of buffers/streams.
    void prepare_pipeline();

    // Pipelined inference: process multiple inference requests across N streams.
    // Each element in batch_inputs is one complete set of input buffers for one inference call.
    std::vector<InferenceResult> infer_pipelined(
        const std::vector<std::vector<std::vector<float>>>& batch_inputs);

    // Query tensor information
    std::vector<TensorInfo> get_input_info() const;
    std::vector<TensorInfo> get_output_info() const;

    // Run N warmup iterations
    void warmup(int iterations = 5);

    // Access underlying engine (for advanced usage)
    nvinfer1::ICudaEngine* get_engine() const { return engine_.get(); }

private:
    // Private constructor -- use create() factories
    InferenceEngine(const std::vector<char>& engine_data, const EngineConfig& config);

    void init_context_pool();
    void init_thread_pool();
    void shutdown_thread_pool();

    // Context pool management
    UniqueContext acquire_context();
    void          release_context(UniqueContext ctx);

    // Internal inference logic
    InferenceResult run_inference(const std::vector<std::vector<float>>& input_buffers);

    // Compute volume (product of dims)
    static int64_t volume(const nvinfer1::Dims& dims);

    // Engine state
    EngineConfig   config_;
    UniqueRuntime  runtime_;
    UniqueEngine   engine_;

    // Context pool
    std::mutex                   ctx_mutex_;
    std::condition_variable      ctx_cv_;
    std::queue<UniqueContext>     ctx_pool_;

    // Dynamic shape overrides (name -> dims)
    std::mutex                                   shape_mutex_;
    std::unordered_map<std::string, std::vector<int>> shape_overrides_;

    // Pre-allocated inference resources (fast path)
    struct PreparedBuffers {
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<DeviceBuffer> input_device_bufs;
        std::vector<DeviceBuffer> output_device_bufs;
        std::vector<PinnedBuffer> input_pinned_bufs;
        std::vector<size_t> input_byte_sizes;
        std::vector<size_t> output_elem_counts;
        // Cached shape overrides snapshot (avoids mutex per call)
        std::vector<std::pair<std::string, nvinfer1::Dims>> cached_shapes;
        std::unique_ptr<CudaStream> stream;
        std::unique_ptr<CudaEvent> start_event;
        std::unique_ptr<CudaEvent> end_event;
        bool ready = false;
    };
    PreparedBuffers prepared_;

    // CUDA graph manager for multi-shape graph caching
    CudaGraphManager graph_manager_;

    // Pre-allocated pipeline resources for multi-stream pipelining
    struct PipelineStreamSet {
        CudaStream stream;
        CudaEvent start_event;
        CudaEvent end_event;
        std::vector<DeviceBuffer> input_device_bufs;
        std::vector<DeviceBuffer> output_device_bufs;
        std::vector<PinnedBuffer> input_pinned_bufs;
    };
    struct PipelineResources {
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<size_t> input_byte_sizes;
        std::vector<size_t> output_elem_counts;
        std::vector<std::pair<std::string, nvinfer1::Dims>> cached_shapes;
        std::vector<std::unique_ptr<PipelineStreamSet>> stream_sets;
        bool ready = false;
    };
    PipelineResources pipeline_;

    // Thread pool for async inference
    std::vector<std::thread>                     workers_;
    std::queue<std::packaged_task<InferenceResult()>> task_queue_;
    std::mutex                                   queue_mutex_;
    std::condition_variable                       queue_cv_;
    std::atomic<bool>                            shutdown_{false};
};

}  // namespace trt_engine
