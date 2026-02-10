#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_engine {

// ── CUDA Graph executor for repeated-shape inference ────────────────────
class CudaGraphExecutor {
public:
    CudaGraphExecutor();
    ~CudaGraphExecutor();

    // Not copyable or movable
    CudaGraphExecutor(const CudaGraphExecutor&) = delete;
    CudaGraphExecutor& operator=(const CudaGraphExecutor&) = delete;

    /// Capture a CUDA graph from an enqueueV3 call.
    /// Performs a pre-capture flush (one enqueueV3 call) before capturing.
    /// @param context   The execution context (with shapes and I/O addresses set)
    /// @param stream    The CUDA stream to capture on
    /// @return true on success
    bool capture(nvinfer1::IExecutionContext* context, cudaStream_t stream);

    /// Launch the captured graph on the given stream.
    /// @return true on success; false if no graph is captured
    bool launch(cudaStream_t stream);

    /// Whether a graph has been successfully captured.
    bool is_captured() const;

    /// Destroy the captured graph and allow re-capture.
    void reset();

private:
    mutable std::mutex  mutex_;
    cudaGraph_t         graph_     = nullptr;
    cudaGraphExec_t     instance_  = nullptr;
    bool                captured_  = false;
};

// ── CUDA Graph manager for multiple shape configurations ────────────────
class CudaGraphManager {
public:
    CudaGraphManager() = default;
    ~CudaGraphManager() = default;

    // Not copyable or movable
    CudaGraphManager(const CudaGraphManager&) = delete;
    CudaGraphManager& operator=(const CudaGraphManager&) = delete;

    /// Capture a CUDA graph for a specific shape key.
    bool capture(const std::string& key, nvinfer1::IExecutionContext* context,
                 cudaStream_t stream);

    /// Launch the cached graph for a given key.
    bool launch(const std::string& key, cudaStream_t stream);

    /// Check if a graph exists for the given key.
    bool has_graph(const std::string& key) const;

    /// Remove a cached graph by key.
    void remove(const std::string& key);

    /// Remove all cached graphs.
    void clear();

    /// Number of cached graphs.
    size_t size() const;

    /// Build a key string from shape pairs.
    static std::string make_key(
        const std::vector<std::pair<std::string, nvinfer1::Dims>>& shapes);

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<CudaGraphExecutor>> graphs_;
};

}  // namespace trt_engine
