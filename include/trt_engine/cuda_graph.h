#pragma once

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>

#include <mutex>
#include <string>
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

}  // namespace trt_engine
