#include <trt_engine/cuda_graph.h>

#include <sstream>
#include <stdexcept>

namespace trt_engine {

CudaGraphExecutor::CudaGraphExecutor() = default;

CudaGraphExecutor::~CudaGraphExecutor() {
    reset();
}

bool CudaGraphExecutor::capture(nvinfer1::IExecutionContext* context,
                                cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Destroy any previous capture
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    captured_ = false;

    if (!context || !stream) {
        get_logger().error("CudaGraphExecutor::capture: null context or stream");
        return false;
    }

    // Step 1: Pre-capture flush -- run enqueueV3 once to flush deferred updates
    if (!context->enqueueV3(stream)) {
        get_logger().error("CudaGraphExecutor: pre-capture flush enqueueV3 failed");
        return false;
    }
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        get_logger().error("CudaGraphExecutor: pre-capture sync failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }

    // Step 2: Begin capture
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        get_logger().error("cudaStreamBeginCapture failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }

    // Step 3: Enqueue the inference call inside the capture
    if (!context->enqueueV3(stream)) {
        // End capture to clean up, discard graph
        cudaGraph_t discard = nullptr;
        cudaStreamEndCapture(stream, &discard);
        if (discard) cudaGraphDestroy(discard);
        get_logger().error("CudaGraphExecutor: enqueueV3 during capture failed");
        return false;
    }

    // Step 4: End capture
    err = cudaStreamEndCapture(stream, &graph_);
    if (err != cudaSuccess || !graph_) {
        get_logger().error("cudaStreamEndCapture failed: " +
                           std::string(cudaGetErrorString(err)));
        graph_ = nullptr;
        return false;
    }

    // Step 5: Instantiate the graph
    err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        get_logger().error("cudaGraphInstantiate failed: " +
                           std::string(cudaGetErrorString(err)));
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        return false;
    }

    captured_ = true;
    get_logger().info("CUDA graph captured successfully");
    return true;
}

bool CudaGraphExecutor::launch(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!captured_ || !instance_) {
        get_logger().error("CudaGraphExecutor::launch: no graph captured");
        return false;
    }

    cudaError_t err = cudaGraphLaunch(instance_, stream);
    if (err != cudaSuccess) {
        get_logger().error("cudaGraphLaunch failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }
    return true;
}

bool CudaGraphExecutor::is_captured() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return captured_;
}

void CudaGraphExecutor::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    captured_ = false;
}

// ── CudaGraphManager ───────────────────────────────────────────────────

bool CudaGraphManager::capture(const std::string& key,
                               nvinfer1::IExecutionContext* context,
                               cudaStream_t stream) {
    auto executor = std::make_unique<CudaGraphExecutor>();
    if (!executor->capture(context, stream)) {
        get_logger().error("CudaGraphManager: capture failed for key: " + key);
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    graphs_[key] = std::move(executor);
    get_logger().info("CudaGraphManager: cached graph for key: " + key);
    return true;
}

bool CudaGraphManager::launch(const std::string& key, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
        return false;
    }
    return it->second->launch(stream);
}

bool CudaGraphManager::has_graph(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.count(key) > 0;
}

void CudaGraphManager::remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    graphs_.erase(key);
}

void CudaGraphManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    graphs_.clear();
}

size_t CudaGraphManager::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
}

std::string CudaGraphManager::make_key(
    const std::vector<std::pair<std::string, nvinfer1::Dims>>& shapes) {
    std::ostringstream oss;
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i > 0) oss << ',';
        oss << shapes[i].first << ':';
        const auto& dims = shapes[i].second;
        for (int d = 0; d < dims.nbDims; ++d) {
            if (d > 0) oss << 'x';
            oss << dims.d[d];
        }
    }
    return oss.str();
}

}  // namespace trt_engine
