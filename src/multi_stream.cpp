#include <trt_engine/multi_stream.h>

#include <fstream>
#include <stdexcept>

namespace trt_engine {

// ── Helpers ─────────────────────────────────────────────────────────────

static std::vector<char> load_engine_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw EngineException("MultiStreamEngine: failed to open engine file: " + path);
    }
    auto size = file.tellg();
    if (size <= 0) {
        throw EngineException("MultiStreamEngine: engine file is empty: " + path);
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(data.data(), size);
    return data;
}

int64_t MultiStreamEngine::volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) v *= dims.d[i];
    return v;
}

// ── Constructor / Destructor ────────────────────────────────────────────

MultiStreamEngine::MultiStreamEngine(const std::string& engine_path,
                                     int num_streams,
                                     const EngineConfig& config)
    : num_streams_(num_streams), config_(config) {

    if (num_streams_ < 1) {
        throw EngineException("MultiStreamEngine: num_streams must be >= 1");
    }

    CUDA_CHECK(cudaSetDevice(config_.device_id));

    auto engine_data = load_engine_file(engine_path);

    runtime_.reset(nvinfer1::createInferRuntime(get_logger()));
    if (!runtime_) {
        throw EngineException("MultiStreamEngine: failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(
        engine_data.data(), engine_data.size()));
    if (!engine_) {
        throw EngineException("MultiStreamEngine: failed to deserialize engine");
    }

    get_logger().info("MultiStreamEngine: loaded engine with " +
                      std::to_string(num_streams_) + " streams");

    // Launch worker threads
    for (int i = 0; i < num_streams_; ++i) {
        workers_.emplace_back(&MultiStreamEngine::worker_loop, this, i);
    }
}

MultiStreamEngine::~MultiStreamEngine() {
    shutdown();
}

// ── Shutdown ────────────────────────────────────────────────────────────

void MultiStreamEngine::shutdown() {
    if (shutdown_.exchange(true)) return;  // already shut down

    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    get_logger().info("MultiStreamEngine: shutdown complete");
}

// ── Public interface ────────────────────────────────────────────────────

InferenceResult MultiStreamEngine::infer(
    const std::vector<std::vector<float>>& input_buffers) {
    return submit(input_buffers).get();
}

std::future<InferenceResult> MultiStreamEngine::submit(
    const std::vector<std::vector<float>>& input_buffers) {

    auto req = std::make_shared<InferRequest>();
    req->inputs = input_buffers;
    auto future = req->promise.get_future();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (shutdown_.load()) {
            req->promise.set_value(InferenceResult{
                {}, 0.0f, false, "MultiStreamEngine is shut down"});
            return future;
        }
        request_queue_.push(std::move(req));
    }
    queue_cv_.notify_one();
    return future;
}

// ── Worker loop ─────────────────────────────────────────────────────────

void MultiStreamEngine::worker_loop(int worker_id) {
    try {
        CUDA_CHECK(cudaSetDevice(config_.device_id));
    } catch (...) {
        get_logger().error("MultiStreamEngine worker " +
                           std::to_string(worker_id) + ": failed to set device");
        return;
    }

    // Each worker has its own context and stream
    UniqueContext ctx(engine_->createExecutionContext());
    if (!ctx) {
        get_logger().error("MultiStreamEngine worker " +
                           std::to_string(worker_id) +
                           ": failed to create execution context");
        return;
    }

    CudaStream stream;
    CudaEvent start_event, end_event;

    while (true) {
        std::shared_ptr<InferRequest> req;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]{
                return shutdown_.load() || !request_queue_.empty();
            });
            if (shutdown_.load() && request_queue_.empty()) return;
            req = std::move(request_queue_.front());
            request_queue_.pop();
        }

        InferenceResult result;
        result.success = false;

        try {
            // Gather I/O tensor names
            int nb_io = engine_->getNbIOTensors();
            std::vector<std::string> input_names, output_names;
            for (int i = 0; i < nb_io; ++i) {
                const char* name = engine_->getIOTensorName(i);
                if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
                    input_names.emplace_back(name);
                else
                    output_names.emplace_back(name);
            }

            if (req->inputs.size() != input_names.size()) {
                result.error_msg = "Input count mismatch";
                req->promise.set_value(std::move(result));
                continue;
            }

            // Allocate and copy inputs
            std::vector<DeviceBuffer> in_bufs;
            in_bufs.reserve(input_names.size());
            for (size_t i = 0; i < input_names.size(); ++i) {
                size_t bytes = req->inputs[i].size() * sizeof(float);
                in_bufs.emplace_back(bytes);
                async_memcpy_h2d(in_bufs[i].data(), req->inputs[i].data(),
                                 bytes, stream.get());
                ctx->setTensorAddress(input_names[i].c_str(), in_bufs[i].data());
            }

            // Allocate outputs
            std::vector<DeviceBuffer> out_bufs;
            std::vector<size_t> out_elem_counts;
            out_bufs.reserve(output_names.size());
            for (const auto& name : output_names) {
                nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
                int64_t vol = volume(dims);
                if (vol <= 0) {
                    dims = engine_->getTensorShape(name.c_str());
                    vol = volume(dims);
                }
                nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
                size_t bytes = static_cast<size_t>(vol) * datatype_size(dt);
                out_bufs.emplace_back(bytes);
                out_elem_counts.push_back(static_cast<size_t>(vol));
                ctx->setTensorAddress(name.c_str(), out_bufs.back().data());
            }

            start_event.record(stream.get());

            if (!ctx->enqueueV3(stream.get())) {
                result.error_msg = "enqueueV3 failed on worker " +
                                   std::to_string(worker_id);
                req->promise.set_value(std::move(result));
                continue;
            }

            end_event.record(stream.get());

            // Copy outputs to host
            result.outputs.resize(output_names.size());
            for (size_t i = 0; i < output_names.size(); ++i) {
                result.outputs[i].resize(out_elem_counts[i]);
                async_memcpy_d2h(result.outputs[i].data(), out_bufs[i].data(),
                                 out_elem_counts[i] * sizeof(float), stream.get());
            }

            stream.synchronize();
            result.latency_ms = CudaEvent::elapsed_time(start_event, end_event);
            result.success = true;

        } catch (const std::exception& e) {
            result.error_msg = e.what();
            result.success = false;
        }

        req->promise.set_value(std::move(result));
    }
}

}  // namespace trt_engine
