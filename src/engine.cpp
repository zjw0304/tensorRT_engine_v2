#include <trt_engine/engine.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace trt_engine {

// ── Helpers ─────────────────────────────────────────────────────────────

static std::vector<char> read_engine_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw EngineException("Failed to open engine file: " + path);
    }
    auto size = file.tellg();
    if (size <= 0) {
        throw EngineException("Engine file is empty: " + path);
    }
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(data.data(), size);
    return data;
}

// ── Factory methods ─────────────────────────────────────────────────────

std::unique_ptr<InferenceEngine> InferenceEngine::create(
    const std::string& engine_path, const EngineConfig& config) {
    auto data = read_engine_file(engine_path);
    return std::unique_ptr<InferenceEngine>(new InferenceEngine(data, config));
}

std::unique_ptr<InferenceEngine> InferenceEngine::create(
    const std::vector<char>& engine_data, const EngineConfig& config) {
    return std::unique_ptr<InferenceEngine>(new InferenceEngine(engine_data, config));
}

// ── Constructor / Destructor ────────────────────────────────────────────

InferenceEngine::InferenceEngine(const std::vector<char>& engine_data,
                                 const EngineConfig& config)
    : config_(config) {
    CUDA_CHECK(cudaSetDevice(config_.device_id));

    // Create TensorRT runtime
    runtime_.reset(nvinfer1::createInferRuntime(get_logger()));
    if (!runtime_) {
        throw EngineException("Failed to create TensorRT runtime");
    }

    // Deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(
        engine_data.data(), engine_data.size()));
    if (!engine_) {
        throw EngineException("Failed to deserialize TensorRT engine");
    }

    get_logger().info("Engine loaded: " +
                      std::to_string(engine_->getNbIOTensors()) + " I/O tensors");

    init_context_pool();
    init_thread_pool();
}

InferenceEngine::~InferenceEngine() {
    shutdown_thread_pool();
}

// ── Context pool ────────────────────────────────────────────────────────

void InferenceEngine::init_context_pool() {
    for (int i = 0; i < config_.context_pool_size; ++i) {
        UniqueContext ctx(engine_->createExecutionContext());
        if (!ctx) {
            throw EngineException("Failed to create execution context " +
                                  std::to_string(i));
        }
        ctx_pool_.push(std::move(ctx));
    }
}

UniqueContext InferenceEngine::acquire_context() {
    std::unique_lock<std::mutex> lock(ctx_mutex_);
    ctx_cv_.wait(lock, [this]{ return !ctx_pool_.empty(); });
    auto ctx = std::move(ctx_pool_.front());
    ctx_pool_.pop();
    return ctx;
}

void InferenceEngine::release_context(UniqueContext ctx) {
    {
        std::lock_guard<std::mutex> lock(ctx_mutex_);
        ctx_pool_.push(std::move(ctx));
    }
    ctx_cv_.notify_one();
}

// ── Thread pool ─────────────────────────────────────────────────────────

void InferenceEngine::init_thread_pool() {
    for (int i = 0; i < config_.thread_pool_size; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::packaged_task<InferenceResult()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    queue_cv_.wait(lock, [this]{
                        return shutdown_.load() || !task_queue_.empty();
                    });
                    if (shutdown_.load() && task_queue_.empty()) return;
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                task();
            }
        });
    }
}

void InferenceEngine::shutdown_thread_pool() {
    shutdown_.store(true);
    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

// ── Volume helper ───────────────────────────────────────────────────────

int64_t InferenceEngine::volume(const nvinfer1::Dims& dims) {
    int64_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        vol *= dims.d[i];
    }
    return vol;
}

// ── Tensor info queries ─────────────────────────────────────────────────

std::vector<TensorInfo> InferenceEngine::get_input_info() const {
    std::vector<TensorInfo> infos;
    int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
            continue;

        TensorInfo ti;
        ti.name = name;
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        ti.shape.assign(dims.d, dims.d + dims.nbDims);

        nvinfer1::DataType dt = engine_->getTensorDataType(name);
        ti.size_bytes = static_cast<size_t>(volume(dims)) * datatype_size(dt);

        switch (dt) {
            case nvinfer1::DataType::kFLOAT: ti.dtype = Precision::FP32; break;
            case nvinfer1::DataType::kHALF:  ti.dtype = Precision::FP16; break;
            case nvinfer1::DataType::kINT8:  ti.dtype = Precision::INT8; break;
            case nvinfer1::DataType::kFP8:   ti.dtype = Precision::FP8;  break;
            default:                         ti.dtype = Precision::FP32; break;
        }
        infos.push_back(std::move(ti));
    }
    return infos;
}

std::vector<TensorInfo> InferenceEngine::get_output_info() const {
    std::vector<TensorInfo> infos;
    int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT)
            continue;

        TensorInfo ti;
        ti.name = name;
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        ti.shape.assign(dims.d, dims.d + dims.nbDims);

        nvinfer1::DataType dt = engine_->getTensorDataType(name);
        ti.size_bytes = static_cast<size_t>(volume(dims)) * datatype_size(dt);

        switch (dt) {
            case nvinfer1::DataType::kFLOAT: ti.dtype = Precision::FP32; break;
            case nvinfer1::DataType::kHALF:  ti.dtype = Precision::FP16; break;
            case nvinfer1::DataType::kINT8:  ti.dtype = Precision::INT8; break;
            case nvinfer1::DataType::kFP8:   ti.dtype = Precision::FP8;  break;
            default:                         ti.dtype = Precision::FP32; break;
        }
        infos.push_back(std::move(ti));
    }
    return infos;
}

// ── Dynamic shape ───────────────────────────────────────────────────────

void InferenceEngine::set_input_shape(const std::string& name,
                                      const std::vector<int>& dims) {
    std::lock_guard<std::mutex> lock(shape_mutex_);
    shape_overrides_[name] = dims;
    // Invalidate prepared buffers when shapes change
    prepared_.ready = false;
}

// ── Buffer pre-allocation ───────────────────────────────────────────────

void InferenceEngine::prepare_buffers() {
    CUDA_CHECK(cudaSetDevice(config_.device_id));

    prepared_.input_names.clear();
    prepared_.output_names.clear();
    prepared_.input_device_bufs.clear();
    prepared_.output_device_bufs.clear();
    prepared_.input_pinned_bufs.clear();
    prepared_.input_byte_sizes.clear();
    prepared_.output_elem_counts.clear();

    // Create persistent stream and events if not already created
    if (!prepared_.stream) {
        prepared_.stream = std::make_unique<CudaStream>();
        prepared_.start_event = std::make_unique<CudaEvent>();
        prepared_.end_event = std::make_unique<CudaEvent>();
    }

    // Collect tensor names
    int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            prepared_.input_names.emplace_back(name);
        } else {
            prepared_.output_names.emplace_back(name);
        }
    }

    // We need a temporary context to resolve output shapes
    auto ctx = acquire_context();

    // Apply shape overrides to context and cache them
    prepared_.cached_shapes.clear();
    {
        std::lock_guard<std::mutex> slock(shape_mutex_);
        for (auto& [name, dims] : shape_overrides_) {
            nvinfer1::Dims trt_dims;
            trt_dims.nbDims = static_cast<int>(dims.size());
            for (int d = 0; d < trt_dims.nbDims; ++d) {
                trt_dims.d[d] = dims[d];
            }
            ctx->setInputShape(name.c_str(), trt_dims);
            prepared_.cached_shapes.emplace_back(name, trt_dims);
        }
    }

    // Allocate input device buffers and pinned host buffers
    for (const auto& name : prepared_.input_names) {
        // Compute size from shape overrides
        std::lock_guard<std::mutex> slock(shape_mutex_);
        auto it = shape_overrides_.find(name);
        size_t byte_size;
        if (it != shape_overrides_.end()) {
            int64_t vol = 1;
            for (int d : it->second) vol *= d;
            nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
            byte_size = static_cast<size_t>(vol) * datatype_size(dt);
        } else {
            nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());
            int64_t vol = volume(dims);
            nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
            byte_size = static_cast<size_t>(vol) * datatype_size(dt);
        }
        prepared_.input_byte_sizes.push_back(byte_size);
        prepared_.input_device_bufs.emplace_back(byte_size);
        prepared_.input_pinned_bufs.emplace_back(byte_size);
    }

    // Allocate output device buffers
    for (const auto& name : prepared_.output_names) {
        nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
        int64_t vol = volume(dims);
        if (vol <= 0) {
            dims = engine_->getTensorShape(name.c_str());
            vol = volume(dims);
        }
        nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
        size_t byte_size = static_cast<size_t>(vol) * datatype_size(dt);
        prepared_.output_device_bufs.emplace_back(byte_size);
        prepared_.output_elem_counts.push_back(static_cast<size_t>(vol));
    }

    release_context(std::move(ctx));
    prepared_.ready = true;
}

// ── Core inference ──────────────────────────────────────────────────────

InferenceResult InferenceEngine::run_inference(
    const std::vector<std::vector<float>>& input_buffers) {

    InferenceResult result;
    result.success = false;

    // Fast path: use pre-allocated buffers
    if (prepared_.ready) {
        try {
            if (input_buffers.size() != prepared_.input_names.size()) {
                result.error_msg = "Input count mismatch: expected " +
                                   std::to_string(prepared_.input_names.size()) + " got " +
                                   std::to_string(input_buffers.size());
                return result;
            }

            auto ctx = acquire_context();

            // Apply cached shapes (no mutex needed - snapshot from prepare_buffers)
            for (const auto& [name, trt_dims] : prepared_.cached_shapes) {
                ctx->setInputShape(name.c_str(), trt_dims);
            }

            auto stream = prepared_.stream->get();

            // Copy inputs to pinned memory, then H2D async
            for (size_t i = 0; i < prepared_.input_names.size(); ++i) {
                size_t byte_size = input_buffers[i].size() * sizeof(float);
                std::memcpy(prepared_.input_pinned_bufs[i].data(),
                            input_buffers[i].data(), byte_size);
                async_memcpy_h2d(prepared_.input_device_bufs[i].data(),
                                 prepared_.input_pinned_bufs[i].data(),
                                 byte_size, stream);
                ctx->setTensorAddress(prepared_.input_names[i].c_str(),
                                      prepared_.input_device_bufs[i].data());
            }

            // Bind outputs
            for (size_t i = 0; i < prepared_.output_names.size(); ++i) {
                ctx->setTensorAddress(prepared_.output_names[i].c_str(),
                                      prepared_.output_device_bufs[i].data());
            }

            // Measure and run
            prepared_.start_event->record(stream);

            if (!ctx->enqueueV3(stream)) {
                result.error_msg = "enqueueV3 failed";
                release_context(std::move(ctx));
                return result;
            }

            prepared_.end_event->record(stream);

            // Copy outputs back to host
            result.outputs.resize(prepared_.output_names.size());
            for (size_t i = 0; i < prepared_.output_names.size(); ++i) {
                result.outputs[i].resize(prepared_.output_elem_counts[i]);
                async_memcpy_d2h(result.outputs[i].data(),
                                 prepared_.output_device_bufs[i].data(),
                                 prepared_.output_elem_counts[i] * sizeof(float),
                                 stream);
            }

            prepared_.stream->synchronize();
            result.latency_ms = CudaEvent::elapsed_time(*prepared_.start_event,
                                                         *prepared_.end_event);
            result.success = true;

            release_context(std::move(ctx));
        } catch (const std::exception& e) {
            result.error_msg = e.what();
            result.success = false;
        }
        return result;
    }

    // Slow path: allocate everything per call (original behavior)

    try {
        CUDA_CHECK(cudaSetDevice(config_.device_id));

        auto ctx = acquire_context();
        CudaStream stream;
        CudaEvent start_event, end_event;

        // Collect input/output tensor names
        int nb_io = engine_->getNbIOTensors();
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;

        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                input_names.emplace_back(name);
            } else {
                output_names.emplace_back(name);
            }
        }

        if (input_buffers.size() != input_names.size()) {
            result.error_msg = "Input count mismatch: expected " +
                               std::to_string(input_names.size()) + " got " +
                               std::to_string(input_buffers.size());
            release_context(std::move(ctx));
            return result;
        }

        // Apply shape overrides
        {
            std::lock_guard<std::mutex> slock(shape_mutex_);
            for (auto& [name, dims] : shape_overrides_) {
                nvinfer1::Dims trt_dims;
                trt_dims.nbDims = static_cast<int>(dims.size());
                for (int d = 0; d < trt_dims.nbDims; ++d) {
                    trt_dims.d[d] = dims[d];
                }
                if (!ctx->setInputShape(name.c_str(), trt_dims)) {
                    result.error_msg = "Failed to set input shape for: " + name;
                    release_context(std::move(ctx));
                    return result;
                }
            }
        }

        // Allocate device buffers for inputs and bind
        std::vector<DeviceBuffer> input_device_bufs;
        input_device_bufs.reserve(input_names.size());

        for (size_t i = 0; i < input_names.size(); ++i) {
            const auto& name = input_names[i];
            size_t byte_size = input_buffers[i].size() * sizeof(float);
            input_device_bufs.emplace_back(byte_size);

            // Copy input to device
            async_memcpy_h2d(input_device_bufs[i].data(),
                             input_buffers[i].data(),
                             byte_size, stream.get());

            if (!ctx->setTensorAddress(name.c_str(), input_device_bufs[i].data())) {
                result.error_msg = "Failed to set input tensor address: " + name;
                release_context(std::move(ctx));
                return result;
            }
        }

        // Allocate device buffers for outputs and bind
        std::vector<DeviceBuffer> output_device_bufs;
        std::vector<size_t> output_elem_counts;
        output_device_bufs.reserve(output_names.size());
        output_elem_counts.reserve(output_names.size());

        for (const auto& name : output_names) {
            nvinfer1::Dims dims = ctx->getTensorShape(name.c_str());
            int64_t vol = volume(dims);
            if (vol <= 0) {
                // If shape is not yet resolved, use engine profile max shape
                dims = engine_->getTensorShape(name.c_str());
                vol = volume(dims);
            }
            nvinfer1::DataType dt = engine_->getTensorDataType(name.c_str());
            size_t byte_size = static_cast<size_t>(vol) * datatype_size(dt);
            output_device_bufs.emplace_back(byte_size);
            output_elem_counts.push_back(static_cast<size_t>(vol));

            if (!ctx->setTensorAddress(name.c_str(), output_device_bufs.back().data())) {
                result.error_msg = "Failed to set output tensor address: " + name;
                release_context(std::move(ctx));
                return result;
            }
        }

        // Measure latency
        start_event.record(stream.get());

        // Run inference
        if (!ctx->enqueueV3(stream.get())) {
            result.error_msg = "enqueueV3 failed";
            release_context(std::move(ctx));
            return result;
        }

        end_event.record(stream.get());

        // Copy outputs back to host
        result.outputs.resize(output_names.size());
        for (size_t i = 0; i < output_names.size(); ++i) {
            result.outputs[i].resize(output_elem_counts[i]);
            async_memcpy_d2h(result.outputs[i].data(),
                             output_device_bufs[i].data(),
                             output_elem_counts[i] * sizeof(float),
                             stream.get());
        }

        // Synchronize and compute latency
        stream.synchronize();
        result.latency_ms = CudaEvent::elapsed_time(start_event, end_event);
        result.success = true;

        release_context(std::move(ctx));
    } catch (const std::exception& e) {
        result.error_msg = e.what();
        result.success = false;
    }

    return result;
}

InferenceResult InferenceEngine::infer(
    const std::vector<std::vector<float>>& input_buffers) {
    return run_inference(input_buffers);
}

std::future<InferenceResult> InferenceEngine::infer_async(
    const std::vector<std::vector<float>>& input_buffers) {
    auto task = std::packaged_task<InferenceResult()>(
        [this, input_buffers]() -> InferenceResult {
            return run_inference(input_buffers);
        });
    auto future = task.get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
    return future;
}

// ── Warmup ──────────────────────────────────────────────────────────────

void InferenceEngine::warmup(int iterations) {
    get_logger().info("Running " + std::to_string(iterations) + " warmup iterations");

    // Build dummy inputs from engine's input specs
    auto inputs = get_input_info();
    std::vector<std::vector<float>> dummy_inputs;
    dummy_inputs.reserve(inputs.size());

    for (auto& ti : inputs) {
        int64_t vol = 1;
        for (int d : ti.shape) {
            // Dynamic dims (-1) are treated as 1 for warmup
            vol *= (d > 0) ? d : 1;
        }
        dummy_inputs.emplace_back(static_cast<size_t>(vol), 0.0f);
    }

    for (int i = 0; i < iterations; ++i) {
        auto res = infer(dummy_inputs);
        if (!res.success) {
            get_logger().warning("Warmup iteration " + std::to_string(i) +
                                 " failed: " + res.error_msg);
        }
    }

    get_logger().info("Warmup complete");
}

}  // namespace trt_engine
