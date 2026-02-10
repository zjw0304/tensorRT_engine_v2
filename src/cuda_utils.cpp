#include <trt_engine/cuda_utils.h>

#include <chrono>
#include <stdexcept>

namespace trt_engine {

// ── CudaStream ─────────────────────────────────────────────────────────────

CudaStream::CudaStream() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaStream::CudaStream(unsigned int flags) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) {
            // Log but don't throw in destructor
            get_logger().error("CudaStream::~CudaStream failed: " +
                               std::string(cudaGetErrorString(err)));
        }
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ── CudaEvent ──────────────────────────────────────────────────────────────

CudaEvent::CudaEvent() {
    CUDA_CHECK(cudaEventCreate(&event_));
}

CudaEvent::CudaEvent(unsigned int flags) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

CudaEvent::~CudaEvent() {
    if (event_) {
        cudaError_t err = cudaEventDestroy(event_);
        if (err != cudaSuccess) {
            get_logger().error("CudaEvent::~CudaEvent failed: " +
                               std::string(cudaGetErrorString(err)));
        }
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept
    : event_(other.event_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_) {
            cudaEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void CudaEvent::record(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaEvent::synchronize() {
    CUDA_CHECK(cudaEventSynchronize(event_));
}

float CudaEvent::elapsed_time(const CudaEvent& start, const CudaEvent& end) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
    return ms;
}

// ── StreamPool ─────────────────────────────────────────────────────────────

StreamPool::StreamPool(size_t pool_size) : total_size_(pool_size) {
    for (size_t i = 0; i < pool_size; ++i) {
        available_streams_.push(std::make_shared<CudaStream>());
    }
}

StreamPool::~StreamPool() {
    // shared_ptrs clean up automatically
}

std::shared_ptr<CudaStream> StreamPool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (available_streams_.empty()) {
        // Create a new stream if pool is exhausted
        get_logger().warning("StreamPool exhausted, creating additional stream");
        return std::make_shared<CudaStream>();
    }
    auto stream = available_streams_.front();
    available_streams_.pop();
    return stream;
}

void StreamPool::release(std::shared_ptr<CudaStream> stream) {
    if (!stream) return;
    std::lock_guard<std::mutex> lock(mutex_);
    available_streams_.push(std::move(stream));
}

size_t StreamPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_streams_.size();
}

// ── Async memcpy helpers ───────────────────────────────────────────────────

void async_memcpy_h2d(void* dst, const void* src, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
}

void async_memcpy_d2h(void* dst, const void* src, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
}

// ── Device query functions ─────────────────────────────────────────────────

int get_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

DeviceProperties get_device_properties(int device_id) {
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    DeviceProperties dp;
    dp.name                      = props.name;
    dp.compute_capability_major  = props.major;
    dp.compute_capability_minor  = props.minor;
    dp.total_global_memory       = props.totalGlobalMem;
    dp.multi_processor_count     = props.multiProcessorCount;
    dp.max_threads_per_block     = props.maxThreadsPerBlock;
    dp.shared_memory_per_block   = props.sharedMemPerBlock;
    dp.warp_size                 = props.warpSize;
    dp.clock_rate_khz            = props.clockRate;
    dp.memory_clock_rate_khz     = props.memoryClockRate;
    dp.memory_bus_width_bits     = props.memoryBusWidth;

    return dp;
}

// ── Spin-wait synchronization ───────────────────────────────────────────

void spin_wait_event(const CudaEvent& event) {
    while (true) {
        cudaError_t status = cudaEventQuery(event.get());
        if (status == cudaSuccess) {
            return;
        }
        if (status != cudaErrorNotReady) {
            CUDA_CHECK(status);
        }
    }
}

void hybrid_wait_event(const CudaEvent& event, cudaStream_t stream,
                       uint64_t spin_ns) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        cudaError_t status = cudaEventQuery(event.get());
        if (status == cudaSuccess) {
            return;
        }
        if (status != cudaErrorNotReady) {
            CUDA_CHECK(status);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                    .count()) >= spin_ns) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return;
        }
    }
}

void sync_stream(CudaStream& stream, CudaEvent& event, SyncMode mode,
                 uint64_t hybrid_spin_ns) {
    switch (mode) {
        case SyncMode::BLOCKING:
            stream.synchronize();
            break;
        case SyncMode::SPIN_WAIT:
            spin_wait_event(event);
            break;
        case SyncMode::HYBRID:
            hybrid_wait_event(event, stream.get(), hybrid_spin_ns);
            break;
    }
}

}  // namespace trt_engine
