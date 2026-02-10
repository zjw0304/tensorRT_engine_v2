#pragma once

#include <cuda_runtime.h>
#include <trt_engine/logger.h>
#include <trt_engine/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace trt_engine {

// ── RAII CUDA stream ───────────────────────────────────────────────────────
class CudaStream {
public:
    CudaStream();
    explicit CudaStream(unsigned int flags);
    ~CudaStream();

    // Not copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Movable
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

    void synchronize();

private:
    cudaStream_t stream_ = nullptr;
};

// ── RAII CUDA event ────────────────────────────────────────────────────────
class CudaEvent {
public:
    enum class Type { TIMING, SYNC_ONLY };

    CudaEvent();
    explicit CudaEvent(Type type);
    explicit CudaEvent(unsigned int flags);
    ~CudaEvent();

    // Not copyable
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Movable
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }

    void record(cudaStream_t stream = nullptr);
    void synchronize();

    // Compute elapsed time in ms between start and this event
    static float elapsed_time(const CudaEvent& start, const CudaEvent& end);

private:
    cudaEvent_t event_ = nullptr;
};

// ── Stream pool ────────────────────────────────────────────────────────────
class StreamPool {
public:
    explicit StreamPool(size_t pool_size = 4);
    ~StreamPool();

    // Not copyable or movable
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    // Acquire a stream from the pool (blocks if none available)
    std::shared_ptr<CudaStream> acquire();

    // Release a stream back to the pool
    void release(std::shared_ptr<CudaStream> stream);

    size_t pool_size() const { return total_size_; }
    size_t available() const;

private:
    mutable std::mutex mutex_;
    std::queue<std::shared_ptr<CudaStream>> available_streams_;
    size_t total_size_;
};

// ── Spin-wait synchronization ────────────────────────────────────────────
void spin_wait_event(const CudaEvent& event);
void hybrid_wait_event(const CudaEvent& event, cudaStream_t stream,
                       uint64_t spin_ns = 100000);
void sync_stream(CudaStream& stream, CudaEvent& event, SyncMode mode,
                 uint64_t hybrid_spin_ns = 100000);

// ── Async memcpy helpers ───────────────────────────────────────────────────
void async_memcpy_h2d(void* dst, const void* src, size_t size, cudaStream_t stream);
void async_memcpy_d2h(void* dst, const void* src, size_t size, cudaStream_t stream);

// ── Device query functions ─────────────────────────────────────────────────
int get_device_count();
DeviceProperties get_device_properties(int device_id);

}  // namespace trt_engine
