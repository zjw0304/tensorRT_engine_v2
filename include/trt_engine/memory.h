#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <trt_engine/logger.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace trt_engine {

// ── Custom GPU allocator for TensorRT ──────────────────────────────────────
class GpuAllocator : public nvinfer1::IGpuAllocator {
public:
    GpuAllocator();
    ~GpuAllocator() override;

    // nvinfer1::IGpuAllocator interface
    void* allocate(uint64_t const size, uint64_t const alignment, nvinfer1::AllocatorFlags const flags) noexcept override;
    bool  deallocate(void* const memory) noexcept override;

    size_t get_total_allocated() const;
    size_t get_peak_allocated() const;
    size_t get_allocation_count() const;

    void reset_stats();

private:
    mutable std::mutex mutex_;

    struct AllocationInfo {
        size_t size;
    };

    std::unordered_map<void*, AllocationInfo> allocations_;
    size_t total_allocated_    = 0;
    size_t peak_allocated_     = 0;
    size_t allocation_count_   = 0;
};

// ── RAII GPU device memory ─────────────────────────────────────────────────
class DeviceBuffer {
public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t size);
    ~DeviceBuffer();

    // Not copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void allocate(size_t size);
    void free();

    void*       data()       { return ptr_; }
    const void* data() const { return ptr_; }
    size_t      size() const { return size_; }
    bool        empty() const { return ptr_ == nullptr; }

    // Typed access
    template <typename T>
    T* as() { return static_cast<T*>(ptr_); }

    template <typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }

private:
    void*  ptr_  = nullptr;
    size_t size_ = 0;
};

// ── RAII pinned host memory ────────────────────────────────────────────────
class PinnedBuffer {
public:
    PinnedBuffer();
    explicit PinnedBuffer(size_t size);
    ~PinnedBuffer();

    // Not copyable
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Movable
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void allocate(size_t size);
    void free();

    void*       data()       { return ptr_; }
    const void* data() const { return ptr_; }
    size_t      size() const { return size_; }
    bool        empty() const { return ptr_ == nullptr; }

    template <typename T>
    T* as() { return static_cast<T*>(ptr_); }

    template <typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }

private:
    void*  ptr_  = nullptr;
    size_t size_ = 0;
};

// ── Memory manager ─────────────────────────────────────────────────────────
class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Allocate a device buffer and track it
    std::unique_ptr<DeviceBuffer> allocate_device(size_t size);

    // Allocate a pinned buffer and track it
    std::unique_ptr<PinnedBuffer> allocate_pinned(size_t size);

    // Typed allocation helpers
    template <typename T>
    std::unique_ptr<DeviceBuffer> allocate_device_typed(size_t count) {
        return allocate_device(count * sizeof(T));
    }

    template <typename T>
    std::unique_ptr<PinnedBuffer> allocate_pinned_typed(size_t count) {
        return allocate_pinned(count * sizeof(T));
    }

    // Stats
    size_t get_total_device_allocated() const;
    size_t get_total_pinned_allocated() const;
    size_t get_peak_device_allocated() const;
    size_t get_peak_pinned_allocated() const;
    size_t get_device_allocation_count() const;
    size_t get_pinned_allocation_count() const;

    void reset_stats();

    // Get the custom GPU allocator (for passing to TensorRT)
    GpuAllocator& get_gpu_allocator() { return gpu_allocator_; }

private:
    mutable std::mutex mutex_;
    GpuAllocator       gpu_allocator_;

    size_t total_device_allocated_ = 0;
    size_t peak_device_allocated_  = 0;
    size_t device_alloc_count_     = 0;

    size_t total_pinned_allocated_ = 0;
    size_t peak_pinned_allocated_  = 0;
    size_t pinned_alloc_count_     = 0;
};

// ── Async copy helpers ─────────────────────────────────────────────────────
void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);
void copy_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);

}  // namespace trt_engine
