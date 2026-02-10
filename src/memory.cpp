#include <trt_engine/memory.h>

#include <cstring>
#include <stdexcept>

namespace trt_engine {

// ── GpuAllocator ───────────────────────────────────────────────────────────

GpuAllocator::GpuAllocator() = default;

GpuAllocator::~GpuAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [ptr, info] : allocations_) {
        cudaFree(ptr);
    }
    allocations_.clear();
}

void* GpuAllocator::allocate(uint64_t const size, uint64_t const /*alignment*/, nvinfer1::AllocatorFlags const /*flags*/) noexcept {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        get_logger().error("GpuAllocator::allocate failed: " +
                           std::string(cudaGetErrorString(err)));
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = AllocationInfo{static_cast<size_t>(size)};
    total_allocated_ += size;
    if (total_allocated_ > peak_allocated_) {
        peak_allocated_ = total_allocated_;
    }
    ++allocation_count_;

    return ptr;
}

bool GpuAllocator::deallocate(void* const memory) noexcept {
    if (!memory) return true;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(memory);
    if (it == allocations_.end()) {
        get_logger().warning("GpuAllocator::deallocate called on unknown pointer");
        return false;
    }

    total_allocated_ -= it->second.size;
    allocations_.erase(it);

    cudaError_t err = cudaFree(memory);
    if (err != cudaSuccess) {
        get_logger().error("GpuAllocator::deallocate cudaFree failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }
    return true;
}

size_t GpuAllocator::get_total_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

size_t GpuAllocator::get_peak_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_allocated_;
}

size_t GpuAllocator::get_allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocation_count_;
}

void GpuAllocator::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_allocated_ = total_allocated_;
    allocation_count_ = 0;
}

// ── DeviceBuffer ───────────────────────────────────────────────────────────

DeviceBuffer::DeviceBuffer() = default;

DeviceBuffer::DeviceBuffer(size_t size) {
    allocate(size);
}

DeviceBuffer::~DeviceBuffer() {
    free();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DeviceBuffer::allocate(size_t size) {
    free();
    if (size == 0) return;
    CUDA_CHECK(cudaMalloc(&ptr_, size));
    size_ = size;
}

void DeviceBuffer::free() {
    if (ptr_) {
        cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            get_logger().error("DeviceBuffer::free failed: " +
                               std::string(cudaGetErrorString(err)));
        }
        ptr_ = nullptr;
        size_ = 0;
    }
}

// ── PinnedBuffer ───────────────────────────────────────────────────────────

PinnedBuffer::PinnedBuffer() = default;

PinnedBuffer::PinnedBuffer(size_t size) {
    allocate(size);
}

PinnedBuffer::~PinnedBuffer() {
    free();
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        free();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void PinnedBuffer::allocate(size_t size) {
    free();
    if (size == 0) return;
    CUDA_CHECK(cudaMallocHost(&ptr_, size));
    size_ = size;
}

void PinnedBuffer::free() {
    if (ptr_) {
        cudaError_t err = cudaFreeHost(ptr_);
        if (err != cudaSuccess) {
            get_logger().error("PinnedBuffer::free failed: " +
                               std::string(cudaGetErrorString(err)));
        }
        ptr_ = nullptr;
        size_ = 0;
    }
}

// ── MemoryManager ──────────────────────────────────────────────────────────

MemoryManager::MemoryManager() = default;
MemoryManager::~MemoryManager() = default;

std::unique_ptr<DeviceBuffer> MemoryManager::allocate_device(size_t size) {
    auto buf = std::make_unique<DeviceBuffer>(size);

    std::lock_guard<std::mutex> lock(mutex_);
    total_device_allocated_ += size;
    if (total_device_allocated_ > peak_device_allocated_) {
        peak_device_allocated_ = total_device_allocated_;
    }
    ++device_alloc_count_;

    return buf;
}

std::unique_ptr<PinnedBuffer> MemoryManager::allocate_pinned(size_t size) {
    auto buf = std::make_unique<PinnedBuffer>(size);

    std::lock_guard<std::mutex> lock(mutex_);
    total_pinned_allocated_ += size;
    if (total_pinned_allocated_ > peak_pinned_allocated_) {
        peak_pinned_allocated_ = total_pinned_allocated_;
    }
    ++pinned_alloc_count_;

    return buf;
}

size_t MemoryManager::get_total_device_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_device_allocated_;
}

size_t MemoryManager::get_total_pinned_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_pinned_allocated_;
}

size_t MemoryManager::get_peak_device_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_device_allocated_;
}

size_t MemoryManager::get_peak_pinned_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return peak_pinned_allocated_;
}

size_t MemoryManager::get_device_allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_alloc_count_;
}

size_t MemoryManager::get_pinned_allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pinned_alloc_count_;
}

void MemoryManager::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    peak_device_allocated_ = total_device_allocated_;
    peak_pinned_allocated_ = total_pinned_allocated_;
    device_alloc_count_ = 0;
    pinned_alloc_count_ = 0;
}

// ── Async copy helpers ─────────────────────────────────────────────────────

void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
}

void copy_to_host(void* dst, const void* src, size_t size, cudaStream_t stream) {
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
}

}  // namespace trt_engine
