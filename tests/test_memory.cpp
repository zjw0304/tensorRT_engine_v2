// test_memory.cpp - Unit tests for memory management classes
//
// Tests DeviceBuffer, PinnedBuffer, MemoryManager, and copy operations.
// Tests that require a real CUDA GPU are guarded by runtime checks.

#include <gtest/gtest.h>
#include <trt_engine/memory.h>

#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace trt_engine;

// ── Helper: check if CUDA is available ──────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// ── DeviceBuffer tests ──────────────────────────────────────────────────────

TEST(DeviceBufferTest, DefaultConstruction) {
    DeviceBuffer buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
}

TEST(DeviceBufferTest, AllocateAndFree) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    DeviceBuffer buf(1024);
    EXPECT_FALSE(buf.empty());
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 1024u);

    buf.free();
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
}

TEST(DeviceBufferTest, MoveConstruction) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    DeviceBuffer buf1(512);
    void* original_ptr = buf1.data();
    size_t original_size = buf1.size();

    DeviceBuffer buf2(std::move(buf1));
    EXPECT_EQ(buf2.data(), original_ptr);
    EXPECT_EQ(buf2.size(), original_size);
    EXPECT_TRUE(buf1.empty());
}

TEST(DeviceBufferTest, MoveAssignment) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    DeviceBuffer buf1(256);
    DeviceBuffer buf2;
    void* ptr1 = buf1.data();

    buf2 = std::move(buf1);
    EXPECT_EQ(buf2.data(), ptr1);
    EXPECT_TRUE(buf1.empty());
}

TEST(DeviceBufferTest, Reallocate) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    DeviceBuffer buf(128);
    EXPECT_EQ(buf.size(), 128u);

    buf.allocate(256);
    EXPECT_EQ(buf.size(), 256u);
}

TEST(DeviceBufferTest, ZeroSizeAllocate) {
    DeviceBuffer buf;
    buf.allocate(0);
    EXPECT_TRUE(buf.empty());
}

TEST(DeviceBufferTest, TypedAccess) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    DeviceBuffer buf(sizeof(float) * 10);
    EXPECT_NE(buf.as<float>(), nullptr);
    EXPECT_NE(static_cast<const DeviceBuffer&>(buf).as<float>(), nullptr);
}

// ── PinnedBuffer tests ──────────────────────────────────────────────────────

TEST(PinnedBufferTest, DefaultConstruction) {
    PinnedBuffer buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 0u);
}

TEST(PinnedBufferTest, AllocateAndFree) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    PinnedBuffer buf(2048);
    EXPECT_FALSE(buf.empty());
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 2048u);

    buf.free();
    EXPECT_TRUE(buf.empty());
}

TEST(PinnedBufferTest, MoveConstruction) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    PinnedBuffer buf1(1024);
    void* ptr = buf1.data();

    PinnedBuffer buf2(std::move(buf1));
    EXPECT_EQ(buf2.data(), ptr);
    EXPECT_TRUE(buf1.empty());
}

TEST(PinnedBufferTest, MoveAssignment) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    PinnedBuffer buf1(512);
    PinnedBuffer buf2;

    buf2 = std::move(buf1);
    EXPECT_FALSE(buf2.empty());
    EXPECT_TRUE(buf1.empty());
}

TEST(PinnedBufferTest, ReadWriteAccess) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    PinnedBuffer buf(sizeof(float) * 4);
    float* p = buf.as<float>();
    p[0] = 1.0f;
    p[1] = 2.0f;
    p[2] = 3.0f;
    p[3] = 4.0f;

    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[3], 4.0f);
}

// ── MemoryManager tests ─────────────────────────────────────────────────────

TEST(MemoryManagerTest, AllocateDevice) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    MemoryManager mm;
    auto buf = mm.allocate_device(1024);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(buf->size(), 1024u);
    EXPECT_EQ(mm.get_total_device_allocated(), 1024u);
    EXPECT_EQ(mm.get_device_allocation_count(), 1u);
}

TEST(MemoryManagerTest, AllocatePinned) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    MemoryManager mm;
    auto buf = mm.allocate_pinned(2048);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(buf->size(), 2048u);
    EXPECT_EQ(mm.get_total_pinned_allocated(), 2048u);
    EXPECT_EQ(mm.get_pinned_allocation_count(), 1u);
}

TEST(MemoryManagerTest, PeakTracking) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    MemoryManager mm;
    auto buf1 = mm.allocate_device(1000);
    auto buf2 = mm.allocate_device(2000);

    EXPECT_EQ(mm.get_total_device_allocated(), 3000u);
    EXPECT_EQ(mm.get_peak_device_allocated(), 3000u);
    EXPECT_EQ(mm.get_device_allocation_count(), 2u);
}

TEST(MemoryManagerTest, TypedAllocation) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    MemoryManager mm;
    auto buf = mm.allocate_device_typed<float>(100);
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(buf->size(), 100 * sizeof(float));
}

TEST(MemoryManagerTest, ResetStats) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    MemoryManager mm;
    auto buf = mm.allocate_device(512);
    EXPECT_EQ(mm.get_device_allocation_count(), 1u);

    mm.reset_stats();
    EXPECT_EQ(mm.get_device_allocation_count(), 0u);
    // Peak should equal current after reset
    EXPECT_EQ(mm.get_peak_device_allocated(), mm.get_total_device_allocated());
}

// ── Copy operation tests ────────────────────────────────────────────────────

TEST(MemoryCopyTest, CopyToDeviceAndBack) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    constexpr size_t count = 256;
    std::vector<float> src(count);
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<float>(i) * 0.5f;
    }

    DeviceBuffer dev(count * sizeof(float));

    // Copy to device (synchronous)
    copy_to_device(dev.data(), src.data(), count * sizeof(float));

    // Copy back to host
    std::vector<float> dst(count, 0.0f);
    copy_to_host(dst.data(), dev.data(), count * sizeof(float));

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src[i]);
    }
}

TEST(MemoryCopyTest, PinnedCopyToDeviceAndBack) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    constexpr size_t count = 128;
    PinnedBuffer pinned(count * sizeof(float));
    float* p = pinned.as<float>();
    for (size_t i = 0; i < count; ++i) {
        p[i] = static_cast<float>(i);
    }

    DeviceBuffer dev(count * sizeof(float));
    copy_to_device(dev.data(), pinned.data(), count * sizeof(float));

    PinnedBuffer result(count * sizeof(float));
    copy_to_host(result.data(), dev.data(), count * sizeof(float));

    float* r = result.as<float>();
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(r[i], static_cast<float>(i));
    }
}

// ── GpuAllocator tests ──────────────────────────────────────────────────────

TEST(GpuAllocatorTest, BasicAllocateFree) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    GpuAllocator alloc;
    void* ptr = alloc.allocate(1024, 256, 0);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(alloc.get_total_allocated(), 1024u);
    EXPECT_EQ(alloc.get_allocation_count(), 1u);

    bool freed = alloc.deallocate(ptr);
    EXPECT_TRUE(freed);
    EXPECT_EQ(alloc.get_total_allocated(), 0u);
}

TEST(GpuAllocatorTest, FreeNull) {
    GpuAllocator alloc;
    bool result = alloc.deallocate(nullptr);
    EXPECT_TRUE(result);
}

TEST(GpuAllocatorTest, PeakTracking) {
    if (!cuda_available()) GTEST_SKIP() << "No CUDA device";

    GpuAllocator alloc;
    void* p1 = alloc.allocate(1000, 256, 0);
    void* p2 = alloc.allocate(2000, 256, 0);
    EXPECT_EQ(alloc.get_peak_allocated(), 3000u);

    alloc.deallocate(p1);
    EXPECT_EQ(alloc.get_total_allocated(), 2000u);
    EXPECT_EQ(alloc.get_peak_allocated(), 3000u);

    alloc.deallocate(p2);
}
