# TRT Engine API Reference

## Table of Contents

- [Enums](#enums)
- [Structs](#structs)
- [Logger](#logger)
- [Memory Management](#memory-management)
- [CUDA Utilities](#cuda-utilities)
- [Model Converter](#model-converter)
- [Engine Builder](#engine-builder)
- [Calibrators](#calibrators)
- [Inference Engine](#inference-engine)
- [CUDA Graph Executor](#cuda-graph-executor)
- [Multi-Stream Engine](#multi-stream-engine)
- [Dynamic Batcher](#dynamic-batcher)
- [Multi-GPU Engine](#multi-gpu-engine)
- [Configuration Guide](#configuration-guide)

---

## Enums

### `trt_engine::Precision`

```cpp
enum class Precision {
    FP32,   // 32-bit floating point (default)
    FP16,   // 16-bit floating point
    INT8,   // 8-bit integer (requires calibration)
    FP8     // 8-bit floating point (E4M3)
};
```

### `trt_engine::LogSeverity`

```cpp
enum class LogSeverity {
    INTERNAL_ERROR = 0,  // Internal TensorRT errors
    ERROR          = 1,  // Runtime errors
    WARNING        = 2,  // Potential issues (default threshold)
    INFO           = 3,  // Informational messages
    VERBOSE        = 4   // Detailed debug messages
};
```

### `trt_engine::ModelFormat`

```cpp
enum class ModelFormat {
    ONNX,              // .onnx files
    TENSORFLOW,        // .pb files or SavedModel directories
    PYTORCH,           // .pt, .pth, .torchscript files
    TENSORRT_ENGINE,   // .engine, .plan, .trt files
    UNKNOWN            // Unrecognized format
};
```

---

## Structs

### `DynamicShapeProfile`

```cpp
struct DynamicShapeProfile {
    std::string      name;       // Input tensor name
    std::vector<int> min_dims;   // Minimum dimensions
    std::vector<int> opt_dims;   // Optimal dimensions (used for tuning)
    std::vector<int> max_dims;   // Maximum dimensions
};
```

### `BuilderConfig`

```cpp
struct BuilderConfig {
    Precision   precision           = Precision::FP32;      // Target precision
    size_t      max_workspace_size  = 1ULL << 30;           // 1 GB workspace
    bool        enable_cuda_graph   = false;                // Enable CUDA graph hints
    bool        enable_dla          = false;                // Enable DLA (Jetson)
    int         dla_core            = 0;                    // DLA core index
    std::string timing_cache_path;                          // Path for timing cache file
    int         max_aux_streams     = 0;                    // Auxiliary streams (0-7)
    bool        strongly_typed      = false;                // Enforce strict type matching
    std::vector<DynamicShapeProfile> dynamic_shapes;        // Dynamic shape profiles
};
```

### `DeviceConfig`

```cpp
struct DeviceConfig {
    int    device_id      = 0;              // CUDA device ID
    size_t workspace_size = 1ULL << 30;     // 1 GB workspace
};
```

### `EngineConfig`

```cpp
struct EngineConfig {
    int    device_id          = 0;    // CUDA device ID
    int    context_pool_size  = 2;    // Number of pre-created execution contexts
    bool   enable_cuda_graph  = false;// Enable CUDA graph support
    int    thread_pool_size   = 2;    // Worker threads for async inference
};
```

### `InferenceResult`

```cpp
struct InferenceResult {
    std::vector<std::vector<float>> outputs;   // Output tensor data
    float       latency_ms = 0.0f;             // GPU execution time in milliseconds
    bool        success    = false;            // Whether inference succeeded
    std::string error_msg;                     // Error message if success is false
};
```

### `TensorInfo`

```cpp
struct TensorInfo {
    std::string      name;           // Tensor name
    std::vector<int> shape;          // Tensor dimensions
    Precision        dtype;          // Data type
    size_t           size_bytes = 0; // Total size in bytes
};
```

### `DeviceProperties`

```cpp
struct DeviceProperties {
    std::string name;                           // GPU name (e.g., "NVIDIA A100")
    int         compute_capability_major = 0;   // Compute capability major version
    int         compute_capability_minor = 0;   // Compute capability minor version
    size_t      total_global_memory      = 0;   // Total GPU memory in bytes
    int         multi_processor_count    = 0;   // Number of SMs
    int         max_threads_per_block    = 0;   // Max threads per block
    size_t      shared_memory_per_block  = 0;   // Shared memory per block
    int         warp_size                = 0;   // Warp size (typically 32)
    int         clock_rate_khz           = 0;   // Core clock rate in kHz
    int         memory_clock_rate_khz    = 0;   // Memory clock rate in kHz
    int         memory_bus_width_bits    = 0;   // Memory bus width in bits
};
```

---

## Logger

### Class: `trt_engine::Logger`

Thread-safe singleton logger implementing `nvinfer1::ILogger`.

```cpp
class Logger : public nvinfer1::ILogger {
public:
    static Logger& instance();

    void log(Severity severity, const char* msg) noexcept override;

    void set_severity(LogSeverity severity);
    LogSeverity get_severity() const;

    void enable_file_output(const std::string& path);
    void disable_file_output();

    void error(const std::string& msg);
    void warning(const std::string& msg);
    void info(const std::string& msg);
    void verbose(const std::string& msg);
};
```

#### Free Function

```cpp
Logger& get_logger();   // Returns Logger::instance()
```

#### Example

```cpp
#include <trt_engine/logger.h>

auto& logger = trt_engine::get_logger();
logger.set_severity(trt_engine::LogSeverity::INFO);
logger.enable_file_output("/var/log/trt_engine.log");
logger.info("Engine initialized");
```

### Macros

```cpp
CUDA_CHECK(call)   // Checks cudaError_t, throws std::runtime_error on failure
TRT_CHECK(expr)    // Checks bool expression, throws std::runtime_error if false
```

---

## Memory Management

### Class: `trt_engine::GpuAllocator`

Custom GPU allocator implementing `nvinfer1::IGpuAllocator`.

```cpp
class GpuAllocator : public nvinfer1::IGpuAllocator {
public:
    void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept override;
    bool  free(void* memory) noexcept override;

    size_t get_total_allocated() const;
    size_t get_peak_allocated() const;
    size_t get_allocation_count() const;
    void   reset_stats();
};
```

### Class: `trt_engine::DeviceBuffer`

RAII GPU device memory. Move-only.

```cpp
class DeviceBuffer {
public:
    DeviceBuffer();
    explicit DeviceBuffer(size_t size);      // Allocates immediately

    void allocate(size_t size);
    void free();

    void*       data();
    const void* data() const;
    size_t      size() const;
    bool        empty() const;

    template <typename T> T* as();
    template <typename T> const T* as() const;
};
```

### Class: `trt_engine::PinnedBuffer`

RAII pinned (page-locked) host memory. Move-only. Same interface as `DeviceBuffer`.

### Class: `trt_engine::MemoryManager`

```cpp
class MemoryManager {
public:
    std::unique_ptr<DeviceBuffer> allocate_device(size_t size);
    std::unique_ptr<PinnedBuffer> allocate_pinned(size_t size);

    template <typename T>
    std::unique_ptr<DeviceBuffer> allocate_device_typed(size_t count);

    template <typename T>
    std::unique_ptr<PinnedBuffer> allocate_pinned_typed(size_t count);

    size_t get_total_device_allocated() const;
    size_t get_total_pinned_allocated() const;
    size_t get_peak_device_allocated() const;
    size_t get_peak_pinned_allocated() const;
    size_t get_device_allocation_count() const;
    size_t get_pinned_allocation_count() const;
    void   reset_stats();

    GpuAllocator& get_gpu_allocator();
};
```

### Free Functions

```cpp
void copy_to_device(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);
void copy_to_host(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);
```

#### Example

```cpp
#include <trt_engine/memory.h>

trt_engine::MemoryManager mm;
auto dev_buf = mm.allocate_device(1024 * sizeof(float));
auto pin_buf = mm.allocate_pinned(1024 * sizeof(float));

// Fill pinned buffer with data
float* host_data = pin_buf->as<float>();
for (int i = 0; i < 1024; ++i) host_data[i] = static_cast<float>(i);

// Copy to device
trt_engine::copy_to_device(dev_buf->data(), pin_buf->data(), 1024 * sizeof(float));
```

---

## CUDA Utilities

### Class: `trt_engine::CudaStream`

```cpp
class CudaStream {
public:
    CudaStream();                          // Creates default stream
    explicit CudaStream(unsigned int flags); // Creates stream with flags

    cudaStream_t get() const;
    operator cudaStream_t() const;
    void synchronize();
};
```

### Class: `trt_engine::CudaEvent`

```cpp
class CudaEvent {
public:
    CudaEvent();
    explicit CudaEvent(unsigned int flags);

    cudaEvent_t get() const;
    operator cudaEvent_t() const;
    void record(cudaStream_t stream = nullptr);
    void synchronize();

    static float elapsed_time(const CudaEvent& start, const CudaEvent& end);
};
```

### Class: `trt_engine::StreamPool`

```cpp
class StreamPool {
public:
    explicit StreamPool(size_t pool_size = 4);

    std::shared_ptr<CudaStream> acquire();
    void release(std::shared_ptr<CudaStream> stream);
    size_t pool_size() const;
    size_t available() const;
};
```

### Free Functions

```cpp
void async_memcpy_h2d(void* dst, const void* src, size_t size, cudaStream_t stream);
void async_memcpy_d2h(void* dst, const void* src, size_t size, cudaStream_t stream);
int  get_device_count();
DeviceProperties get_device_properties(int device_id);
```

#### Example

```cpp
#include <trt_engine/cuda_utils.h>

trt_engine::CudaStream stream;
trt_engine::CudaEvent start, end;

start.record(stream.get());
// ... kernel or memcpy operations on stream ...
end.record(stream.get());
stream.synchronize();

float elapsed_ms = trt_engine::CudaEvent::elapsed_time(start, end);
```

---

## Model Converter

### Class: `trt_engine::ModelConverter`

All methods are static.

```cpp
class ModelConverter {
public:
    static ModelFormat detect_format(const std::string& path);
    static bool convert(const std::string& input_path, const std::string& output_path);
    static bool validate_onnx(const std::string& path);
    static bool optimize_onnx(const std::string& input_path, const std::string& output_path);
    static bool convert_tensorflow_to_onnx(const std::string& input_path, const std::string& output_path);
    static bool convert_pytorch_to_onnx(const std::string& input_path, const std::string& output_path);
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `detect_format` | `path`: model file path | `ModelFormat` | Detects format from extension |
| `convert` | `input_path`, `output_path` | `bool` | Converts any supported format to ONNX |
| `validate_onnx` | `path` | `bool` | Validates ONNX file existence and header |
| `optimize_onnx` | `input_path`, `output_path` | `bool` | Runs shape inference and onnxsim |
| `convert_tensorflow_to_onnx` | `input_path`, `output_path` | `bool` | Converts TF via tf2onnx |
| `convert_pytorch_to_onnx` | `input_path`, `output_path` | `bool` | Converts TorchScript via torch.onnx.export |

#### Example

```cpp
#include <trt_engine/model_converter.h>

// Convert PyTorch model to ONNX
trt_engine::ModelConverter::convert("model.pt", "model.onnx");

// Optimize the ONNX model
trt_engine::ModelConverter::optimize_onnx("model.onnx", "model_opt.onnx");
```

---

## Engine Builder

### Class: `trt_engine::EngineBuilder`

```cpp
class EngineBuilder {
public:
    explicit EngineBuilder(Logger& logger);

    std::vector<char> build_engine(const std::string& onnx_path,
                                    const BuilderConfig& config);

    static bool save_engine(const std::vector<char>& engine_data,
                            const std::string& path);

    static std::vector<char> load_engine(const std::string& path);

    void set_calibrator(nvinfer1::IInt8Calibrator* calibrator);
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `build_engine` | `onnx_path`, `config` | `vector<char>` | Builds serialized engine from ONNX |
| `save_engine` | `engine_data`, `path` | `bool` | Writes engine to disk |
| `load_engine` | `path` | `vector<char>` | Reads engine from disk |
| `set_calibrator` | `calibrator` | `void` | Sets INT8 calibrator |

#### Example

```cpp
#include <trt_engine/builder.h>

auto& logger = trt_engine::get_logger();
trt_engine::EngineBuilder builder(logger);

trt_engine::BuilderConfig config;
config.precision = trt_engine::Precision::FP16;
config.max_workspace_size = 1ULL << 30;  // 1 GB
config.timing_cache_path = "timing.cache";
config.dynamic_shapes = {{
    "input",
    {1, 3, 224, 224},    // min
    {8, 3, 224, 224},    // opt
    {32, 3, 224, 224}    // max
}};

auto engine_data = builder.build_engine("model.onnx", config);
trt_engine::EngineBuilder::save_engine(engine_data, "model.engine");
```

---

## Calibrators

### Class: `trt_engine::EntropyCalibratorV2`

Implements `nvinfer1::IInt8EntropyCalibrator2`.

```cpp
class EntropyCalibratorV2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    EntropyCalibratorV2(const std::string& data_dir,
                        int batch_size,
                        const std::string& input_name,
                        const std::vector<int>& input_dims,
                        const std::string& cache_file);

    int         getBatchSize() const noexcept override;
    bool        getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void        writeCalibrationCache(const void* cache, size_t length) noexcept override;
};
```

### Class: `trt_engine::MinMaxCalibrator`

Implements `nvinfer1::IInt8MinMaxCalibrator`. Same constructor and interface as `EntropyCalibratorV2`.

| Parameter | Description |
|-----------|-------------|
| `data_dir` | Directory containing `.bin` or `.raw` calibration data files |
| `batch_size` | Number of samples per calibration batch |
| `input_name` | Name of the input tensor in the model |
| `input_dims` | Dimensions of a single input sample (excluding batch) |
| `cache_file` | Path to read/write calibration cache |

#### Example

```cpp
#include <trt_engine/calibrator.h>
#include <trt_engine/builder.h>

trt_engine::EntropyCalibratorV2 calibrator(
    "calibration_data/",       // directory with .bin files
    32,                        // batch size
    "input",                   // input tensor name
    {3, 224, 224},             // per-sample dimensions
    "calibration.cache"        // cache file
);

auto& logger = trt_engine::get_logger();
trt_engine::EngineBuilder builder(logger);
builder.set_calibrator(&calibrator);

trt_engine::BuilderConfig config;
config.precision = trt_engine::Precision::INT8;

auto engine_data = builder.build_engine("model.onnx", config);
```

---

## Inference Engine

### Class: `trt_engine::InferenceEngine`

```cpp
class InferenceEngine {
public:
    // Factory methods
    static std::unique_ptr<InferenceEngine> create(
        const std::string& engine_path, const EngineConfig& config = {});
    static std::unique_ptr<InferenceEngine> create(
        const std::vector<char>& engine_data, const EngineConfig& config = {});

    // Inference
    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);
    std::future<InferenceResult> infer_async(
        const std::vector<std::vector<float>>& input_buffers);

    // Dynamic shapes
    void set_input_shape(const std::string& name, const std::vector<int>& dims);

    // Tensor info
    std::vector<TensorInfo> get_input_info() const;
    std::vector<TensorInfo> get_output_info() const;

    // Warmup
    void warmup(int iterations = 5);

    // Advanced
    nvinfer1::ICudaEngine* get_engine() const;
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `create` (file) | `engine_path`, `config` | `unique_ptr<InferenceEngine>` | Loads engine from disk |
| `create` (data) | `engine_data`, `config` | `unique_ptr<InferenceEngine>` | Loads engine from memory |
| `infer` | `input_buffers` | `InferenceResult` | Synchronous inference |
| `infer_async` | `input_buffers` | `future<InferenceResult>` | Async inference |
| `set_input_shape` | `name`, `dims` | `void` | Override input dimensions |
| `get_input_info` | - | `vector<TensorInfo>` | Query input tensor metadata |
| `get_output_info` | - | `vector<TensorInfo>` | Query output tensor metadata |
| `warmup` | `iterations` | `void` | Run N dummy inferences |

#### Example

```cpp
#include <trt_engine/engine.h>

auto engine = trt_engine::InferenceEngine::create("model.engine");
engine->warmup(5);

// Query tensor info
auto inputs = engine->get_input_info();
auto outputs = engine->get_output_info();

// Prepare input data
std::vector<std::vector<float>> input_buffers = {
    std::vector<float>(3 * 224 * 224, 0.5f)  // single input tensor
};

// Synchronous inference
auto result = engine->infer(input_buffers);
if (result.success) {
    std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
    std::cout << "Output size: " << result.outputs[0].size() << std::endl;
}

// Async inference
auto future = engine->infer_async(input_buffers);
auto async_result = future.get();
```

---

## CUDA Graph Executor

### Class: `trt_engine::CudaGraphExecutor`

```cpp
class CudaGraphExecutor {
public:
    CudaGraphExecutor();

    bool capture(nvinfer1::IExecutionContext* context, cudaStream_t stream);
    bool launch(cudaStream_t stream);
    bool is_captured() const;
    void reset();
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `capture` | `context`, `stream` | `bool` | Captures a CUDA graph from enqueueV3 |
| `launch` | `stream` | `bool` | Launches the captured graph |
| `is_captured` | - | `bool` | Whether a graph is currently captured |
| `reset` | - | `void` | Destroys graph, allows re-capture |

#### Example

```cpp
#include <trt_engine/cuda_graph.h>

trt_engine::CudaGraphExecutor graph;
// Assume context and stream are set up with tensor addresses bound

if (graph.capture(context, stream.get())) {
    // Launch repeatedly for same-shape inputs
    for (int i = 0; i < 1000; ++i) {
        graph.launch(stream.get());
        cudaStreamSynchronize(stream.get());
    }
}
```

---

## Multi-Stream Engine

### Class: `trt_engine::MultiStreamEngine`

```cpp
class MultiStreamEngine {
public:
    MultiStreamEngine(const std::string& engine_path,
                      int num_streams,
                      const EngineConfig& config = {});

    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);
    std::future<InferenceResult> submit(
        const std::vector<std::vector<float>>& input_buffers);
    void shutdown();
    int num_streams() const;
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `infer` | `input_buffers` | `InferenceResult` | Blocking inference on next worker |
| `submit` | `input_buffers` | `future<InferenceResult>` | Non-blocking submit |
| `shutdown` | - | `void` | Stops all worker threads |
| `num_streams` | - | `int` | Number of parallel streams |

#### Example

```cpp
#include <trt_engine/multi_stream.h>

trt_engine::MultiStreamEngine engine("model.engine", 4);

// Submit multiple requests concurrently
std::vector<std::future<trt_engine::InferenceResult>> futures;
for (int i = 0; i < 10; ++i) {
    std::vector<std::vector<float>> input = {
        std::vector<float>(3 * 224 * 224, 0.5f)
    };
    futures.push_back(engine.submit(input));
}

// Collect results
for (auto& f : futures) {
    auto result = f.get();
    std::cout << "Latency: " << result.latency_ms << " ms" << std::endl;
}
```

---

## Dynamic Batcher

### Class: `trt_engine::DynamicBatcher`

```cpp
class DynamicBatcher {
public:
    DynamicBatcher(std::shared_ptr<InferenceEngine> engine,
                   int max_batch_size,
                   int max_wait_time_ms);

    std::future<InferenceResult> submit(
        const std::vector<std::vector<float>>& single_input);

    int max_batch_size() const;
    int max_wait_time_ms() const;
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `submit` | `single_input` | `future<InferenceResult>` | Submit single sample |
| `max_batch_size` | - | `int` | Configured max batch size |
| `max_wait_time_ms` | - | `int` | Configured max wait time |

#### Example

```cpp
#include <trt_engine/batcher.h>

auto engine = std::make_shared<trt_engine::InferenceEngine>(
    *trt_engine::InferenceEngine::create("model.engine"));

// Note: for DynamicBatcher, create engine with shared_ptr
auto engine_ptr = std::shared_ptr<trt_engine::InferenceEngine>(
    trt_engine::InferenceEngine::create("model.engine").release());

trt_engine::DynamicBatcher batcher(engine_ptr, /*max_batch=*/16, /*wait_ms=*/10);

// Submit individual requests from multiple threads
auto future = batcher.submit({std::vector<float>(3 * 224 * 224, 0.5f)});
auto result = future.get();
```

---

## Multi-GPU Engine

### Class: `trt_engine::MultiGPUEngine`

```cpp
class MultiGPUEngine {
public:
    MultiGPUEngine(const std::string& engine_path,
                   const std::vector<int>& device_ids,
                   const EngineConfig& config = {});

    InferenceResult infer(const std::vector<std::vector<float>>& input_buffers);
    std::future<InferenceResult> infer_async(
        const std::vector<std::vector<float>>& input_buffers);

    int get_device_count() const;
    DeviceProperties get_device_info(int index) const;
    const std::vector<int>& get_device_ids() const;
};
```

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `infer` | `input_buffers` | `InferenceResult` | Sync inference (round-robin) |
| `infer_async` | `input_buffers` | `future<InferenceResult>` | Async inference (round-robin) |
| `get_device_count` | - | `int` | Number of GPUs in use |
| `get_device_info` | `index` | `DeviceProperties` | Device properties by index |
| `get_device_ids` | - | `const vector<int>&` | List of device IDs |

#### Example

```cpp
#include <trt_engine/multi_gpu.h>

trt_engine::MultiGPUEngine engine("model.engine", {0, 1});

auto result = engine.infer({std::vector<float>(3 * 224 * 224, 0.5f)});
std::cout << "Executed on one of " << engine.get_device_count() << " GPUs" << std::endl;
```

---

## Configuration Guide

### BuilderConfig Options

| Field | Default | Range/Values | Description |
|-------|---------|-------------|-------------|
| `precision` | `FP32` | FP32, FP16, INT8, FP8 | Target inference precision |
| `max_workspace_size` | `1 << 30` (1 GB) | Any positive value | Max GPU memory for TensorRT workspace |
| `enable_cuda_graph` | `false` | true/false | Hint for CUDA graph compatibility |
| `enable_dla` | `false` | true/false | Enable DLA on supported platforms |
| `dla_core` | `0` | 0-N | DLA core to use |
| `timing_cache_path` | `""` | File path | Persistence path for timing cache |
| `max_aux_streams` | `0` | 0-7 | Auxiliary streams for parallel layer execution |
| `strongly_typed` | `false` | true/false | Enforce strict type matching |
| `dynamic_shapes` | `{}` | Vector of profiles | Dynamic shape min/opt/max |

### EngineConfig Options

| Field | Default | Range | Description |
|-------|---------|-------|-------------|
| `device_id` | `0` | 0-N | CUDA device to use |
| `context_pool_size` | `2` | >= 1 | Pre-created execution contexts |
| `enable_cuda_graph` | `false` | true/false | Enable CUDA graph support |
| `thread_pool_size` | `2` | >= 1 | Worker threads for async inference |

### Precision Selection Guide

| Precision | Accuracy | Performance | Requirements |
|-----------|----------|-------------|-------------|
| FP32 | Highest | Baseline | None |
| FP16 | Near-FP32 | ~2x FP32 | GPU with Tensor Cores |
| INT8 | Good (with calibration) | ~4x FP32 | Calibration data + calibrator |
| FP8 | Good | ~4x FP32 | Ada Lovelace+ GPU, Q/DQ nodes in model |

### Python API

Python bindings expose the same API with Pythonic conventions:

```python
import trt_engine

# Build engine
builder = trt_engine.EngineBuilder(trt_engine.get_logger())
config = trt_engine.BuilderConfig()
config.precision = trt_engine.Precision.FP16
engine_data = builder.build_engine("model.onnx", config)

# Run inference
engine = trt_engine.InferenceEngine.create("model.engine")
engine.warmup(5)

import numpy as np
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32).flatten().tolist()
result = engine.infer([input_data])
print(f"Latency: {result.latency_ms:.2f} ms")

# Dynamic batching
batcher = trt_engine.DynamicBatcher(engine, max_batch_size=16, max_wait_time_ms=10)
future = batcher.submit([input_data])
result = future.get()
```
