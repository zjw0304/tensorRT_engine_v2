# TRT Engine Developer Guide

## 1. Project Overview

**trt_engine** is a high-performance C++17 inference library built on top of NVIDIA TensorRT. It provides a clean, modular API for converting models from ONNX, TensorFlow, and PyTorch formats into optimized TensorRT engines, and running inference with support for FP32, FP16, INT8, and FP8 precision modes. The library includes CUDA graph acceleration, multi-stream execution, dynamic batching, and multi-GPU scaling, all exposed through thread-safe interfaces with RAII resource management.

### Goals

- Provide a production-ready C++ inference engine with minimal boilerplate.
- Abstract TensorRT complexity behind factory methods and configuration structs.
- Support advanced features (CUDA graphs, multi-stream, multi-GPU) without sacrificing API simplicity.
- Offer Python bindings via pybind11 for rapid prototyping and integration.
- Achieve near-zero overhead compared to raw TensorRT API usage.

---

## 2. Architecture Overview

```
+-------------------------------------------------------------+
|                    Python API (pybind11)                      |
+-------------------------------------------------------------+
|                      C++ Public API                          |
|  +----------+  +-----------+  +----------+  +-----------+   |
|  | Engine   |  | Multi     |  | Dynamic  |  | Multi     |   |
|  | Builder  |  | Stream    |  | Batcher  |  | GPU       |   |
|  +----+-----+  +-----+-----+  +----+-----+  +-----+-----+  |
|       |              |              |              |         |
+-------+--------------+--------------+--------------+---------+
|                  Core Engine Layer                            |
|  +----------------------------------------------------------+|
|  |            InferenceEngine (Core)                        ||
|  |  +----------+ +----------+ +----------+ +--------+      ||
|  |  |  Model   | | Context  | |  Memory  | |  CUDA  |      ||
|  |  | Converter| |   Pool   | |  Manager | |  Graph |      ||
|  |  +----------+ +----------+ +----------+ +--------+      ||
|  +----------------------------------------------------------+|
+--------------------------------------------------------------+
|         TensorRT SDK / CUDA Runtime / cuDNN / NVML           |
+--------------------------------------------------------------+
```

### Module Dependency Graph

```
types.h  <--  logger.h  <--  memory.h  <--  cuda_utils.h
                  |               |               |
                  v               v               v
          model_converter.h   builder.h      engine.h
                                  |               |
                                  v               v
                           calibrator.h    cuda_graph.h
                                          multi_stream.h
                                          batcher.h
                                          multi_gpu.h
                                                |
                                                v
                                         trt_engine.h  (umbrella)
```

---

## 3. Module Descriptions

### 3.1 Types (`include/trt_engine/types.h`)

Defines all shared types, enums, and utility functions used across the library.

| Type | Purpose |
|------|---------|
| `Precision` | Enum: FP32, FP16, INT8, FP8 |
| `LogSeverity` | Enum: INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE |
| `DynamicShapeProfile` | Struct holding name, min/opt/max dimension vectors |
| `BuilderConfig` | Struct for engine build configuration |
| `DeviceConfig` | Struct with device_id and workspace_size |
| `InferenceResult` | Struct with outputs, latency_ms, success flag, error_msg |
| `TensorInfo` | Struct describing a tensor: name, shape, dtype, size_bytes |
| `DeviceProperties` | Struct with GPU device properties |
| `TRTDeleter` | Custom deleter for TensorRT objects used with unique_ptr |

RAII type aliases are also defined here: `UniqueRuntime`, `UniqueEngine`, `UniqueContext`, `UniqueBuilder`, `UniqueNetwork`, `UniqueParser`, `UniqueProfile`.

Utility functions include `precision_to_string`, `string_to_precision`, `precision_to_trt`, `datatype_size`, and severity conversion helpers.

### 3.2 Logger (`include/trt_engine/logger.h`, `src/logger.cpp`)

Singleton logger implementing `nvinfer1::ILogger`. Thread-safe with `std::mutex`.

- **Severity filtering**: Messages below `min_severity_` are discarded.
- **Console output**: Color-coded via ANSI escape codes (bold red for INTERNAL_ERROR, red for ERROR, yellow for WARNING, gray for VERBOSE).
- **File output**: Optional file logging via `enable_file_output(path)`.
- **Timestamps**: Each message is timestamped with millisecond precision.
- **Convenience methods**: `error()`, `warning()`, `info()`, `verbose()`.
- **Error checking macros**: `CUDA_CHECK(call)` and `TRT_CHECK(expr)` throw `std::runtime_error` on failure with file/line information.

Access the global logger via `trt_engine::get_logger()` or `Logger::instance()`.

### 3.3 Memory Manager (`include/trt_engine/memory.h`, `src/memory.cpp`)

Provides RAII wrappers for GPU and pinned host memory, plus a custom TensorRT allocator.

- **GpuAllocator**: Implements `nvinfer1::IGpuAllocator`. Tracks all allocations in a hash map. Provides `get_total_allocated()`, `get_peak_allocated()`, and `get_allocation_count()` for monitoring.
- **DeviceBuffer**: RAII wrapper around `cudaMalloc`/`cudaFree`. Move-only. Provides `data()`, `size()`, `empty()`, and typed `as<T>()` accessors.
- **PinnedBuffer**: RAII wrapper around `cudaMallocHost`/`cudaFreeHost`. Same interface as DeviceBuffer but for page-locked host memory.
- **MemoryManager**: Central manager that creates DeviceBuffer and PinnedBuffer instances while tracking aggregate statistics (total allocated, peak, count) for both device and pinned memory. Thread-safe.
- **Copy helpers**: `copy_to_device()` and `copy_to_host()` support both synchronous and async (stream-based) transfers.

### 3.4 CUDA Utilities (`include/trt_engine/cuda_utils.h`, `src/cuda_utils.cpp`)

RAII wrappers for CUDA runtime objects and device query functions.

- **CudaStream**: Wraps `cudaStream_t` with automatic creation/destruction. Supports custom flags. `synchronize()` blocks until all stream operations complete.
- **CudaEvent**: Wraps `cudaEvent_t`. `record(stream)` inserts a timestamp. `synchronize()` blocks until the event completes. Static `elapsed_time(start, end)` computes milliseconds between two events.
- **StreamPool**: Pre-allocates a pool of CudaStreams. `acquire()` returns a shared_ptr; if the pool is exhausted it creates additional streams with a warning. `release()` returns a stream to the pool. Thread-safe.
- **Async memcpy helpers**: `async_memcpy_h2d()` and `async_memcpy_d2h()` for stream-based transfers.
- **Device queries**: `get_device_count()` and `get_device_properties(device_id)` return device count and a populated `DeviceProperties` struct.

### 3.5 Model Converter (`include/trt_engine/model_converter.h`, `src/model_converter.cpp`)

Converts models from various frameworks to ONNX format for TensorRT consumption. All methods are static.

- **detect_format(path)**: Returns `ModelFormat` enum based on file extension (`.onnx`, `.pb`, `.savedmodel`, `.pt`, `.pth`, `.torchscript`, `.engine`, `.plan`, `.trt`). Also checks for SavedModel directories.
- **convert(input_path, output_path)**: Auto-detects format and routes to the appropriate converter. ONNX models are validated and copied.
- **validate_onnx(path)**: Checks file existence and basic header validity.
- **optimize_onnx(input_path, output_path)**: Runs shape inference and optional onnxsim simplification via Python subprocess.
- **convert_tensorflow_to_onnx()**: Invokes `tf2onnx` for SavedModel directories or frozen graphs.
- **convert_pytorch_to_onnx()**: Loads TorchScript models and exports via `torch.onnx.export` with dynamic batch axis.

### 3.6 Engine Builder (`include/trt_engine/builder.h`, `src/builder.cpp`)

Builds serialized TensorRT engines from ONNX models.

- **build_engine(onnx_path, config)**: Full build pipeline: parse ONNX, configure precision, set up dynamic shapes, apply timing cache, build serialized network. Returns `std::vector<char>`.
- **save_engine(engine_data, path)** / **load_engine(path)**: Static methods for engine file I/O.
- **set_calibrator(calibrator)**: Attach an INT8 calibrator before building.
- Internally configures workspace size, DLA, auxiliary streams, strongly-typed mode, and optimization profiles.
- Timing cache is automatically loaded/saved when `timing_cache_path` is set in `BuilderConfig`.

### 3.7 Calibrator (`include/trt_engine/calibrator.h`, `src/calibrator.cpp`)

INT8 calibration support with two algorithms.

- **EntropyCalibratorV2**: Implements `nvinfer1::IInt8EntropyCalibrator2`. Recommended for CNN-based models. Reads `.bin`/`.raw` files from a data directory, uploads batches to GPU, and supports calibration cache for persistence.
- **MinMaxCalibrator**: Implements `nvinfer1::IInt8MinMaxCalibrator`. Uses full activation range. Recommended for NLP/transformer models. Same data pipeline as EntropyCalibratorV2.

Both calibrators:
- Pre-allocate a device buffer for the full batch.
- Iterate through data files sorted alphabetically.
- Support cache read/write to avoid re-calibration.

### 3.8 Inference Engine (`include/trt_engine/engine.h`, `src/engine.cpp`)

The core inference class. Created via factory methods, not direct construction.

- **create(engine_path, config)** / **create(engine_data, config)**: Factory methods that deserialize the engine and initialize context pool + thread pool.
- **infer(input_buffers)**: Synchronous inference. Acquires a context from the pool, allocates device buffers, copies inputs H2D, runs `enqueueV3`, copies outputs D2H, measures latency via CUDA events. Returns `InferenceResult`.
- **infer_async(input_buffers)**: Submits inference to an internal thread pool. Returns `std::future<InferenceResult>`.
- **set_input_shape(name, dims)**: Thread-safe dynamic shape override.
- **get_input_info()** / **get_output_info()**: Query tensor metadata.
- **warmup(iterations)**: Runs N dummy inferences to warm up the engine, JIT compilation caches, and memory allocators.
- **Context pool**: Configurable pool size (`context_pool_size`). Contexts are acquired/released with condition variable signaling.
- **Thread pool**: Configurable size (`thread_pool_size`). Processes `packaged_task` objects from a queue.

### 3.9 CUDA Graph Executor (`include/trt_engine/cuda_graph.h`, `src/cuda_graph.cpp`)

Captures and replays CUDA graphs for reduced kernel launch overhead.

- **capture(context, stream)**: (1) Pre-capture flush via `enqueueV3` + sync, (2) `cudaStreamBeginCapture`, (3) `enqueueV3` inside capture, (4) `cudaStreamEndCapture`, (5) `cudaGraphInstantiate`. Returns true on success.
- **launch(stream)**: Replays the captured graph via `cudaGraphLaunch`.
- **is_captured()**: Query capture state.
- **reset()**: Destroys the graph and instance, allowing re-capture.
- Thread-safe via mutex. Properly cleans up on failed captures.

### 3.10 Multi-Stream Engine (`include/trt_engine/multi_stream.h`, `src/multi_stream.cpp`)

Worker thread pool where each thread owns its own CUDA stream and execution context.

- **Constructor**: Loads engine, creates N worker threads, each with its own `IExecutionContext` and `CudaStream`.
- **infer(input_buffers)**: Blocking dispatch to next available worker.
- **submit(input_buffers)**: Non-blocking submit, returns `std::future<InferenceResult>`.
- **shutdown()**: Gracefully stops all workers.
- Request queue is protected by mutex + condition variable.

### 3.11 Dynamic Batcher (`include/trt_engine/batcher.h`, `src/batcher.cpp`)

Collects individual inference requests and batches them for improved throughput.

- **Constructor**: Takes a shared `InferenceEngine`, `max_batch_size`, and `max_wait_time_ms`.
- **submit(single_input)**: Submits one sample. Returns `std::future<InferenceResult>`.
- **Batch loop**: Background thread waits for requests. Once `max_batch_size` requests arrive or `max_wait_time_ms` elapses, the batch is formed.
- **execute_batch()**: Concatenates inputs along batch dimension, sets input shapes on the engine, runs batched inference, then slices outputs back to individual results.
- Single-sample batches bypass concatenation/slicing.

### 3.12 Multi-GPU Engine (`include/trt_engine/multi_gpu.h`, `src/multi_gpu.cpp`)

Data-parallel inference across multiple GPUs.

- **Constructor**: Creates one `InferenceEngine` per device ID. Validates device IDs against available GPUs.
- **infer(input_buffers)**: Synchronous inference on the next device (round-robin).
- **infer_async(input_buffers)**: Async inference via the selected device's engine.
- **get_device_count()** / **get_device_info(index)** / **get_device_ids()**: Query device information.
- Round-robin selection uses `std::atomic<uint64_t>` with `memory_order_relaxed`.

---

## 4. Design Patterns

### 4.1 RAII (Resource Acquisition Is Initialization)

Every CUDA and TensorRT resource is wrapped in an RAII type:
- `DeviceBuffer` / `PinnedBuffer` for memory
- `CudaStream` / `CudaEvent` for stream/event objects
- `UniqueEngine`, `UniqueContext`, etc. using `std::unique_ptr<T, TRTDeleter>`

This guarantees cleanup even during exception unwinding.

### 4.2 Factory Method

`InferenceEngine` uses private constructors with public static `create()` methods:
```cpp
auto engine = InferenceEngine::create("model.engine");
```
This allows the factory to perform validation, throw on failure, and return a `unique_ptr`.

### 4.3 Future/Promise (Async Inference)

Async operations return `std::future<InferenceResult>`:
```cpp
auto future = engine->infer_async(inputs);
// ... do other work ...
auto result = future.get();
```
This pattern is used in `InferenceEngine`, `MultiStreamEngine`, `DynamicBatcher`, and `MultiGPUEngine`.

### 4.4 Producer-Consumer

`MultiStreamEngine` and `DynamicBatcher` use a producer-consumer queue pattern:
- Producers call `submit()` which pushes requests into a thread-safe queue.
- Consumer threads pull from the queue, process requests, and fulfill promises.
- `std::condition_variable` provides efficient wake-up signaling.

### 4.5 Context Pool

`InferenceEngine` maintains a pool of `IExecutionContext` objects:
- `acquire_context()` blocks if the pool is empty (via condition variable).
- `release_context()` returns the context and notifies waiters.
- Pool size is configurable via `EngineConfig::context_pool_size`.

### 4.6 Singleton

`Logger` uses the Meyers singleton pattern:
```cpp
Logger& Logger::instance() {
    static Logger logger;
    return logger;
}
```

---

## 5. Thread Safety Model

| Component | Thread Safety |
|-----------|--------------|
| `Logger` | Thread-safe (mutex-protected) |
| `MemoryManager` | Thread-safe (mutex-protected) |
| `GpuAllocator` | Thread-safe (mutex-protected) |
| `InferenceEngine::infer()` | Thread-safe (via context pool) |
| `InferenceEngine::infer_async()` | Thread-safe (via task queue) |
| `InferenceEngine::set_input_shape()` | Thread-safe (via shape mutex) |
| `MultiStreamEngine` | Thread-safe (worker threads + request queue) |
| `DynamicBatcher` | Thread-safe (batch thread + request queue) |
| `MultiGPUEngine` | Thread-safe (atomic round-robin + per-device engine isolation) |
| `CudaGraphExecutor` | Thread-safe (mutex-protected) |
| `StreamPool` | Thread-safe (mutex-protected) |
| `EngineBuilder` | NOT thread-safe (build one engine at a time) |

**Key guarantees:**
- Concurrent `infer()` calls on the same `InferenceEngine` are safe because each call acquires its own execution context.
- `EngineBuilder` should not be used from multiple threads simultaneously.
- Each `MultiStreamEngine` worker has its own dedicated context and stream, eliminating contention.

---

## 6. Memory Management

### Buffer Types

| Type | Allocation | Deallocation | Use Case |
|------|-----------|--------------|----------|
| `DeviceBuffer` | `cudaMalloc` | `cudaFree` | GPU memory for tensors |
| `PinnedBuffer` | `cudaMallocHost` | `cudaFreeHost` | Page-locked host memory for async transfers |

### Custom GPU Allocator

`GpuAllocator` implements `nvinfer1::IGpuAllocator` to give TensorRT control over memory allocation while maintaining tracking:
- All allocations are recorded in an `unordered_map<void*, AllocationInfo>`.
- Peak, total, and count statistics are maintained.
- The allocator is accessible via `MemoryManager::get_gpu_allocator()` for passing to TensorRT builders/runtimes.

### Copy Operations

```cpp
// Synchronous copy
copy_to_device(dst_gpu, src_host, size_bytes);

// Async copy (requires pinned host memory for true async behavior)
async_memcpy_h2d(dst_gpu, src_pinned, size_bytes, stream);
async_memcpy_d2h(dst_pinned, src_gpu, size_bytes, stream);
```

---

## 7. Error Handling Strategy

### Macros

```cpp
CUDA_CHECK(cudaMalloc(&ptr, size));   // Throws std::runtime_error with file:line
TRT_CHECK(builder != nullptr);        // Throws std::runtime_error if expression is false
```

Both macros log the error via `get_logger().error()` before throwing.

### Exception Types

- `std::runtime_error`: Used by CUDA_CHECK and TRT_CHECK.
- `EngineException`: Extends `std::runtime_error` for engine-specific errors.
- `std::invalid_argument`: Used by `string_to_precision()` for invalid input.

### Error Propagation

- `InferenceResult::success` and `InferenceResult::error_msg` provide a non-throwing error path for inference calls.
- Factory methods (`InferenceEngine::create()`) throw on initialization failure.
- `EngineBuilder::build_engine()` returns an empty vector on failure.
- `EngineBuilder::save_engine()` / `load_engine()` return bool or empty vector.

### Destructor Safety

All destructors catch CUDA errors and log them rather than throwing:
```cpp
CudaStream::~CudaStream() {
    if (stream_) {
        cudaError_t err = cudaStreamDestroy(stream_);
        if (err != cudaSuccess) {
            get_logger().error("...");  // Log but don't throw
        }
    }
}
```

---

## 8. Performance Optimization Techniques

### 8.1 CUDA Graphs

CUDA graphs eliminate per-inference kernel launch overhead by capturing and replaying a sequence of GPU operations. The `CudaGraphExecutor` implements the recommended TensorRT capture pattern:

1. **Pre-capture flush**: Run one `enqueueV3` + sync to flush deferred updates.
2. **Capture**: `cudaStreamBeginCapture` -> `enqueueV3` -> `cudaStreamEndCapture`.
3. **Instantiate**: `cudaGraphInstantiate` creates an executable graph.
4. **Launch**: `cudaGraphLaunch` replays the captured operations.

Graphs must be re-captured after input shape changes. Use `reset()` before re-capturing.

### 8.2 Multi-Stream Execution

`MultiStreamEngine` runs N worker threads, each with its own CUDA stream and execution context. This allows concurrent inference requests to execute in parallel on the GPU:

- Each worker independently handles memory allocation, H2D/D2H copies, and `enqueueV3` on its own stream.
- The GPU scheduler can overlap operations from different streams.
- Ideal for serving multiple clients simultaneously.

### 8.3 Dynamic Batching

`DynamicBatcher` improves throughput by combining individual requests into batches:

- Requests accumulate until `max_batch_size` or `max_wait_time_ms` is reached.
- Batched inference amortizes per-layer overhead.
- Results are automatically split back to individual futures.
- Trade-off: higher batch sizes increase throughput but may increase per-request latency.

### 8.4 Pinned Memory

`PinnedBuffer` uses page-locked (pinned) host memory. Benefits:
- Enables true asynchronous H2D/D2H transfers via CUDA streams.
- Avoids staging copies through the OS page cache.
- DMA transfers can proceed without CPU involvement.

### 8.5 Context Pool

Creating `IExecutionContext` is expensive. The context pool pre-creates contexts and recycles them:
- Eliminates context creation/destruction overhead per inference.
- Pool size is configurable to balance memory usage vs. concurrency.

---

## 9. Code Flow Walkthroughs

### 9.1 Building an Engine from ONNX

```
User code                       EngineBuilder                     TensorRT
   |                                 |                                |
   | build_engine("model.onnx", cfg) |                                |
   |------------------------------->|                                |
   |                                 | createInferBuilder()           |
   |                                 |------------------------------->|
   |                                 | createNetworkV2(EXPLICIT_BATCH)|
   |                                 |------------------------------->|
   |                                 | createParser() + parseFromFile |
   |                                 |------------------------------->|
   |                                 | createBuilderConfig()          |
   |                                 | configure_precision()          |
   |                                 | configure_dynamic_shapes()     |
   |                                 | load/create timing cache       |
   |                                 | buildSerializedNetwork()       |
   |                                 |------------------------------->|
   |                                 |<------- serialized engine -----|
   |                                 | save timing cache              |
   | <-- vector<char> engine_data ---|                                |
   |                                 |                                |
   | save_engine(data, "model.engine")                                |
```

### 9.2 Running Inference

```
User code             InferenceEngine                CUDA / TensorRT
   |                        |                              |
   | infer(input_buffers)   |                              |
   |----------------------->|                              |
   |                        | acquire_context()            |
   |                        | cudaSetDevice()              |
   |                        |----------------------------->|
   |                        | allocate DeviceBuffers       |
   |                        | async_memcpy_h2d (per input) |
   |                        |----------------------------->|
   |                        | setTensorAddress (in/out)    |
   |                        | start_event.record()         |
   |                        | enqueueV3(stream)            |
   |                        |----------------------------->|
   |                        | end_event.record()           |
   |                        | async_memcpy_d2h (per output)|
   |                        |----------------------------->|
   |                        | stream.synchronize()         |
   |                        | elapsed_time(start, end)     |
   |                        | release_context()            |
   | <-- InferenceResult ---|                              |
```

### 9.3 Using CUDA Graphs

```
User code          CudaGraphExecutor          CUDA
   |                      |                      |
   | capture(ctx, stream) |                      |
   |--------------------->|                      |
   |                      | enqueueV3 (flush)    |
   |                      |--------------------->|
   |                      | sync                 |
   |                      | beginCapture         |
   |                      |--------------------->|
   |                      | enqueueV3 (capture)  |
   |                      |--------------------->|
   |                      | endCapture           |
   |                      |--------------------->|
   |                      | instantiate          |
   | <-- true ------------|                      |
   |                      |                      |
   | launch(stream)       |                      |
   |--------------------->|                      |
   |                      | cudaGraphLaunch      |
   |                      |--------------------->|
   | <-- true ------------|                      |
```

### 9.4 Dynamic Batching

```
Client 1     Client 2     DynamicBatcher          InferenceEngine
   |              |              |                        |
   | submit(x1)   |              |                        |
   |------------->|              |                        |
   |              | submit(x2)   |                        |
   |              |------------->|                        |
   |              |              | batch_loop wakes up    |
   |              |              | wait for more or timeout|
   |              |              | concatenate x1, x2     |
   |              |              | set_input_shape(batch=2)|
   |              |              | infer(batched_inputs)   |
   |              |              |----------------------->|
   |              |              | <-- batch result ------|
   |              |              | slice outputs          |
   | <-- result1 --|              |                        |
   |              | <-- result2 --|                        |
```

---

## 10. How to Extend

### 10.1 Adding New Model Formats

1. Add a new value to `ModelFormat` in `model_converter.h`.
2. Update `detect_format()` with the file extension(s).
3. Implement a new `convert_<format>_to_onnx()` static method.
4. Add a case in `convert()` to route to your converter.

The converter pattern uses Python subprocesses, so any framework with a Python ONNX export API can be integrated.

### 10.2 Custom TensorRT Plugins

1. Implement your plugin following the `nvinfer1::IPluginV2` or `IPluginV3` interface.
2. Register the plugin creator with TensorRT's plugin registry.
3. Include the plugin in your ONNX model (ONNX custom op nodes will be matched to registered plugins).
4. Build the engine normally via `EngineBuilder` -- TensorRT will pick up registered plugins automatically.

### 10.3 Custom Calibrators

Implement the `nvinfer1::IInt8EntropyCalibrator2` or `nvinfer1::IInt8MinMaxCalibrator` interface:
1. Override `getBatchSize()`, `getBatch()`, `readCalibrationCache()`, `writeCalibrationCache()`.
2. Pass your calibrator to `EngineBuilder::set_calibrator()`.
3. Set `BuilderConfig::precision = Precision::INT8`.

### 10.4 Adding New Load Balancing Strategies

To add load-balancing strategies beyond round-robin in `MultiGPUEngine`:
1. Add a `LoadBalanceStrategy` enum.
2. Modify `select_device()` to implement the new strategy (e.g., least-loaded, random, GPU utilization-based).
3. For utilization-based strategies, query NVML for GPU metrics.

---

## 11. Best Practices

### For Engine Building

- **Set opt dimensions to your most common input size.** TensorRT optimizes most aggressively for opt dimensions.
- **Use timing cache** to speed up repeated builds and improve tactic consistency.
- **Keep min/max ranges narrow** in optimization profiles. Wider ranges disable certain tactics.
- **Use FP16** for the best performance-accuracy trade-off. Only use INT8 with proper calibration.

### For Inference

- **Warm up the engine** before benchmarking or serving: `engine->warmup(10)`.
- **Use pinned memory** for input/output data to enable true async transfers.
- **Match input sizes to opt dimensions** for best performance.
- **Reuse engines** -- deserialization is much faster than building from scratch.

### For Multi-GPU Setups

- **Build the engine once** and load it on each GPU. Engine files are portable across identical GPU architectures.
- **Use round-robin distribution** for uniform request sizes. Consider load-based distribution for variable workloads.

### For Production

- **Lock GPU clock frequencies** during benchmarking for deterministic results: `sudo nvidia-smi -lgc <freq>`
- **Use CUDA graphs** for small models where enqueue overhead is significant.
- **Set appropriate context pool sizes** -- too few causes contention, too many wastes GPU memory.
- **Monitor memory usage** via `MemoryManager` statistics.
- **Enable file logging** for production debugging: `get_logger().enable_file_output("/var/log/trt_engine.log")`

---

## 12. Build System

The project uses CMake (>= 3.18) with C++17 and CUDA support. See the [Deployment Guide](deployment_guide.md) for full build instructions.

Key CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `TRT_ENGINE_BUILD_PYTHON` | OFF | Build Python bindings |
| `TRT_ENGINE_BUILD_TESTS` | OFF | Build unit tests |
| `TRT_ENGINE_BUILD_BENCHMARKS` | OFF | Build benchmarks |

Supported CUDA architectures: 70 (V100), 75 (T4), 80 (A100), 86 (RTX 3090), 89 (L40/RTX 4090), 90 (H100).

---

## 13. File Reference

| File | Description |
|------|-------------|
| `include/trt_engine/trt_engine.h` | Umbrella header -- includes all public headers |
| `include/trt_engine/types.h` | Enums, structs, type aliases, utility functions |
| `include/trt_engine/logger.h` | Logger class and error-checking macros |
| `include/trt_engine/memory.h` | DeviceBuffer, PinnedBuffer, MemoryManager, GpuAllocator |
| `include/trt_engine/cuda_utils.h` | CudaStream, CudaEvent, StreamPool, device queries |
| `include/trt_engine/model_converter.h` | ModelConverter for ONNX/TF/PyTorch |
| `include/trt_engine/builder.h` | EngineBuilder |
| `include/trt_engine/calibrator.h` | EntropyCalibratorV2, MinMaxCalibrator |
| `include/trt_engine/engine.h` | InferenceEngine with context pool and thread pool |
| `include/trt_engine/cuda_graph.h` | CudaGraphExecutor |
| `include/trt_engine/multi_stream.h` | MultiStreamEngine |
| `include/trt_engine/batcher.h` | DynamicBatcher |
| `include/trt_engine/multi_gpu.h` | MultiGPUEngine |
| `src/logger.cpp` | Logger implementation |
| `src/memory.cpp` | Memory management implementation |
| `src/cuda_utils.cpp` | CUDA utility implementations |
| `src/model_converter.cpp` | Model conversion implementations |
| `src/builder.cpp` | Engine builder implementation |
| `src/calibrator.cpp` | Calibrator implementations |
| `src/engine.cpp` | Core inference engine implementation |
| `src/cuda_graph.cpp` | CUDA graph executor implementation |
| `src/multi_stream.cpp` | Multi-stream engine implementation |
| `src/batcher.cpp` | Dynamic batcher implementation |
| `src/multi_gpu.cpp` | Multi-GPU engine implementation |
| `CMakeLists.txt` | Top-level build configuration |
