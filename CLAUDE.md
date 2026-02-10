# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRT Engine is a high-performance TensorRT inference engine library. C++17 core with optional Python bindings (pybind11). Supports ONNX models natively; TensorFlow and PyTorch models via subprocess conversion to ONNX.

## Build Commands

```bash
# Configure (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DTRT_ENGINE_BUILD_TESTS=ON \
  -DTRT_ENGINE_BUILD_BENCHMARKS=ON

# Build
cmake --build build

# Run all tests
cd build && ctest --output-on-failure

# Run a single test by name pattern
ctest -R "BERTModelTest.BuildEngine" --output-on-failure

# Run a test suite
ctest -R "EngineConfigTest" --output-on-failure

# Build Python bindings (requires pybind11)
cmake -B build -DTRT_ENGINE_BUILD_PYTHON=ON && cmake --build build
```

GoogleTest is auto-downloaded (v1.14.0 via FetchContent) if not found on the system.

## Architecture

The library is a single shared library (`libtrt_engine.so`) built from 12 source files under `src/`. All public API headers live in `include/trt_engine/`. Include `<trt_engine/trt_engine.h>` for the full API.

### Core Pipeline

1. **EngineBuilder** (`builder.h/cpp`) — Builds serialized TensorRT engines from ONNX files. Configures precision (FP32/FP16/INT8/FP8), dynamic shape optimization profiles, DLA offloading, timing cache.

2. **InferenceEngine** (`engine.h/cpp`) — Core inference. Factory-created from engine file or in-memory data. Uses a context pool (mutex + condition_variable) for thread safety and an internal thread pool for `infer_async()`. Inputs are `vector<vector<float>>`; for int64 tensors (NLP models), pack bytes via memcpy into float vectors (see `create_int64_as_float` in tests).

3. **ModelConverter** (`model_converter.h/cpp`) — Detects model format and converts TF/PyTorch to ONNX via external Python subprocesses (`tf2onnx`, `torch.onnx.export`).

### Advanced Features (layered on top of InferenceEngine)

- **CudaGraphExecutor** (`cuda_graph.h/cpp`) — Captures and replays CUDA graphs for reduced kernel launch overhead.
- **MultiStreamEngine** (`multi_stream.h/cpp`) — Worker-per-stream architecture with work-stealing queue.
- **DynamicBatcher** (`batcher.h/cpp`) — Collects individual requests into batches (by count or timeout), returns futures.
- **MultiGPUEngine** (`multi_gpu.h/cpp`) — One InferenceEngine per GPU, round-robin load balancing.

### Supporting Infrastructure

- **Logger** (`logger.h/cpp`) — Thread-safe singleton implementing `nvinfer1::ILogger`. Macros: `CUDA_CHECK`, `TRT_CHECK`.
- **Memory** (`memory.h/cpp`) — `GpuAllocator` (implements `nvinfer1::IGpuAllocator`), `DeviceBuffer`, `PinnedBuffer` RAII wrappers, `MemoryManager` with allocation tracking.
- **CudaUtils** (`cuda_utils.h/cpp`) — `CudaStream`, `CudaEvent`, `StreamPool` RAII wrappers; async memcpy helpers.
- **Calibrator** (`calibrator.h/cpp`) — `EntropyCalibratorV2` and `MinMaxCalibrator` for INT8 quantization from binary calibration data files.
- **Profiler** (`profiler.h/cpp`) — Layer-level timing (`TRTProfiler` via `nvinfer1::IProfiler`), high-level `PerformanceProfiler` with percentile stats, GPU metrics via NVML.

### RAII Type Aliases (defined in `types.h`)

`UniqueRuntime`, `UniqueEngine`, `UniqueContext`, `UniqueBuilder`, `UniqueNetwork`, `UniqueParser` — all use `TRTDeleter` with `std::unique_ptr`.

## Key Types

- `DynamicShapeProfile` — `{name, min_dims, opt_dims, max_dims}` for optimization profiles
- `BuilderConfig` — precision, workspace, dynamic_shapes, DLA, timing cache, strongly_typed
- `EngineConfig` — device_id, context_pool_size, enable_cuda_graph, thread_pool_size
- `InferenceResult` — `{outputs: vector<vector<float>>, latency_ms, success, error_msg}`
- `TensorInfo` — `{name, shape, dtype, size_bytes}`

## TensorRT API Notes

This targets TensorRT 10.x. Key differences from older versions:
- `IGpuAllocator` uses `deallocate()` not `free()`, and `allocate()` takes `uint64_t` not `size_t`
- `kSTRONGLY_TYPED` is a `NetworkDefinitionCreationFlag`, not a `BuilderFlag`
- Engine build uses `buildSerializedNetwork()`, inference uses `enqueueV3()` with `setTensorAddress()`

## NLP Model Testing

Tests in `test_nlp_models.cpp` cover BERT, DistilBERT, GPT-2, T5-small, and ResNet-18. NLP models use int64 inputs packed into float buffers. Models are stored in `models/` (gitignored, ~1.5 GB total).

GPT-2's ONNX export has batch_size=1 hardcoded in output shape, so its dynamic shape profiles must use fixed batch (`batch_dynamic=false` in `make_nlp_profiles()`). Other NLP models support dynamic batch.

## Dependencies

| Required | Version |
|----------|---------|
| CUDA Toolkit | 11.8+ |
| TensorRT | 8.6+ (targets 10.x API) |
| CMake | 3.18+ |
| C++ Compiler | GCC 9+ / Clang 11+ |
| GPU Compute | SM 70+ (V100 through H100) |

Optional: pybind11 (Python bindings), NVML (GPU metrics), numpy 1.20+ (Python).

## CUDA Architectures

Built for: 70, 75, 80, 86, 89, 90 (V100, T4, A100, A10/L40, L4/Ada, H100).
