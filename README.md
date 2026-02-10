# TRT Engine - High-Performance TensorRT Inference Engine

A production-ready C++17 inference library built on NVIDIA TensorRT for accelerating machine learning models on NVIDIA GPUs. Supports ONNX, TensorFlow, and PyTorch model formats with FP32, FP16, INT8, and FP8 precision modes.

## Key Features

- **Multi-Format Support**: Convert and optimize models from ONNX, TensorFlow, and PyTorch
- **Precision Modes**: FP32, FP16, INT8 (with calibration), and FP8 support
- **CUDA Graph Acceleration**: Capture and replay inference graphs to eliminate launch overhead
- **Multi-Stream Execution**: Concurrent inference across multiple CUDA streams
- **Dynamic Batching**: Automatic request batching with configurable timeout and batch size
- **Multi-GPU Support**: Data-parallel inference across multiple GPUs with load balancing
- **Dynamic Shapes**: Runtime input shape changes via optimization profiles
- **Thread-Safe**: Context pool and mutex-based synchronization for concurrent access
- **Python Bindings**: Full pybind11 bindings with NumPy integration
- **Performance Profiling**: Layer-level timing, latency percentiles, GPU metrics via NVML

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build with all features

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_ENGINE_BUILD_PYTHON=ON \
  -DTRT_ENGINE_BUILD_TESTS=ON \
  -DTRT_ENGINE_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

### C++ Usage

```cpp
#include <trt_engine/trt_engine.h>

int main() {
    // Initialize logger
    auto& logger = trt_engine::Logger::instance();

    // Build engine from ONNX
    trt_engine::EngineBuilder builder(logger);
    trt_engine::BuilderConfig config;
    config.precision = trt_engine::Precision::FP16;

    auto engine_data = builder.build_engine("model.onnx", config);
    builder.save_engine(engine_data, "model.engine");

    // Run inference
    auto engine = trt_engine::InferenceEngine::create("model.engine");
    engine->warmup(5);

    std::vector<std::vector<float>> inputs = { /* input data */ };
    auto result = engine->infer(inputs);

    if (result.success) {
        // Process result.outputs
    }
    return 0;
}
```

### Python Usage

```python
import trt_engine

# Build engine
builder = trt_engine.EngineBuilder()
config = trt_engine.BuilderConfig()
config.precision = trt_engine.Precision.FP16

engine_data = builder.build_engine("model.onnx", config)
builder.save_engine(engine_data, "model.engine")

# Run inference
engine = trt_engine.InferenceEngine.create("model.engine")
engine.warmup(5)

import numpy as np
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = engine.infer([input_data])
```

## System Requirements

| Component | Minimum Version |
|-----------|----------------|
| CUDA Toolkit | 11.8+ |
| TensorRT | 8.6+ |
| CMake | 3.18+ |
| C++ Compiler | GCC 9+ / Clang 11+ (C++17) |
| GPU Compute Capability | 7.0+ (V100, T4, A100, L40, H100) |
| Python (optional) | 3.8+ |
| pybind11 (optional) | 2.10+ |

## Project Structure

```
trt_engine/
├── include/trt_engine/     # Public headers
│   ├── trt_engine.h        # Main convenience header
│   ├── types.h             # Common types and enums
│   ├── logger.h            # Logging (nvinfer1::ILogger)
│   ├── memory.h            # GPU memory management
│   ├── cuda_utils.h        # CUDA stream/event wrappers
│   ├── model_converter.h   # Model format conversion
│   ├── builder.h           # TensorRT engine builder
│   ├── calibrator.h        # INT8 calibration
│   ├── engine.h            # Core inference engine
│   ├── cuda_graph.h        # CUDA graph executor
│   ├── multi_stream.h      # Multi-stream engine
│   ├── batcher.h           # Dynamic batching
│   ├── multi_gpu.h         # Multi-GPU engine
│   └── profiler.h          # Performance profiling
├── src/                    # Implementation files
├── python/                 # Python bindings (pybind11)
├── tests/                  # GoogleTest unit tests
├── benchmarks/             # Performance benchmarks
├── examples/               # C++ and Python examples
├── docs/                   # Documentation
│   ├── developer_guide.md  # Architecture & design guide
│   ├── api_reference.md    # Complete API reference
│   ├── deployment_guide.md # Deployment instructions
│   └── benchmark_results.md
└── research/               # Research notes
```

## Documentation

- [Developer Guide](docs/developer_guide.md) - Architecture, design patterns, code walkthroughs
- [API Reference](docs/api_reference.md) - Complete C++ and Python API documentation
- [Deployment Guide](docs/deployment_guide.md) - Build, Docker, GPU configuration, troubleshooting
- [Benchmark Results](docs/benchmark_results.md) - Performance benchmarks and tuning

## Examples

| Example | Description |
|---------|-------------|
| [basic_inference.cpp](examples/basic_inference.cpp) | Simple ONNX model inference |
| [dynamic_shapes.cpp](examples/dynamic_shapes.cpp) | Dynamic input shapes |
| [int8_inference.cpp](examples/int8_inference.cpp) | INT8 quantized inference |
| [cuda_graph_inference.cpp](examples/cuda_graph_inference.cpp) | CUDA graph acceleration |
| [multi_stream.cpp](examples/multi_stream.cpp) | Multi-stream concurrent inference |
| [multi_gpu.cpp](examples/multi_gpu.cpp) | Multi-GPU data parallelism |
| [basic_inference.py](examples/python/basic_inference.py) | Python inference |
| [batch_inference.py](examples/python/batch_inference.py) | Python dynamic batching |

## License

Apache 2.0
