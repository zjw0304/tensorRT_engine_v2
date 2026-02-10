# Open-Source TensorRT Implementations & Industry Best Practices

## 1. NVIDIA Official TensorRT Samples

### 1.1 Sample Projects Overview

The official NVIDIA TensorRT repository (`NVIDIA/TensorRT`, 12.7k stars) provides the following C++ samples:

| Sample | Format | Description |
|--------|--------|-------------|
| sampleOnnxMNIST | ONNX | "Hello World" introductory TensorRT sample |
| sampleDynamicReshape | ONNX | Digit recognition with dynamic shapes |
| sampleEditableTimingCache | INetwork | Deterministic builds via editable timing cache |
| sampleNonZeroPlugin | INetwork | Plugin with data-dependent output shapes |
| sampleOnnxMnistCoordConvAC | ONNX | Custom plugin for CoordConv |
| sampleIOFormats | ONNX | TensorRT I/O format specification |
| sampleProgressMonitor | ONNX | Progress Monitor API usage |
| trtexec | All | Command-line benchmarking and engine builder |
| sampleSafeMNIST | ONNX | Safety engine building |
| sampleSafePluginV3 | ONNX | Safety-supported plugins |
| sampleCudla | INetwork | CuDLA API integration (aarch64 only) |

### 1.2 Key Design Patterns from Official Samples

#### Engine Building Pattern
Two primary approaches:
- **ONNX Parser Path**: Load `.onnx` file, let parser populate network definition. Used by most samples (sampleOnnxMNIST, sampleDynamicReshape).
- **INetwork API Path**: Manually construct network layer-by-layer. Used by sampleNonZeroPlugin, giving fine-grained control over topology.

#### Inference Execution Pattern
1. Create execution context from built engine
2. Allocate input/output device buffers
3. Transfer data host-to-device
4. Invoke `enqueueV3()` for async execution
5. Copy results device-to-host

#### Error Handling: Logger Callback Pattern
All samples use TensorRT's `ILogger` interface:
- Custom `ILogger` implementation receives severity-tagged messages
- Common utilities in `samples/common/` provide shared logger
- Macros for checking return codes from CUDA and TensorRT API calls
- `ErrorRecorder.h` implements `IErrorRecorder` interface for capturing and tracking errors

#### Memory Management: RAII Pattern
- CUDA memory via explicit `cudaMalloc`/`cudaFree` wrapped in RAII helpers
- TensorRT objects (builder, network, engine, context) use reference-counted smart pointers with custom deleters
- `buffers.h` provides buffer management classes abstracting host/device memory

### 1.3 Common Utilities Infrastructure (`samples/common/`)

| File | Purpose |
|------|---------|
| `buffers.h` | Buffer management for host/device memory |
| `logger.h/.cpp` | Core TensorRT logger implementation |
| `logging.h` | Additional logging abstractions |
| `ErrorRecorder.h` | `IErrorRecorder` interface implementation |
| `common.h` | Shared type definitions, smart pointer wrappers, macros |
| `sampleEngines.h/.cpp` | Engine construction, serialization, deserialization |
| `sampleInference.h/.cpp` | Running inference with built engines |
| `sampleOptions.h/.cpp` | Command-line option parsing |
| `sampleReporting.h/.cpp` | Performance metric reporting |
| `sampleDevice.h/.cpp` | GPU device selection and configuration |
| `EntropyCalibrator.h` | INT8 entropy calibration support |
| `BatchStream.h` | Batch data streaming for calibration |
| `half.h` | FP16 data type support |
| `bfloat16.h/.cpp` | BFloat16 data type support |
| `parserOnnxConfig.h` | ONNX parser configuration |
| `debugTensorWriter.h/.cpp` | Debug tensor output to disk |
| `safeCommon.h` | Safety-certified common utilities |
| `safeCudaAllocator.h` | Safety-focused CUDA allocator |
| `safeErrorRecorder.h` | Safety-variant error recorder |

---

## 2. Triton Inference Server Architecture

### 2.1 Architecture Overview
Triton Inference Server is a modular inference-serving system with a layered architecture:

```
Requests --> [HTTP/REST | gRPC | C API] --> Per-Model Scheduler --> Backend --> Response
```

**Key Architectural Principle**: Decoupling -- protocol layer, scheduling logic, and model execution backends are independent subsystems.

### 2.2 Backend Interface (Plugin Architecture)
- Extensible via a **Backend C API** contract
- Supported backends: TensorRT, TensorRT-LLM, vLLM, ONNX Runtime, PyTorch, Python, FIL, DALI
- Each backend is responsible for:
  - Loading a model from the repository
  - Executing inference with batched inputs
  - Producing output tensors
- Custom backends can be written in C/C++ or Python

### 2.3 Model Repository Structure
Filesystem-based store:
```
model_repository/
  model_name/
    config.pbtxt          # Model configuration
    1/                    # Version 1
      model.plan          # TensorRT engine
    2/                    # Version 2
      model.plan
```

### 2.4 Scheduling Strategies

| Scheduler | Model Type | Key Behavior |
|-----------|-----------|--------------|
| Default | Stateless | Simple serial execution per instance |
| Dynamic Batcher | Stateless | Combines requests into batches for throughput |
| Sequence Batcher (Direct) | Stateful | Routes all sequence requests to same batch slot |
| Sequence Batcher (Oldest) | Stateful | Dynamic batching across sequences with correlation IDs |
| Ensemble Scheduler | Pipelines | Event-driven DAG orchestration |

### 2.5 Key Design Decisions
1. **Protocol agnosticism**: Same models accessible via HTTP, gRPC, or in-process C/Python/Java APIs
2. **Per-model scheduler configuration**: Each model independently selects its scheduler and batching strategy
3. **Instance groups as scaling primitive**: Concurrency controlled by instance count configuration
4. **Health and observability built in**: Readiness/liveness endpoints and utilization/throughput/latency metrics
5. **Backend as a contract**: Minimal API contract -- new frameworks integrate without modifying core server
6. **Model management API**: Runtime querying and control over which models are loaded

### 2.6 Performance Tools
- **Performance Analyzer**: Benchmarking tool for throughput and latency
- **Model Analyzer**: Optimizes model configuration through profiling
- **Metrics**: GPU utilization, server throughput, latency indicators

---

## 3. TensorRT-LLM Architecture & Patterns

### 3.1 Architecture
- Built on **PyTorch-native architecture**
- High-level Python LLM API for inference across diverse deployment configurations
- Modular and easy to modify runtime
- Models defined using native PyTorch code

### 3.2 Design Patterns
- **Layered API Design**: High-level Python LLM API abstracts complexity while allowing low-level customization
- **Ecosystem Integration**: Seamless interoperability with NVIDIA Dynamo and Triton Inference Server
- **AutoDeploy**: Beta backend to simplify deployment of PyTorch models
- **Parallelism Abstractions**: Built-in multi-GPU and multi-node support

### 3.3 Performance Optimization Techniques

| Category | Techniques |
|----------|-----------|
| **Attention** | Custom attention kernels, XQA-kernel, Multiblock Attention, Skip Softmax Attention |
| **Batching** | Inflight batching (continuous batching) |
| **Memory** | Paged KV caching, KV cache reuse optimizations |
| **Quantization** | FP8, FP4 (Blackwell), INT4 AWQ, INT8 SmoothQuant |
| **Decoding** | Speculative decoding (up to 3.6x throughput), N-Gram speculative decoding |
| **Parallelism** | Expert Parallelism (EP), Tensor Parallelism (TP), Pipeline Parallelism (PP) |
| **Serving** | Disaggregated serving, ADP balance strategy |
| **Runtime** | CUDA Graphs (up to 22% throughput increase), Overlap Scheduler |

### 3.4 Multi-GPU & Scaling Strategies

#### Tensor Parallelism (TP)
- Shards model weights across multiple GPUs
- Every GPU processes same input tokens but holds fraction of weight matrices
- Results combined via inter-GPU communication
- Best for: small batch sizes, memory-constrained deployments

#### Pipeline Parallelism (PP)
- Assigns different layers to different GPUs
- Activations flow sequentially between GPUs
- Best for: very large models that cannot fit in single GPU memory

#### Data Parallelism (DP)
- Replicates full model on each GPU
- Different requests routed to different replicas
- KV-cache partitioned across replicas
- Best for: large batch sizes, high-throughput scenarios

#### Expert Parallelism (EP)
- For Mixture-of-Experts models
- Three execution patterns: TP, EP, Hybrid ETP
- Constraint: `moe_tp_size * moe_ep_size == tp_size`

#### Wide Expert Parallelism
- Advanced EP variant for large MoE models (DeepSeek-V3/R1, LLaMA4, Qwen3)
- Expert slots decoupled from specific experts
- EPLB (Expert Parallelism Load Balancer): offline and online load balancing
- Custom communication kernels optimized for NVIDIA GB200 MNNVL

#### Context Parallelism (CP)
- Distributes processing of long input sequences across GPUs
- Best for long-context scenarios

### 3.5 API Design
- **LLM API**: Primary interface supporting single-GPU to multi-node deployments
- **Python-first**: Easy-to-use Python API
- **Deprecation Policy**: 3-month migration period, runtime warnings, semantic versioning

---

## 4. Torch-TensorRT Integration

### 4.1 Two Compilation Workflows

#### JIT-style (`torch.compile` backend)
```python
import torch_tensorrt
optimized_model = torch.compile(model, backend="tensorrt")
optimized_model(x)  # compilation happens on first invocation
```
- Lazy compilation during first forward pass
- Zero-friction API
- Drop-in replacement

#### AOT-style (Export Workflow)
```python
# Compile and serialize
trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
torch_tensorrt.save(trt_gm, "trt.ep", inputs=inputs)       # ExportedProgram
torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", inputs=inputs)  # TorchScript
```
- Ahead-of-time optimization
- Dual serialization: `.ep` for Python, `.ts` for C++ deployment via libtorch

### 4.2 API Design Principles

| Principle | Implementation |
|-----------|---------------|
| PyTorch-native | Uses `torch.compile`, `torch.export`, standard serialization |
| Multiple IRs | Supports Dynamo IR for graph capture |
| Dual serialization | ExportedProgram for Python, TorchScript for C++ |
| Minimal surface area | Core API: `compile`, `save`, `load` |
| Fallback mechanisms | Unsupported ops fall back to PyTorch execution |

### 4.3 Key Features
- Cross-platform deployment (Python and C++ via libtorch)
- FP8 post-training quantization
- Broad model coverage (diffusion, LLMs, vision architectures)
- Up to 5x latency improvement vs eager execution

---

## 5. Popular Open-Source TensorRT Wrappers

### 5.1 tensorrtx (7.7k stars) - C++ Network Definition API
- Implements popular models using TensorRT's network definition API directly (not ONNX)
- Custom `.wts` weight file format for weight extraction from PyTorch
- CLI pattern: `-s` for serialization, `-d` for deserialization and inference
- Supports: YOLO family, ResNet, EfficientNet, DETR, ArcFace, U-Net, CRNN, and many more
- Custom plugins for unsupported operations (Mish, Hard Sigmoid, YOLO decode layers)
- Integrates pre/post-processing into TensorRT network graph

### 5.2 tensorRT_Pro (2.9k stars) - Production C++ Library

**Key Design Patterns:**

| Pattern | Usage |
|---------|-------|
| Factory Method | `create_infer()` encapsulates engine construction |
| Future/Promise | `commit().get()` for async inference |
| Producer-Consumer | Decouples submission from execution |
| Strategy | Type enum selects decoding strategy per model family |
| Facade | Single high-level API hides TensorRT complexity |
| RAII | Smart pointers manage engine lifetime |

**API Design (3 lines of code):**
```cpp
auto engine = Yolo::create_infer("yolox_m.fp32.trtmodel", Yolo::Type::X, 0);
auto image = cv::imread("1.jpg");
auto box = engine->commit(image).get(); // returns vector<Box>
```

**Thread Management:**
- Producer-consumer architecture
- `commit()` returns future-like object
- Background worker thread processes inference requests asynchronously
- Pipelined execution: preprocessing overlaps with inference

**Features:**
- FP32, FP16, INT8 precision support
- Multi-model support (RetinaFace, YOLO family, ArcFace, AlphaPose, etc.)
- Python bindings via pybind
- Docker support
- RESTful server example
- INT8 calibration support

### 5.3 torch2trt (4.9k stars) - Python PyTorch Wrapper
- Lightweight PyTorch to TensorRT converter
- Pythonic API with minimal code changes
- Targets Jetson platforms and desktop GPUs

### 5.4 jetson-inference (8.7k stars) - Embedded Inference
- Pre-built C++ and Python inference pipelines
- Image recognition, object detection, segmentation
- Targets full Jetson lineup
- Tutorial-oriented with extensive documentation

### 5.5 Multi-Backend Frameworks
- **TNN (Tencent, 4.6k stars)**: C++ cross-platform (TensorRT, CoreML, NCNN, OpenVINO)
- **lite.ai.toolkit (4.4k stars)**: C++ unified interface, 100+ models, TensorRT/ONNX Runtime/MNN
- **mmdeploy (3.1k stars)**: Python deployment framework for OpenMMLab projects

---

## 6. Industry Best Practices for Production Inference Engines

### 6.1 Engine Building Best Practices
1. **ONNX Preparation**: Run constant folding with Polygraphy before TensorRT conversion
2. **Optimization Profiles**: Define min/opt/max dimensions for dynamic shape support
3. **Precision Selection**: Choose FP16 for best performance/accuracy tradeoff; INT8/FP8 for maximum throughput with calibration
4. **Timing Cache**: Use editable timing cache for deterministic, reproducible builds
5. **Engine Serialization**: Cache compiled engines; plan files require exact version matching across major.minor.patch.build
6. **Calibration Caches**: Reusable within a major version but not guaranteed across patches
7. **Graph Modification**: Use ONNX-GraphSurgeon to replace unsupported subgraphs with plugins

### 6.2 Error Handling Approaches

#### Logger Callback Pattern (TensorRT Standard)
```cpp
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};
```

#### Error Recorder Pattern
- Implements `IErrorRecorder` to capture and track errors during engine building and inference
- Safety-variant error recorders for functional safety use cases

#### Production Patterns
- Nullable factory returns (`create_infer()` returns nullptr on failure)
- Build-time validation (compute capability must match GPU)
- Version compatibility checks
- Return code checking macros for CUDA and TensorRT API calls

### 6.3 API Design Patterns

#### C++ API Patterns
1. **Factory Method**: Encapsulate engine construction with `create_infer()` static methods
2. **Builder Pattern**: Configure engine building with chained configuration calls
3. **RAII Wrappers**: Smart pointers with custom deleters for TensorRT objects
4. **Future/Promise**: Async inference submission with `commit().get()`
5. **Producer-Consumer**: Background worker threads for pipelined execution
6. **Strategy Pattern**: Type enums for selecting model-specific decoding strategies
7. **Facade Pattern**: Single high-level API hiding TensorRT complexity

#### Python API Patterns
1. **Minimal Surface Area**: Core API centers on `compile`, `save`, `load` functions
2. **PyTorch-native Integration**: Use `torch.compile(backend="tensorrt")`
3. **Dual Serialization**: ExportedProgram for Python, TorchScript for C++ deployment
4. **Fallback Mechanisms**: Unsupported ops automatically fall back to PyTorch

### 6.4 Performance Optimization Techniques

#### Engine-Level
- Layer fusion (conv+bias+ReLU automatic fusion)
- Kernel auto-tuning via timing cache
- Mixed precision inference (FP16/INT8/FP8)
- Workspace memory configuration for tactic selection
- Multi-instance GPU (MIG) for low-utilization workloads

#### Runtime-Level
- CUDA Graphs to reduce kernel launch overhead (up to 22% throughput increase)
- Overlap Scheduler to hide CPU latency behind GPU work
- Paged KV caching for memory efficiency in LLM inference
- Inflight (continuous) batching for LLM serving
- Dynamic batching combining individual requests into batches
- CUDA stream management for async execution with `enqueueV3()`

#### Data Pipeline
- CUDA-accelerated preprocessing (WarpAffine, normalization)
- Integration of pre/post-processing into TensorRT network graph
- Hardware video decode integration
- NVIDIA DALI for high-performance preprocessing pipelines

### 6.5 Multi-GPU & Scaling Strategies

| Strategy | When to Use | Key Consideration |
|----------|------------|-------------------|
| Tensor Parallelism | Small batches, memory-constrained | High inter-GPU bandwidth needed |
| Pipeline Parallelism | Very large models | Sequential layer assignment |
| Data Parallelism | Large batches, high throughput | Full model replicated per GPU |
| Expert Parallelism | MoE architectures | Load balancing critical |
| Context Parallelism | Long sequences | Distributes sequence processing |
| Instance Groups (Triton) | Multiple concurrent requests | Configure instance count per model |
| MIG Partitioning | Multiple small models on one GPU | Application-specific partitioning |

### 6.6 Production Deployment Patterns

1. **Containerized Deployment**: Docker containers are the recommended approach (Triton, TensorRT-LLM)
2. **Kubernetes with Helm**: For cloud-scale deployment (GCP, AWS, NVIDIA FleetCommand)
3. **Model Versioning**: Filesystem-based version directories in model repository
4. **Health & Observability**: Readiness/liveness endpoints, GPU utilization metrics, throughput/latency metrics
5. **Protocol Support**: HTTP/REST and gRPC via KServe protocol
6. **In-Process API**: C API for edge and embedded use cases without server overhead
7. **RESTful Server Pattern**: Dedicated REST API for serving inference results (as in tensorRT_Pro)

### 6.7 Profiling & Debugging
- **NVIDIA Nsight Systems**: Primary profiling tool for TensorRT integration
- **Nsight Deep Learning Designer**: IDE for ONNX model editing, profiling, and engine building
- **Performance Analyzer (Triton)**: Benchmarking throughput and latency
- **Model Analyzer (Triton)**: Configuration optimization through profiling
- **Debug Tensor Writer**: Writing intermediate tensor values to disk for debugging
- **Valgrind and Clang sanitizers**: Code analysis tools referenced in troubleshooting

---

## 7. Key Takeaways for Our Implementation

### From Official TensorRT Samples
- Use RAII wrappers with custom deleters for all TensorRT objects
- Implement `ILogger` interface for centralized logging
- Use the `samples/common` pattern of shared infrastructure libraries
- Support both ONNX parser and network definition API paths

### From Triton Inference Server
- Adopt a plugin/backend architecture for extensibility
- Use per-model configuration for scheduling and batching strategies
- Build in health endpoints and observability from the start
- Support model versioning and dynamic model loading/unloading

### From TensorRT-LLM
- Design for multi-GPU from the start with parallelism abstractions
- Use CUDA Graphs to reduce kernel launch overhead
- Implement overlap scheduling to hide CPU latency
- Support paged memory management for large model inference

### From tensorRT_Pro
- Factory method pattern for simple 3-line inference API
- Future/Promise pattern for async inference submission
- Producer-consumer architecture for pipelined execution
- Support FP32/FP16/INT8 precision modes
- Provide Python bindings alongside C++ API

### From torch-tensorrt
- Minimal API surface area (`compile`, `save`, `load`)
- Support dual serialization formats (Python and C++)
- Implement fallback mechanisms for unsupported operations
- Follow PyTorch-native conventions

### Architecture Recommendations
1. **Modular Backend Architecture**: Follow Triton's plugin model for framework extensibility
2. **Layered API Design**: High-level Python API wrapping performant C++ core
3. **Async Inference Pipeline**: Producer-consumer with future/promise return types
4. **Comprehensive Error Handling**: Logger callback + error recorder + nullable factory returns
5. **Built-in Observability**: Metrics, health checks, profiling integration
6. **Dynamic Shape Support**: Optimization profiles with min/opt/max dimensions
7. **Engine Caching**: Serialize built engines with version compatibility checks
8. **Multi-Precision Support**: FP32, FP16, INT8, FP8 with calibration support
9. **Multi-GPU Ready**: Design abstractions for TP, PP, DP from the start
10. **Production Deployment**: Docker containers, Kubernetes support, REST/gRPC APIs
