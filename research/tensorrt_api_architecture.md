# TensorRT API Architecture Reference

## Table of Contents

1. [Overview](#overview)
2. [C++ API Structure (nvinfer1 Namespace)](#c-api-structure)
3. [Core Object Lifecycle](#core-object-lifecycle)
4. [IBuilderConfig Options](#ibuilderconfig-options)
5. [Dynamic Shapes and Optimization Profiles](#dynamic-shapes-and-optimization-profiles)
6. [Plugin System and Custom Layers](#plugin-system-and-custom-layers)
7. [DLA (Deep Learning Accelerator) Support](#dla-support)
8. [Memory Allocation Strategies](#memory-allocation-strategies)
9. [Serialization and Deserialization](#serialization-and-deserialization)
10. [ONNX Parser Capabilities and Limitations](#onnx-parser)
11. [Model Conversion Paths](#model-conversion-paths)
12. [CUDA Stream Management](#cuda-stream-management)
13. [Error Handling Patterns](#error-handling-patterns)
14. [Advanced Features](#advanced-features)

---

## 1. Overview <a name="overview"></a>

NVIDIA TensorRT is an SDK for optimizing and accelerating deep learning inference on NVIDIA GPUs. It takes trained models from frameworks such as PyTorch, TensorFlow, and ONNX, and optimizes them for high-performance deployment with support for mixed precision (FP32/FP16/BF16/FP8/INT8), dynamic shapes, and specialized optimizations for transformers and LLMs.

### Two-Phase Architecture

TensorRT operates in two distinct phases:

1. **Build Phase**: Takes a network definition (from ONNX or API), applies optimizations (layer fusion, kernel selection, precision calibration, memory planning), and produces a serialized engine (plan file).
2. **Runtime Phase**: Deserializes the plan file into an engine, creates execution contexts, and runs inference.

### Version Information

- Current version: TensorRT 10.x (10.15.1 latest as of early 2025)
- API follows Semantic Versioning 2.0.0
- Plan files require exact version matching (major, minor, patch, build) unless version compatibility is enabled
- Hardware: Supports Ampere, Ada Lovelace, Hopper, Blackwell GPUs; Volta (SM 7.0) dropped after 10.4

### Complementary Tools

| Tool | Role |
|------|------|
| Triton Inference Server | Serving with REST/gRPC endpoints |
| NVIDIA DALI | Preprocessing pipeline integration |
| Torch-TensorRT | PyTorch module to TensorRT engine conversion |
| Model Optimizer | Quantization, pruning, distillation |
| Nsight Systems | GPU profiling |
| Nsight Deep Learning Designer | ONNX editing, profiling, engine building IDE |
| Polygraphy | Debugging and validation tool |
| ONNX-GraphSurgeon | ONNX graph manipulation |

---

## 2. C++ API Structure (nvinfer1 Namespace) <a name="c-api-structure"></a>

All TensorRT C++ API classes reside in the `nvinfer1` namespace. The ONNX parser uses `nvonnxparser`.

### Core Classes

| Class | Purpose | Creation |
|-------|---------|----------|
| `IBuilder` | Factory for networks, configs, and serialized engines | `nvinfer1::createInferBuilder(logger)` |
| `INetworkDefinition` | Holds the parsed/constructed network graph | `builder->createNetworkV2(flags)` |
| `IBuilderConfig` | Build-time configuration (precision, profiles, memory) | `builder->createBuilderConfig()` |
| `IOptimizationProfile` | Min/opt/max shapes for dynamic dimensions | `builder->createOptimizationProfile()` |
| `IHostMemory` | Serialized engine plan (host memory blob) | `builder->buildSerializedNetwork(network, config)` |
| `IRuntime` | Deserializes plans into engines | `nvinfer1::createInferRuntime(logger)` |
| `ICudaEngine` | Optimized inference engine | `runtime->deserializeCudaEngine(data, size)` |
| `IExecutionContext` | Stateful inference execution handle | `engine->createExecutionContext()` |
| `IRefitter` | Updates engine weights without rebuilding | `nvinfer1::createInferRefitter(engine, logger)` |
| `IEngineInspector` | Inspects engine internals (layers, formats, tactics) | `engine->createEngineInspector()` |

### Parser Classes (nvonnxparser)

| Class | Purpose |
|-------|---------|
| `IParser` | Parses ONNX models into INetworkDefinition |
| `IParserRefitter` | Refits engine weights from ONNX model |

### Logger Interface

```cpp
class ILogger {
public:
    enum class Severity {
        kINTERNAL_ERROR = 0,
        kERROR = 1,
        kWARNING = 2,
        kINFO = 3,
        kVERBOSE = 4
    };
    virtual void log(Severity severity, char const* msg) noexcept = 0;
};
```

The logger is the first object created and is passed to `createInferBuilder()` and `createInferRuntime()`. All diagnostic, warning, and error messages flow through it.

---

## 3. Core Object Lifecycle <a name="core-object-lifecycle"></a>

### Build Phase Pipeline

```
Logger -> Builder -> Network + Config + Parser -> Serialized Engine (Plan)
```

```cpp
// 1. Create logger (user-implemented)
class MyLogger : public nvinfer1::ILogger {
    void log(Severity severity, char const* msg) noexcept override {
        // handle log message
    }
} logger;

// 2. Create builder
auto builder = std::unique_ptr<nvinfer1::IBuilder>(
    nvinfer1::createInferBuilder(logger));

// 3. Create network (strongly typed recommended in TRT 10+)
auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
    builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));

// 4. Create and use ONNX parser
auto parser = std::unique_ptr<nvonnxparser::IParser>(
    nvonnxparser::createParser(*network, logger));
parser->parseFromFile("model.onnx",
    static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

// 5. Create builder config
auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
    builder->createBuilderConfig());

// 6. Set profiling stream
cudaStream_t profileStream;
cudaStreamCreate(&profileStream);
config->setProfileStream(profileStream);

// 7. Build serialized engine
std::unique_ptr<nvinfer1::IHostMemory> plan(
    builder->buildSerializedNetwork(*network, *config));

// 8. Save to file
std::ofstream engineFile("model.engine", std::ios::binary);
engineFile.write(static_cast<char*>(plan->data()), plan->size());
```

### Runtime Phase Pipeline

```
Logger -> Runtime -> Engine -> ExecutionContext -> Inference
```

```cpp
// 1. Create runtime
auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
    nvinfer1::createInferRuntime(logger));

// 2. Deserialize engine from plan
auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(planData, planSize));

// 3. Create execution context
auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
    engine->createExecutionContext());

// 4. Set tensor addresses and execute
for (int32_t i = 0; i < engine->getNbIOTensors(); i++) {
    auto const name = engine->getIOTensorName(i);
    context->setTensorAddress(name, deviceBuffers[name]);
}

// 5. Run inference
bool success = context->enqueueV3(stream);
```

### Object Ownership Rules

- All TensorRT objects are created via factory methods and return raw pointers
- Wrap in `std::unique_ptr` for RAII management
- The IRuntime must outlive all engines it deserializes
- The ICudaEngine must outlive all execution contexts created from it
- IHostMemory from `buildSerializedNetwork` can be freed after saving/deserializing
- The IBuilder, INetworkDefinition, and IParser can be freed after building the serialized engine

### Network Definition Modes

**Strongly Typed (recommended in TRT 10+)**:
```cpp
builder->createNetworkV2(
    1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
```
- TensorRT adheres to precision from the model
- Layer-level `setPrecision`/`setOutputType` and builder precision flags (`kFP16`, `kINT8`, etc.) are NOT permitted
- `kTF32` is allowed (controls Tensor Core usage for FP32)
- NOT supported with DLA

**Weakly Typed (deprecated)**:
- Builder precision flags control which reduced precisions TensorRT may autoselect
- TensorRT may still choose higher precision if faster
- Layer-level precision hints can be set

---

## 4. IBuilderConfig Options <a name="ibuilderconfig-options"></a>

### Builder Flags

```cpp
config->setFlag(BuilderFlag::kFLAG_NAME);
config->clearFlag(BuilderFlag::kFLAG_NAME);
```

| Flag | Purpose |
|------|---------|
| `kFP16` | Enable FP16 precision (weakly typed only) |
| `kBF16` | Enable BF16 precision (Ampere+, weakly typed only) |
| `kINT8` | Enable INT8 precision (weakly typed only) |
| `kFP8` | Enable FP8 precision |
| `kINT4` | Enable INT4 weight-only quantization |
| `kFP4` | Enable FP4 (E2M1) |
| `kTF32` | Use TF32 Tensor Cores for FP32 (enabled by default) |
| `kSPARSE_WEIGHTS` | Enable structured sparsity (2:4 pattern, Ampere+) |
| `kREFIT` | Mark all weights as refittable |
| `kREFIT_IDENTICAL` | Optimize assuming refit weights equal build weights |
| `kREFIT_INDIVIDUAL` | Fine-grained per-weight refit control |
| `kSTRIP_PLAN` | Strip refittable weights from serialized plan |
| `kVERSION_COMPATIBLE` | Forward-compatible engine with lean runtime |
| `kEXCLUDE_LEAN_RUNTIME` | Don't embed lean runtime in plan |
| `kWEIGHT_STREAMING` | Enable weight streaming (host-to-device on demand) |
| `kEDITABLE_TIMING_CACHE` | Enable editable timing cache for reproducible builds |
| `kPREFER_PRECISION_CONSTRAINTS` | Prefer but don't require layer precision hints |
| `kOBEY_PRECISION_CONSTRAINTS` | Strictly enforce layer precision hints |
| `kDIRECT_IO` | Avoid reformatting at network I/O boundaries |

### Memory Configuration

```cpp
// Set memory pool limits
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB workspace
config->setMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM, size);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM, size);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM, size);
```

### Profiling and Timing Cache

```cpp
// Set profiling stream for builder autotuning
config->setProfileStream(stream);

// Timing cache for faster rebuilds
ITimingCache* cache = config->createTimingCache(cacheData, cacheSize);
config->setTimingCache(*cache, false); // false = don't ignore mismatch
```

### Hardware Compatibility

```cpp
// Build engine compatible with all Ampere+ GPUs
config->setHardwareCompatibilityLevel(
    nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS);

// Same compute capability (better perf, less portable)
config->setHardwareCompatibilityLevel(
    nvinfer1::HardwareCompatibilityLevel::kSAME_COMPUTE_CAPABILITY);
```

### Cross-Platform Engine Building

```cpp
// Build on Linux for Windows deployment
config->setRuntimePlatform(nvinfer1::RuntimePlatform::kWINDOWS_AMD64);
```

### Tiling Optimization

```cpp
// Enable cross-kernel tiling (0 = disabled, higher = more exploration)
config->setTilingOptimizationLevel(level);

// Hint for L2 cache in multi-task scenarios
config->setL2LimitForTiling(cacheSize);
```

### Profiling Verbosity

```cpp
config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);  // Full detail
config->setProfilingVerbosity(ProfilingVerbosity::kLAYER_NAMES_ONLY);  // Default
config->setProfilingVerbosity(ProfilingVerbosity::kNONE);  // Minimal
```

### DLA Configuration

```cpp
config->setDefaultDeviceType(DeviceType::kDLA);
config->setDLACore(0);  // Use DLA core 0
config->setFlag(BuilderFlag::kGPU_FALLBACK);  // Fall back to GPU for unsupported layers
```

---

## 5. Dynamic Shapes and Optimization Profiles <a name="dynamic-shapes-and-optimization-profiles"></a>

### Overview

Dynamic shapes allow input tensors to have variable dimensions at runtime. Dimensions marked as `-1` in the network definition are dynamic. TensorRT requires optimization profiles that define min/opt/max ranges for each dynamic dimension.

### Creating Optimization Profiles

```cpp
// 1. Define dynamic input in network
auto input = network->addInput("input", nvinfer1::DataType::kFLOAT,
    Dims4{-1, 3, -1, -1});  // batch, channel, height, width dynamic

// 2. Create optimization profile
auto profile = builder->createOptimizationProfile();

// 3. Set min, optimal, and max dimensions
profile->setDimensions(input->getName(),
    OptProfileSelector::kMIN, Dims4{1, 3, 224, 224});
profile->setDimensions(input->getName(),
    OptProfileSelector::kOPT, Dims4{4, 3, 224, 224});  // TRT optimizes for this
profile->setDimensions(input->getName(),
    OptProfileSelector::kMAX, Dims4{16, 3, 640, 640});

// 4. Add profile to config (can add multiple profiles)
config->addOptimizationProfile(profile);
```

### Using Profiles at Runtime

```cpp
// Set actual input dimensions before inference
context->setInputShape("input", Dims4{8, 3, 416, 416});

// Verify all dynamic dimensions are resolved
if (!context->allInputDimensionsSpecified()) {
    // error: some dynamic dimensions not set
}

// Execute
context->enqueueV3(stream);
```

### Querying Profile Dimensions

```cpp
auto const name = engine->getIOTensorName(0);
Dims minDims = engine->getProfileShape(name, 0, OptProfileSelector::kMIN);
Dims optDims = engine->getProfileShape(name, 0, OptProfileSelector::kOPT);
Dims maxDims = engine->getProfileShape(name, 0, OptProfileSelector::kMAX);
```

### Multiple Optimization Profiles

Multiple profiles can be added for different operating ranges. Each profile is independently optimized by TensorRT. At runtime, the appropriate profile is selected:

```cpp
// Profile 0: small images
profile0->setDimensions(name, kMIN, Dims4{1,3,224,224});
profile0->setDimensions(name, kOPT, Dims4{1,3,224,224});
profile0->setDimensions(name, kMAX, Dims4{1,3,224,224});
config->addOptimizationProfile(profile0);

// Profile 1: variable batch, large images
profile1->setDimensions(name, kMIN, Dims4{1,3,640,640});
profile1->setDimensions(name, kOPT, Dims4{8,3,640,640});
profile1->setDimensions(name, kMAX, Dims4{32,3,640,640});
config->addOptimizationProfile(profile1);

// At runtime, select profile
context->setOptimizationProfileAsync(1, stream);  // Use profile 1
```

### Key Considerations

- The `kOPT` dimensions are what TensorRT optimizes kernel selection for
- Wider min-max ranges may lead to less optimal kernel choices
- Each profile adds to build time and engine size
- All dynamic dimensions must be specified before inference
- Shape tensors (for operations like Reshape) also need profile ranges

---

## 6. Plugin System and Custom Layers <a name="plugin-system-and-custom-layers"></a>

### IPluginV3 (Recommended, TRT 10+)

IPluginV3 is the only recommended plugin interface starting in TensorRT 10.0. All older interfaces (IPluginV2, IPluginV2DynamicExt) are deprecated.

IPluginV3 is a wrapper for three capability interfaces:

| Capability | Interface | Purpose |
|-----------|-----------|---------|
| Core | `IPluginV3OneCore` | Plugin attributes common to build and runtime |
| Build | `IPluginV3OneBuild` | Attributes needed by the builder |
| Runtime | `IPluginV3OneRuntime` | Attributes for execution |

Optional: `IPluginV3OneBuildV2` adds I/O aliasing support.

### Plugin Implementation Pattern

```cpp
class MyPlugin : public IPluginV3, public IPluginV3OneCore,
                 public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    // --- IPluginV3 ---
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    // --- IPluginV3OneCore ---
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    IPluginV3* clone() noexcept override;

    // --- IPluginV3OneBuild ---
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
                            int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override;
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) noexcept override;
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) noexcept override;

    // --- IPluginV3OneRuntime ---
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
                          PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
};
```

### Plugin Creator

```cpp
class MyPluginCreator : public IPluginCreatorV3One
{
public:
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;
    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override;
};
```

### Plugin Registration

**Static registration** (always default namespace):
```cpp
REGISTER_TENSORRT_PLUGIN(MyPluginCreator);
```

**Dynamic registration** (supports custom namespaces, preferred):
```cpp
getPluginRegistry()->registerCreator(creator, "my_namespace");
```

### Key Build Methods

| Method | Purpose |
|--------|---------|
| `getNbOutputs()` | Number of output tensors |
| `getOutputShapes()` | Symbolic output dimension expressions |
| `getOutputDataTypes()` | Output data types |
| `supportsFormatCombination()` | Check supported type/format at each I/O |
| `configurePlugin()` | Called before profiling with min/max/opt info |
| `getWorkspaceSize()` | Request scratch memory |
| `getTimingCacheId()` | Enable timing caching (opt-in) |
| `getValidTactics()` | Advertise custom tactics for autotuning |

### Key Runtime Methods

| Method | Purpose |
|--------|---------|
| `onShapeChange()` | Called before enqueue with concrete shapes |
| `enqueue()` | Execute the plugin's CUDA kernels |
| `setTactic()` | Communicate chosen tactic |
| `getFieldsToSerialize()` | Return PluginFieldCollection for serialization |
| `attachToContext()` | Clone and attach to an ExecutionContext |

### Serialization Model (V3)

V3 replaces raw buffer serialization with structured `PluginFieldCollection`:
- **Serialize**: `getFieldsToSerialize()` returns a `PluginFieldCollection`
- **Deserialize**: Same collection passed to `createPlugin(..., TensorRTPhase::kRUNTIME)`

### Data-Dependent Output Shapes

IPluginV3 introduces data-dependent shapes (DDS) via size tensors:

```cpp
// In getOutputShapes():
auto sizeTensor = exprBuilder.declareSizeTensor(outputIndex, upperBound, optValue);
```

### ONNX Parser Integration

The parser maps unrecognized ONNX nodes to plugins by matching `op_type`:
- Default version: `"1"`
- Default namespace: `""`
- `kENABLE_PLUGIN_OVERRIDE` flag gives plugins precedence over built-in ops

### Migration from V2 to V3

Key changes:
- No `initialize()`, `terminate()`, or `destroy()` -- construct initialized, clean up in destructor
- `attachToContext` is clone-and-attach (no cuDNN/cuBLAS handles provided)
- `getOutputDimensions` -> `getOutputShapes` (one-shot signature)
- `configurePlugin` and `getWorkspaceSize` receive `DynamicPluginTensorDesc`
- Void return methods now expect integer status codes

---

## 7. DLA (Deep Learning Accelerator) Support <a name="dla-support"></a>

### Overview

DLA is a dedicated inference processor on NVIDIA SoCs (Jetson, DRIVE). It supports a subset of TensorRT layers and can execute parts of a network while the GPU handles the rest.

### Configuration

```cpp
// Set DLA as default device
config->setDefaultDeviceType(DeviceType::kDLA);
config->setDLACore(0);  // Core 0 or 1

// Enable GPU fallback for unsupported layers
config->setFlag(BuilderFlag::kGPU_FALLBACK);

// Per-layer device assignment
layer->setDeviceType(DeviceType::kDLA);
```

### DLA Supported Layers (Subset)

DLA supports a limited set of operations with specific restrictions:
- Convolution, Deconvolution
- Pooling (Average, Max)
- BatchNormalization (fused with Conv)
- Activation (ReLU, Sigmoid, Tanh)
- ElementWise operations
- Scale layers
- Concatenation
- Softmax (limited)

Each layer type has restrictions on parameters, dimensions, and data types.

### GPU Fallback Mode

When a layer cannot execute on DLA, TensorRT automatically falls back to GPU if `kGPU_FALLBACK` is set. Without this flag, the build fails if any layer in the DLA partition is unsupported.

### DLA Memory Pools

```cpp
config->setMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM, sramSize);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM, localDramSize);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM, globalDramSize);
```

### DLA Standalone Mode

TensorRT can generate standalone DLA loadables that run independently:

```cpp
config->setFlag(BuilderFlag::kDLA_STANDALONE);
```

### I/O Formats on DLA

DLA requires specific tensor formats: `kDLA_LINEAR`, `kDLA_HWC4`. Network inputs/outputs must conform to DLA-supported formats.

### Limitations

- Strongly typed networks are NOT supported with DLA
- Hardware compatibility mode not supported on DRIVE OS or JetPack
- Structured sparsity supported on DLA (Orin+)
- Limited precision support vs GPU

---

## 8. Memory Allocation Strategies <a name="memory-allocation-strategies"></a>

### Memory Pool Configuration

```cpp
// Workspace memory limit (used during build and inference)
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

// DLA memory pools (for Jetson/DRIVE platforms)
config->setMemoryPoolLimit(MemoryPoolType::kDLA_MANAGED_SRAM, size);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_LOCAL_DRAM, size);
config->setMemoryPoolLimit(MemoryPoolType::kDLA_GLOBAL_DRAM, size);
```

### Custom GPU Allocator (IGpuAllocator)

Implement `IGpuAllocator` for custom device memory management:

```cpp
class MyAllocator : public nvinfer1::IGpuAllocator {
    void* allocate(uint64_t size, uint64_t alignment, uint32_t flags) noexcept override;
    bool deallocate(void* memory) noexcept override;
};

// Attach to builder or runtime
builder->setGpuAllocator(&allocator);
runtime->setGpuAllocator(&allocator);
```

### Weight Streaming

For models that exceed GPU memory, weights can be streamed from host:

**Build time**:
```cpp
config->setFlag(BuilderFlag::kWEIGHT_STREAMING);
```

**Runtime**:
```cpp
// Set device memory budget for weights
engine->setWeightStreamingBudgetV2(budgetBytes);

// Query streaming metrics
int64_t streamableSize = engine->getStreamableWeightsSize();
int64_t scratchSize = engine->getWeightStreamingScratchMemorySize();
int64_t contextSize = engine->getDeviceMemorySizeV2();
int64_t autoBudget = engine->getWeightStreamingAutomaticBudget();
```

If the budget exceeds total streamable weight size, streaming is effectively disabled. Budget can be readjusted when no active context exists.

### Output Allocator

For dynamic output shapes where buffer sizes are unknown at setup:

```cpp
class MyOutputAllocator : public nvinfer1::IOutputAllocator {
    void* reallocateOutputAsync(char const* tensorName, void* currentMemory,
        uint64_t size, uint64_t alignment, cudaStream_t stream) override;
    void notifyShape(char const* tensorName, Dims const& dims) noexcept override;
};

context->setOutputAllocator("output_name", &allocator);
```

### Input Buffer Reuse

```cpp
// Signal when input buffers are safe to reuse (for pipelining)
cudaEvent_t inputReady;
cudaEventCreate(&inputReady);
context->setInputConsumedEvent(&inputReady);
```

### Empty Tensors

Tensors with zero-length dimensions are supported. They still require non-null, unique memory addresses.

---

## 9. Serialization and Deserialization <a name="serialization-and-deserialization"></a>

### Building and Saving a Serialized Engine

```cpp
// Method 1: In-memory serialization
std::unique_ptr<IHostMemory> plan(
    builder->buildSerializedNetwork(*network, *config));
std::ofstream file("model.engine", std::ios::binary);
file.write(static_cast<char*>(plan->data()), plan->size());

// Method 2: Stream-based serialization (direct to file, lower memory)
FileStreamWriter writer("model.engine");
builder->buildSerializedNetworkToStream(*network, *config, writer);
```

### Loading and Deserializing

```cpp
// Read plan file
std::ifstream file("model.engine", std::ios::binary);
file.seekg(0, std::ios::end);
size_t size = file.tellg();
file.seekg(0, std::ios::beg);
std::vector<char> plan(size);
file.read(plan.data(), size);

// Deserialize
auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
auto engine = std::unique_ptr<ICudaEngine>(
    runtime->deserializeCudaEngine(plan.data(), plan.size()));
```

### Streaming Deserialization (IStreamReader)

For large engines, avoid loading the entire plan into memory:

```cpp
// File-based streaming reader
auto engine = runtime->deserializeCudaEngine(fileReader);

// Async file reader
auto engine = runtime->deserializeCudaEngine(asyncFileReader);
```

### Re-serialization from Engine

```cpp
// Serialize an existing engine
std::unique_ptr<IHostMemory> serialized(engine->serialize());

// With custom serialization config
auto serConfig = engine->createSerializationConfig();
serConfig->setFlags(flags);  // e.g., EXCLUDE_WEIGHTS, INCLUDE_REFIT
auto serialized = engine->serializeWithConfig(*serConfig);
```

### Version Compatibility

```cpp
// Build time: enable forward compatibility
config->setFlag(BuilderFlag::kVERSION_COMPATIBLE);

// Runtime: allow embedded host code (required for version-compatible plans)
runtime->setEngineHostCodeAllowed(true);

// Optionally exclude lean runtime from plan
config->setFlag(BuilderFlag::kEXCLUDE_LEAN_RUNTIME);

// Load specific lean runtime at deserialization
auto shimRuntime = runtime->loadRuntime(leanRuntimePath);
auto engine = shimRuntime->deserializeCudaEngine(plan);
```

### Plan File Format

- Magic number: `0x74727466` ("ftrt")
- Version info at bytes 24-27 (major, minor, patch, build)
- Plans are NOT portable across TensorRT versions (unless version compatibility enabled)
- Plans are NOT portable across GPU architectures (unless hardware compatibility enabled)

### Engine Refitting

```cpp
// Build refittable engine
config->setFlag(BuilderFlag::kREFIT);
auto plan = builder->buildSerializedNetwork(*network, *config);

// Refit weights
auto refitter = std::unique_ptr<IRefitter>(createInferRefitter(*engine, logger));
Weights newWeights = { DataType::kFLOAT, weightData, weightCount };
refitter->setNamedWeights("layer_name", newWeights);

// Check for missing dependent weights
int32_t n = refitter->getMissingWeights(0, nullptr);
std::vector<const char*> missing(n);
refitter->getMissingWeights(n, missing.data());
// Supply missing weights...

// Apply refit
bool success = refitter->refitCudaEngine();

// Async refit with CUDA stream
bool success = refitter->refitCudaEngineAsync(stream);
```

### Weight-Stripped Engines

For minimal distribution:

```cpp
// Build time
config->setFlag(BuilderFlag::kSTRIP_PLAN);
config->setFlag(BuilderFlag::kREFIT_IDENTICAL);

// Client side: refit from ONNX
auto refitter = createInferRefitter(*engine, logger);
auto parserRefitter = createParserRefitter(*refitter, logger);
parserRefitter->refitFromFile("model.onnx");
refitter->refitCudaEngine();
```

### Fine-Grained Refit

```cpp
// Mark specific weights as refittable (preserves optimizations for others)
network->setWeightsName(weights, "conv1_filter");
network->markWeightsRefittable("conv1_filter");
config->setFlag(BuilderFlag::kREFIT_INDIVIDUAL);
```

---

## 10. ONNX Parser Capabilities and Limitations <a name="onnx-parser"></a>

### Supported Opset Range

TensorRT 10.x ONNX parser supports **opset 9 through opset 20**. The GitHub version may support later opsets than the version shipped with TensorRT.

### Supported Data Types

FP32, FP16, BF16, INT32, INT64, FP8, INT8, INT4, UINT8, BOOL, DOUBLE (downcast to FP32)

### Widely Supported Operators (Partial List)

The parser supports 100+ ONNX operators. Key categories:

**Fully supported**: Abs, Add, ArgMax, ArgMin, AveragePool (2D/3D), BatchNormalization, Cast, Ceil, Clip, Concat, Constant, Conv, ConvTranspose, Cos, DepthToSpace, Div, Dropout, Einsum, Elu, Equal, Erf, Exp, Expand, Flatten, Floor, Gather, GatherElements, GatherND, Gelu, Gemm, GlobalAveragePool, Greater, GroupNormalization, HardSigmoid, HardSwish, Identity, If, InstanceNormalization, LayerNormalization, LeakyRelu, Less, Log, LogSoftmax, Loop, LRN, MatMul, Max, Mean, Min, Mod, Mul, Neg, NonMaxSuppression, NonZero, Not, OneHot, Or, Pad, Pow, PRelu, Range, Reciprocal, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, Relu, Reshape, Resize, ReverseSequence, RoiAlign, Round, Scan, ScatterElements, ScatterND, Selu, Shape, Sigmoid, Sign, Sin, Size, Slice, Softmax, Softplus, SpaceToDepth, Split, Sqrt, Squeeze, Sub, Sum, Tanh, Tile, TopK, Transpose, Trilu, Unsqueeze, Upsample, Where, Xor

### Operators with Restrictions

| Operator | Restriction |
|----------|-------------|
| AveragePool | 2D or 3D only; dilations must be 1 |
| CumSum | axis must be an initializer |
| DequantizeLinear | zero_point must be zero |
| GRU/LSTM/RNN | Bidirectional: activations must match forward/reverse |
| Hardmax | axis dimension must be build-time constant |
| MaxPool | 2D/3D only; no indices output; dilations must be 1 |
| QuantizeLinear | zero_point must be 0 |
| Random* | seed value ignored |
| Reduce* | axes must be initializer |
| Resize | Limited transform modes; stretch aspect ratio only; no antialiasing |
| ScatterND | reduction not supported |
| Slice | axes must be initializer |
| Squeeze/Unsqueeze | axes must be initializer/constant |

### Unsupported Operators

AffineGrid, Bernoulli, BitShift, Bitwise*, BlackmanWindow, Col2Im, Compress, ConvInteger, DFT, Det, DynamicQuantizeLinear, HammingWindow, HannWindow, ImageDecoder, MatMulInteger, MaxRoiPool, MaxUnpool, MelWeightMatrix, Multinomial, NegativeLogLikelihoodLoss, Optional*, QLinear*, RegexFullMatch, Sequence*, SoftmaxCrossEntropyLoss, STFT, String*, TfIdfVectorizer, Unique

### Key Limitations

1. Sequence and Optional types are entirely unsupported
2. String operations not supported
3. Bitwise operations not supported
4. Many operators require axes/parameters as compile-time constants (initializers), not dynamic inputs
5. INT8/FP8/INT4 require explicit Q/DQ nodes with zero-point = 0

### Best Practices

- Export to the latest available ONNX opset
- Run constant folding with Polygraphy after export
- Use ONNX-GraphSurgeon to replace unsupported subgraphs with plugins
- When using version-compatible engines, use `kNATIVE_INSTANCENORM` parser flag

---

## 11. Model Conversion Paths <a name="model-conversion-paths"></a>

### PyTorch to TensorRT

**Path 1: PyTorch -> ONNX -> TensorRT (most common)**

```python
import torch

model = MyModel().eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,       # Use latest supported opset
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"}
    },
)
```

Key considerations:
- Use `opset_version` >= 13 for broadest TensorRT support
- Specify `dynamic_axes` for dimensions that should be variable
- Run `torch.onnx.dynamo_export()` (PyTorch 2.x) as an alternative exporter
- Validate with ONNX Runtime before TensorRT conversion
- Use `torch.onnx.verification` to compare outputs

**Path 2: Torch-TensorRT (direct)**

```python
import torch_tensorrt

# TorchScript path
model_ts = torch.jit.trace(model, dummy_input)
trt_model = torch_tensorrt.compile(model_ts,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 224, 224],
        opt_shape=[4, 3, 224, 224],
        max_shape=[16, 3, 224, 224],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float16}
)

# FX/Dynamo path (PyTorch 2.x)
optimized_model = torch_tensorrt.compile(model,
    ir="dynamo",
    inputs=[torch_tensorrt.Input(shape=[1, 3, 224, 224])]
)
```

### TensorFlow to TensorRT

**Path 1: TensorFlow -> ONNX -> TensorRT (via tf2onnx)**

Command line:
```bash
python -m tf2onnx.convert --saved-model ./saved_model_dir --output model.onnx --opset 17
```

Python API:
```python
import tf2onnx

# From Keras model
model_proto, _ = tf2onnx.convert.from_keras(
    keras_model,
    opset=17,
    output_path="model.onnx"
)

# From SavedModel
model_proto, _ = tf2onnx.convert.from_function(
    tf_function,
    input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)],
    opset=17,
    output_path="model.onnx"
)
```

tf2onnx supports:
- SavedModel, checkpoint, graphdef, TFLite, TensorFlow.js formats
- Opset 6-18 (default 15, tested 14-18)
- NHWC to NCHW transpose (`--inputs-as-nchw`)
- Large model support (external tensor storage for >2 GB)
- Python 3.7-3.12, TensorFlow 1.15 / 2.9-2.15

**Path 2: TF-TRT (TensorFlow integration)**

```python
from tensorflow.python.compiler.tensorrt import trt_convert as trt

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='./saved_model',
    conversion_params=trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=1 << 30
    )
)
converter.convert()
converter.save('./trt_saved_model')
```

### Conversion Pipeline Summary

```
PyTorch Model
    |---> torch.onnx.export() ---> ONNX ---> TensorRT (C++ or trtexec)
    |---> Torch-TensorRT (direct, supports TorchScript/FX/Dynamo)

TensorFlow Model
    |---> tf2onnx ---> ONNX ---> TensorRT (C++ or trtexec)
    |---> TF-TRT (TensorFlow-integrated TensorRT)

ONNX Model ---> TensorRT ONNX Parser ---> TensorRT Engine
```

---

## 12. CUDA Stream Management <a name="cuda-stream-management"></a>

### Build Phase Stream

```cpp
// Profile stream for builder autotuning
cudaStream_t profileStream;
cudaStreamCreate(&profileStream);
config->setProfileStream(profileStream);
```

### Inference with enqueueV3

```cpp
// Create stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// Set tensor addresses
for (int32_t i = 0; i < engine->getNbIOTensors(); i++) {
    auto const name = engine->getIOTensorName(i);
    context->setTensorAddress(name, deviceBuffers[name]);
}

// Enqueue inference on stream
bool success = context->enqueueV3(stream);

// Synchronize
cudaStreamSynchronize(stream);
```

### Multi-Stream Pipelining

TensorRT supports overlapping data transfers with computation using multiple streams:

```cpp
// Three-stream pipeline: input transfer, compute, output transfer
cudaStream_t inputStream, computeStream, outputStream;
cudaEvent_t inputDone, computeDone;

// Stage 1: Copy input to device
cudaMemcpyAsync(deviceInput, hostInput, size, cudaMemcpyHostToDevice, inputStream);
cudaEventRecord(inputDone, inputStream);

// Stage 2: Wait for input, run inference
cudaStreamWaitEvent(computeStream, inputDone, 0);
context->enqueueV3(computeStream);
cudaEventRecord(computeDone, computeStream);

// Stage 3: Wait for compute, copy output
cudaStreamWaitEvent(outputStream, computeDone, 0);
cudaMemcpyAsync(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost, outputStream);
```

### CUDA Graph Capture

For minimal launch overhead, capture the inference call into a CUDA graph:

```cpp
// Warm up
context->enqueueV3(stream);
cudaStreamSynchronize(stream);

// Capture
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
context->enqueueV3(stream);
cudaStreamEndCapture(stream, &graph);

// Create executable graph
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Replay (fast path)
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);
```

### Input Consumed Event

```cpp
// Notify when inputs can be safely overwritten
cudaEvent_t inputReady;
cudaEventCreate(&inputReady);
context->setInputConsumedEvent(&inputReady);

// After enqueueV3, wait for event before reusing input buffers
context->enqueueV3(stream);
cudaEventSynchronize(inputReady);
// Safe to overwrite input buffers now
```

### Key Notes

- `enqueueV3` is the current recommended async inference API
- `executeV2` is the synchronous variant (simpler but blocks CPU)
- DLA execution may cause `enqueueV3()` to behave synchronously
- Multiple execution contexts can share an engine but need separate device memory
- Each context needs its own CUDA stream for concurrent execution

---

## 13. Error Handling Patterns <a name="error-handling-patterns"></a>

### ILogger Implementation

```cpp
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL ERROR: " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "INFO: " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                // Optionally log verbose messages
                break;
        }
    }
};
```

### Error Recorder

```cpp
class MyErrorRecorder : public nvinfer1::IErrorRecorder {
    // Attach to builder or runtime for centralized error collection
};

builder->setErrorRecorder(&recorder);
runtime->setErrorRecorder(&recorder);
```

### Return Value Checking Pattern

TensorRT methods typically return `nullptr` or `false` on failure:

```cpp
auto builder = createInferBuilder(logger);
if (!builder) {
    // Fatal: cannot create builder
    return false;
}

auto network = builder->createNetworkV2(flags);
if (!network) {
    // Fatal: cannot create network
    return false;
}

auto plan = builder->buildSerializedNetwork(*network, *config);
if (!plan) {
    // Build failed - check logger for details
    return false;
}

auto engine = runtime->deserializeCudaEngine(data, size);
if (!engine) {
    // Deserialization failed - version mismatch, corruption, etc.
    return false;
}

bool success = context->enqueueV3(stream);
if (!success) {
    // Inference failed
    return false;
}
```

### CUDA Error Checking

```cpp
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl; \
            return false; \
        } \
    } while(0)

CHECK_CUDA(cudaMalloc(&buffer, size));
CHECK_CUDA(cudaMemcpyAsync(dst, src, size, kind, stream));
CHECK_CUDA(cudaStreamSynchronize(stream));
```

### Common Error Scenarios

| Scenario | Diagnostic |
|----------|-----------|
| Version mismatch | "serialized with version X.Y, current version A.B" |
| GPU mismatch | Check compute capability in logger warnings |
| OOM during build | Reduce workspace, use partitioned builder resources |
| OOM during inference | Check weight streaming, reduce batch size |
| Unsupported ONNX op | Parser error with operator name |
| Dynamic shape not set | `allInputDimensionsSpecified()` returns false |
| Plugin not found | Check plugin registry, ensure registration |

---

## 14. Advanced Features <a name="advanced-features"></a>

### Engine Inspector

```cpp
auto inspector = std::unique_ptr<IEngineInspector>(engine->createEngineInspector());
inspector->setExecutionContext(context);  // For resolved dynamic shapes

// Per-layer information
std::string layerInfo = inspector->getLayerInformation(0, LayerInformationFormat::kJSON);

// Full engine information
std::string engineInfo = inspector->getEngineInformation(LayerInformationFormat::kJSON);
```

Requires `ProfilingVerbosity::kDETAILED` for full output.

### Debug Tensors

Inspect intermediate tensors without significant overhead:

```cpp
// Build time: mark tensors
network->markDebug(&tensor);            // Specific tensor (may disable fusions)
network->markUnfusedTensorsAsDebugTensors();  // All, preserves fusions

// Runtime: listen for debug data
class MyDebugListener : public IDebugListener {
    bool processDebugTensor(void const* addr, TensorLocation location,
        DataType type, Dims const& shape, char const* name,
        cudaStream_t stream) override;
};

context->setDebugListener(&listener);
context->setTensorDebugState("tensor_name", true);
```

### Timing Cache

```cpp
// Build time: create/load timing cache
ITimingCache* cache = config->createTimingCache(existingData, size);
config->setTimingCache(*cache, false);

// After build: save timing cache
auto cacheData = std::unique_ptr<IHostMemory>(cache->serialize());
// Write cacheData to file for reuse

// Editable timing cache (for reproducible builds)
config->setFlag(BuilderFlag::kEDITABLE_TIMING_CACHE);
```

### Optimizer Progress Monitor

```cpp
class MyMonitor : public IProgressMonitor {
    void phaseStart(char const* phaseName, char const* parentPhase,
                    int32_t nbSteps) noexcept override;
    bool stepComplete(char const* phaseName, int32_t step) noexcept override;
    void phaseFinish(char const* phaseName) noexcept override;
};

config->setProgressMonitor(&monitor);
```

Returning `false` from `stepComplete()` terminates the build early.

### Reduced Precision Control (Weakly Typed)

```cpp
// Network level
config->setFlag(BuilderFlag::kFP16);  // Allow FP16

// Layer level precision hints
layer->setPrecision(DataType::kFP16);
layer->setOutputType(0, DataType::kFLOAT);

// Enforcement
config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);   // Strict
config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);  // Best-effort
```

### Structured Sparsity (2:4 Pattern)

```cpp
// Requires Ampere+ GPU, weights with 2:4 sparsity pattern
config->setFlag(BuilderFlag::kSPARSE_WEIGHTS);
config->setFlag(BuilderFlag::kFP16);  // Sparsity works with FP16 or INT8
```

### I/O Format Control

```cpp
auto formats = 1U << static_cast<uint32_t>(TensorFormat::kHWC8);
network->getInput(0)->setAllowedFormats(formats);
network->getInput(0)->setType(DataType::kHALF);
```

Supported formats: `kLINEAR`, `kCHW2`, `kCHW4`, `kHWC8`, `kCHW16`, `kCHW32`, `kDHWC8`, `kCDHW32`, `kHWC`, `kDLA_LINEAR`, `kDLA_HWC4`, `kHWC16`, `kDHWC`.

### Algorithm Selection for Reproducible Builds

TensorRT's kernel autotuning means different builds may produce numerically different results. Use the editable timing cache to lock down tactic selection for reproducibility.

### TF32 (Tensor Float 32)

Enabled by default. Rounds FP32 multiplicands to FP16 precision while keeping FP32 dynamic range:

```cpp
// Disable TF32
config->clearFlag(BuilderFlag::kTF32);

// Environment variable override
// NVIDIA_TF32_OVERRIDE=0 disables TF32 globally
```

---

## Appendix: API Quick Reference

### Minimal Build Pipeline (C++)

```cpp
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <memory>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, char const* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << msg << std::endl;
    }
} gLogger;

bool buildEngine(const char* onnxPath, const char* enginePath) {
    // Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger));
    if (!builder) return false;

    // Network
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    if (!network) return false;

    // Parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        return false;

    // Config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Dynamic shapes
    auto profile = builder->createOptimizationProfile();
    auto inputName = network->getInput(0)->getName();
    profile->setDimensions(inputName,
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(inputName,
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{4, 3, 224, 224});
    profile->setDimensions(inputName,
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{16, 3, 640, 640});
    config->addOptimizationProfile(profile);

    // Build
    std::unique_ptr<nvinfer1::IHostMemory> plan(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan) return false;

    // Save
    std::ofstream file(enginePath, std::ios::binary);
    file.write(static_cast<char*>(plan->data()), plan->size());
    return !file.fail();
}
```

### Minimal Inference Pipeline (C++)

```cpp
bool runInference(const char* enginePath) {
    // Load plan
    std::ifstream file(enginePath, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> plan(size);
    file.read(plan.data(), size);

    // Runtime + Engine
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger));
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine) return false;

    // Context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext());

    // Set dynamic input shape
    context->setInputShape("input", nvinfer1::Dims4{1, 3, 224, 224});

    // Allocate and bind buffers
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int32_t i = 0; i < engine->getNbIOTensors(); i++) {
        auto name = engine->getIOTensorName(i);
        // allocate device memory, set addresses
        context->setTensorAddress(name, deviceBuffer);
    }

    // Execute
    bool success = context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    return success;
}
```

### Header Dependencies

```cpp
#include "NvInfer.h"          // Core TensorRT API
#include "NvInferRuntime.h"    // Runtime API
#include "NvOnnxParser.h"      // ONNX parser
#include "NvInferPlugin.h"     // Built-in plugins (initLibNvInferPlugins)
```

### Link Libraries

```
-lnvinfer          # Core TensorRT
-lnvinfer_plugin   # Built-in plugins
-lnvonnxparser     # ONNX parser
-lcudart           # CUDA runtime
```
