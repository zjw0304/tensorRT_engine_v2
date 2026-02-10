# TensorRT Best Practices - Comprehensive Research Summary

> Sources:
> - [NVIDIA TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
> - [NVIDIA TensorRT Quantization Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
> - [NVIDIA TensorRT Dynamic Shapes Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)
> - [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

---

## Table of Contents

1. [Layer Fusion Strategies](#1-layer-fusion-strategies)
2. [Precision Conversion Best Practices](#2-precision-conversion-best-practices)
3. [Quantization Deep Dive](#3-quantization-deep-dive)
4. [Memory Management Guidelines](#4-memory-management-guidelines)
5. [CUDA Graph Integration](#5-cuda-graph-integration)
6. [Multi-Stream Execution](#6-multi-stream-execution)
7. [Batch Processing Optimization](#7-batch-processing-optimization)
8. [Kernel Tuning Techniques](#8-kernel-tuning-techniques)
9. [Plugin Development](#9-plugin-development)
10. [Dynamic Shapes and Optimization Profiles](#10-dynamic-shapes-and-optimization-profiles)
11. [Tensor Core Optimization](#11-tensor-core-optimization)
12. [Benchmarking and Profiling](#12-benchmarking-and-profiling)
13. [Hardware and Environment Configuration](#13-hardware-and-environment-configuration)

---

## 1. Layer Fusion Strategies

TensorRT automatically fuses layers during the build phase to reduce kernel launches and memory bandwidth overhead. Understanding these fusions is critical for designing models that maximize optimization opportunities.

### 1.1 Supported Layer Fusions

| Fusion Pattern | Requirements / Notes |
|---|---|
| **Conv + Activation (ReLU/GELU/Clip)** | For GELU: input and output precision must match (both FP16 or INT8) |
| **Conv + ElementWise (Sum/Min/Max)** | Sum must not use broadcasting (except across batch size) |
| **Conv + Scale** | Scale must be `kUNIFORM` or `kCHANNEL`; disabled if non-constant power parameter |
| **Conv + Pooling** | Both layers must share the same precision |
| **Conv + Generic Activation** | Any single-input, single-output pointwise layer qualifies (post pointwise fusion) |
| **Padding + Conv/Deconv** | All padding sizes must be non-negative |
| **Depthwise Separable Conv** | Depthwise conv + activation + regular conv + activation; both convolutions must be INT8 (compute capability >= 7.2) |
| **Scale + Activation** | Fused into a single activation layer |
| **Shuffle + Shuffle** | Two consecutive shuffles collapse into one or nothing (restrictions on reshape operations) |
| **Scale Identity Elimination** | Scales adding 0, multiplying by 1, or computing power of 1 are erased |
| **Softmax + Log** | Combined into single LogSoftmax layer |
| **Softmax + TopK** | Combined into single layer |
| **Shuffle + Reduce** | When shuffle performs only permutation (no reshape) and reduce has keepDimensions set |

### 1.2 Reduction Operation Fusions

- **GELU**: Two mathematical representations are recognized and fused into a single operation
- **L1Norm**: `kABS` followed by `kSUM` reduction
- **L2Norm**: Sum of squares followed by `kSQRT`
- **LogSum / LogSumExp**: Composed reduction sequences are detected and collapsed

### 1.3 Pointwise Fusion

Multiple adjacent pointwise layers (Activation, Constant with single value, ElementWise, Scale with `kUNIFORM`, Unary) are fused into a single pointwise kernel. The fused kernel size is not unlimited -- some layers may not be fused.

Eligible layers: Activation, Constant (single value), ElementWise, Scale (`kUNIFORM`), Unary.

Fused layers receive combined names: e.g., `fusedPointwiseNode(add1, relu1)`.

### 1.4 Inspecting Fusions

- Builder logs fusion decisions at `kINFO` level
- Verify fusions via engine inspector APIs or verbose logging during builds
- Use `--profilingVerbosity=detailed` with trtexec to see fusion details

### 1.5 Design Implications for Maximum Fusion

- Prefer standard activation functions (ReLU, GELU, Clip) after convolutions
- Avoid broadcasting in element-wise additions (except batch dimension)
- Ensure consistent precision across fusible layer pairs
- Use pointwise operations adjacently to enable automatic fusion
- Group operations that TensorRT can fuse rather than inserting non-fusible layers between them

---

## 2. Precision Conversion Best Practices

### 2.1 FP16 Mode

Enable with `--fp16` flag in trtexec or via builder config. TensorRT selects FP16 tactics when they offer better performance.

**Key considerations:**
- Models may need retraining to ensure intermediate layer output can be represented without FP16 overflow/underflow
- FP16 dynamic range: approximately 5.96e-8 to 65504
- Tensor Core alignment: 8 elements for FP16 dense, 16 for FP16 sparse

### 2.2 Strongly Typed Networks

The `--stronglyTyped` flag forces TensorRT to strictly follow the data types in the model, including all quantized operations. This is preferred over `--fp16` when using Q/DQ quantized models.

### 2.3 Mixed Precision Strategy

- Use FP32 for layers sensitive to precision (e.g., loss functions, final outputs)
- Use FP16 for compute-heavy layers (convolutions, matrix multiplications)
- Use INT8/FP8 for bandwidth-bound operations and where calibration shows acceptable accuracy

### 2.4 Per-Layer Precision Control

- Force specific layers to higher precision for accuracy-sensitive operations
- Use Polygraphy to dump layer outputs and identify precision-sensitive layers
- Experimental debug precision tool can automatically find layers requiring high precision

### 2.5 Accuracy Debugging

- Use Polygraphy to dump layer outputs and check for NaNs/Infs with `--validate`
- Compare FP32 vs FP16/INT8 outputs layer by layer
- Identify accumulation points where quantization error compounds

---

## 3. Quantization Deep Dive

### 3.1 Supported Quantized Types

| Type | Bits | Representation | Range |
|------|------|---------------|-------|
| **INT8** | 8 | Signed integer | [-128, 127] |
| **FP8 (E4M3)** | 8 | Float (4 exp, 3 mantissa) | [-448, 448] |
| **INT4** | 4 | Signed integer | [-8, 7] |
| **FP4 (E2M1)** | 4 | Float (2 exp, 1 mantissa) | [-6, 6] |

### 3.2 Quantization Workflows

#### Post-Training Quantization (PTQ)

Quantizes a pre-trained model without retraining. Requires a representative calibration dataset.

```bash
pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-modelopt
python3 -m modelopt.onnx.quantization \
    --onnx_path model.onnx \
    --quantize_mode int8 \
    --output_path model_quantized.onnx
```

Use `--stronglyTyped` instead of `--fp16` to enforce strict adherence to quantized types.

**Calibration guidelines:**
- About 500 images are sufficient for ImageNet classification networks
- Large single batches are preferred over small batches (small batches cause reduced histogram resolution)
- Calibration is deterministic -- same inputs in same order on same device produce identical scales

#### Quantization-Aware Training (QAT)

Simulates quantization during training by quantizing weights and activation layers. Training compensates for Q/DQ effects, generally achieving better accuracy than PTQ. Use the NVIDIA TensorRT Model Optimizer toolkit.

### 3.3 Explicit vs. Implicit Quantization

**Implicit quantization is deprecated.** Always use explicit quantization with Q/DQ nodes.

- **Explicit**: Uses `IQuantizeLayer` and `IDequantizeLayer` to precisely control type conversions
- Supports INT8, FP8, INT4, and FP4
- Q/DQ nodes embed scales directly -- no external calibration tables needed

### 3.4 Quantization Granularities

| Granularity | Description | Use Cases |
|---|---|---|
| **Per-Tensor** | Single scalar scale for entire tensor | Activations |
| **Per-Channel** | Scale vector along output-channel axis | Weights (convolutions) |
| **Block Quantization** | Fixed-size blocks, each with own scale | INT4, FP4, MXFP8 |

Block quantization specifics:

| Type | Block Sizes | Weights | Activations |
|------|-------------|---------|-------------|
| INT4 | {64, 128} | Yes (Weight-Only Quantization) | No |
| FP4 (NVFP4) | 16 | Yes | Yes (dynamic) |
| MXFP8 | 32 | Yes | Yes (dynamic) |

### 3.5 Quantization Formulas

**INT8:**
$$x_q = \text{roundTiesToEven}(\text{clip}(x/s, -128, 127))$$
$$x_{dq} = x_q \cdot s$$

**FP8:**
$$x_q = \text{castToFp8}(\text{clip}(x/s, -448, 448))$$

**INT4:**
$$x_q = \text{roundTiesToEven}(\text{clip}(x/s, -8, 7))$$

### 3.6 Dynamic Quantization

Scales are computed at inference time per block. Benefits:
- Improved accuracy from narrower dynamic ranges
- Reduced PTQ overhead (no offline calibration needed)

Per-block scale computation:
$$scale = \max_{i \in \{0..\text{blockSize}-1\}} \left(\frac{|x_i|}{qTypeMax}\right)$$

### 3.7 Weight-Only Quantization (WoQ)

Available only for INT4 block quantization with GEMM layers:
- Weights stored at INT4 precision
- GEMM input and compute remain high precision (FP32/FP16/BF16)
- TensorRT reads low-precision weights and dequantizes before GEMM
- 4-bit weights packed two elements per byte (first in 4 LSBs, second in 4 MSBs)

### 3.8 Q/DQ Layer-Placement Recommendations

1. **Quantize all inputs of weighted operations** (Conv, Transposed Conv, GEMM) -- reduces bandwidth and enables INT8/FP8 compute
2. **Do not quantize outputs of weighted operations by default** -- preserve higher precision for activation functions
3. **Do not simulate batch normalization and ReLU fusions** in training -- TensorRT handles these optimizations
4. **Quantize residual inputs in skip connections** -- enables element-wise addition fusion with weighted layers
5. **Try quantizing non-commuting layers** -- but be conservative; unfused Q/DQ nodes hurt performance
6. **Use per-tensor quantization for activations, per-channel for weights**

### 3.9 Q/DQ Propagation

TensorRT propagation strategy:
- Q nodes propagate **backward** (quantize as early as possible)
- DQ nodes propagate **forward** (dequantize as late as possible)
- This maximizes low-precision graph coverage

A layer commutes with quantization if Q(Op(x)) = Op(Q(x)). Example: Max Pooling commutes with both Q and DQ.

### 3.10 Calibrators for PTQ (Implicit, Deprecated)

| Calibrator | Best For | Notes |
|-----------|----------|-------|
| `IInt8EntropyCalibrator2` | CNN-based networks, DLA | Recommended; calibrates before fusion |
| `IInt8MinMaxCalibrator` | NLP tasks | Uses full activation range; for BERT-like models |
| `IInt8EntropyCalibrator` | General | Original; calibrates after fusion |
| `IInt8LegacyCalibrator` | Fallback | Compatible with TensorRT 2.0 EA |

Calibration cache portability: Portable across devices with `IInt8EntropyCalibrator2` or `IInt8MinMaxCalibrator`. Not portable across TensorRT releases.

### 3.11 ONNX Quantization Support

| Feature | Required ONNX Opset |
|---|---|
| Per-channel quantization | 13+ |
| FP8 (E4M3FN) | 19+ |
| INT4 and block quantization | 21+ |
| FP4E2M1 | 23+ |

Not supported: `QLinearConv`, `QLinearMatmul`, `ConvInteger`, `MatmulInteger`.

---

## 4. Memory Management Guidelines

### 4.1 Device Memory Tracking

Create a custom GPU allocator implementing `IGpuAllocator` that wraps `cudaMalloc`/`cudaFree` to track allocations with timestamps. Set this allocator on `IBuilder` and `IRuntime`.

### 4.2 Shared Execution Context Memory

Use `createExecutionContextWithoutDeviceMemory()` to share activation memory across execution contexts. This is especially useful with CUDA graphs where memory addresses are captured as part of the graph.

### 4.3 Pinned Host Memory

Allocate pinned host memory for input and output data:
```cpp
cudaHostAlloc(&hostPtr, size, cudaHostAllocDefault);
// or
cudaMallocHost(&hostPtr, size);
```

This avoids interference from pageable memory during H2D/D2H transfers and enables DMA-based asynchronous transfers.

### 4.4 Memory Optimization Strategies

- **Reduce engine size**: Use lower precision (FP16/INT8 weights take less space)
- **Share activation memory**: Multiple execution contexts can share activation buffers when not running concurrently
- **Use weight streaming**: For large models that exceed GPU memory, stream weights from host memory to GPU on demand
- **Minimize optimization profiles**: Each profile may increase memory footprint
- **Avoid unnecessary auxiliary streams**: Auxiliary streams can increase memory consumption because activation buffers can no longer be reused

### 4.5 Embedded/Mobile Considerations

On platforms with shared GPU/CPU memory, H2D/D2H copies are unnecessary if host memory is allocated with CUDA APIs and pinned. For memory-constrained devices (e.g., Jetson Nano), increase system swap:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon -a
```

---

## 5. CUDA Graph Integration

### 5.1 Basic Capture Pattern

```cpp
// Step 1: Flush deferred updates
context->enqueueV3(stream);

// Step 2: Capture the graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
context->enqueueV3(stream);
cudaStreamEndCapture(stream, &graph);

// Step 3: Instantiate
cudaGraphInstantiate(&instance, graph, 0);

// Step 4: Launch (repeat for inference)
cudaGraphLaunch(instance, stream);
cudaStreamSynchronize(stream);
```

### 5.2 Key Requirements

- **Pre-capture flush**: After changing input shapes or shape tensor values, call `enqueueV3()` once to flush deferred updates before capturing
- **One context per graph**: Best practice is to use one execution context per captured graph
- **Fixed state**: Input/output buffer locations and activation memory addresses are baked into the graph
- **Re-capture on shape change**: After shape changes, must re-flush and re-capture

### 5.3 Limitations

CUDA graphs cannot handle:
- Loops and conditionals
- Layers requiring data-dependent shapes
- Synchronous APIs (cudaMemcpy)
- Legacy default CUDA stream

Failed captures return `cudaErrorStreamCapture*` errors but the context remains usable for normal inference.

### 5.4 Performance Impact

CUDA graphs are especially valuable for enqueue-bound workloads where `enqueueV3()` time exceeds actual GPU execution time. They eliminate per-launch overhead by replaying a captured sequence of operations.

### 5.5 Concurrent Activity During Capture

- Avoid legacy default CUDA stream
- Use `cudaStreamNonBlocking` flag for streams
- Use asynchronous APIs (e.g., `cudaMemcpyAsync()` instead of `cudaMemcpy()`)

### 5.6 Profiling CUDA Graphs

Add `--cuda-graph-trace=node` to `nsys` to see per-kernel runtime info within CUDA graphs.

---

## 6. Multi-Stream Execution

### 6.1 Within-Inference Multi-Streaming

TensorRT can run layers in parallel across multiple auxiliary streams:

```cpp
config->setMaxAuxStreams(7); // Up to 7 auxiliary + 1 mainstream
```

```python
config.max_aux_streams = 7
```

TensorRT automatically synchronizes auxiliary streams with the mainstream at the beginning and end of each `enqueueV3()` call.

Custom auxiliary streams can be provided:
```cpp
context->setAuxStreams(streams.data(), nbAuxStreams);
```

**Trade-off**: Enabling auxiliary streams can increase memory consumption because some activation buffers can no longer be reused.

### 6.2 Cross-Inference Multi-Streaming

For concurrent inference on multiple inputs:
1. Build engines with multiple optimization profiles
2. Create one execution context per profile
3. Call `enqueueV3()` on different CUDA streams for parallel execution
4. Use `setOptimizationProfileAsync()` to assign profiles

### 6.3 Thread Pool Pattern

A common pattern uses worker threads each owning an execution context and CUDA stream:
- Each thread processes incoming requests independently
- Each thread synchronizes with its own stream without blocking other workers
- Consider limiting compute resources during engine build to match actual runtime resource availability

---

## 7. Batch Processing Optimization

### 7.1 General Principles

Batching is the single most important optimization for GPU throughput:
- Amortizes per-layer overhead and synchronization costs
- Transforms vector-matrix operations into more efficient matrix-matrix operations
- Provides better GPU utilization through larger parallel workloads

### 7.2 Batch Size Guidelines

| Scenario | Recommendation |
|---|---|
| **General** | Larger batch sizes are almost always more efficient on GPU |
| **Very large batches** | Avoid N > 2^16 due to extended index computation |
| **FP16/INT8 with Tensor Cores** | Multiples of 32 tend to have best performance |
| **Ada Lovelace+ GPUs** | Smaller batches may improve throughput when they enable L2 cache utilization |

Always experiment with various batch sizes to find the optimal point for your specific model and hardware.

### 7.3 Opportunistic Batching

For request-based applications: for each incoming request, wait for a time T. If other requests arrive, batch them together. NVIDIA Triton Inference Server provides built-in dynamic batching support.

### 7.4 Specifying Batch Dimensions

Batch dimension is part of tensor dimensions in TensorRT. Use optimization profiles to specify batch size ranges:
```bash
trtexec --shapes=data:4x3x224x224  # batch size 4
```

---

## 8. Kernel Tuning Techniques

### 8.1 Tactic Selection

TensorRT profiles all available tactics (kernel implementations) per layer and selects the fastest. Since selection depends on latency measurements, different builds can select different tactics when latencies are similar.

### 8.2 Deterministic Tactic Selection

Three approaches for reproducibility:

1. **Lock GPU Clock Frequency**:
   ```bash
   sudo nvidia-smi -lgc <freq>
   ```
   Eliminates clock variation during tactic timing.

2. **Increase Average Timing Iterations**:
   ```cpp
   builderConfig->setAvgTimingIterations(8);  // Default is 4
   ```
   More iterations improve determinism but extend build time.

3. **Timing Cache**: Reuse the same timing cache across builds with identical `INetworkDefinition` and builder config.

### 8.3 Timing Cache

- Caches per-layer tactic latencies
- Specific to targeted device, CUDA version, TensorRT version, and `BuilderConfig`
- Reusing caches reduces profiling time and improves tactic consistency
- Supports "editable" mode for accuracy debugging: dump available tactics, identify problematic ones, update with better ones

### 8.4 Limiting Compute Resources (MPS)

When engines will run with shared GPU resources:
```bash
nvidia-cuda-mps-control -d
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
# Build engine here
echo quit | nvidia-cuda-mps-control
```

This optimizes tactic selection for reduced resource availability and generally promotes throughput at the expense of latency.

---

## 9. Plugin Development

### 9.1 Performance Fundamentals

- Start with standalone CUDA applications for correctness verification, then add performance measurement
- Follow standard CUDA best practices (coalesced memory access, occupancy optimization, shared memory usage)
- Support as many data formats as possible in the plugin to eliminate internal reformat operations

### 9.2 Format Support Strategy

Supporting multiple I/O formats prevents TensorRT from inserting costly reformat layers:
- Support at minimum: FP32 (kLINEAR), FP16 (kLINEAR and kHWC8)
- For INT8 quantized networks: support INT8 formats as well
- The more formats supported, the more fusion opportunities available

### 9.3 Plugin Registration

All TensorRT plugins auto-register when the plugin library loads. Custom plugins use the plugin creator registry for serialization/deserialization.

### 9.4 Q/DQ Interaction with Plugins

When plugins consume/produce quantized data:
- Input DQ and output Q nodes must be incorporated into the plugin
- Remove those Q/DQ nodes from the network
- Plugin receives quantization scales for internal handling
- Call `setOutputType(kINT8)` on the plugin layer

### 9.5 Best Practices for Custom Plugins

- Minimize host-device synchronization within plugin execution
- Avoid dynamic memory allocations during `enqueue()`
- Pre-allocate workspace through `getWorkspaceSize()`
- Handle multiple batch sizes efficiently
- Test both standalone correctness and within-engine behavior
- Profile using Nsight Compute for kernel-level optimization

---

## 10. Dynamic Shapes and Optimization Profiles

### 10.1 Dynamic Shape Workflow

1. Mark dynamic dimensions with `-1` at build time
2. Define optimization profiles with min/opt/max dimension ranges
3. At runtime: create context, select profile, set input dimensions, enqueue
4. Re-set dimensions only when input size changes

```cpp
// Build time
networkDefinition.addInput("input", DataType::kFLOAT, Dims4(-1, 3, -1, -1))

// Optimization profile
profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, 224, 224));
profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(4, 3, 224, 224));
profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(16, 3, 224, 224));

// Runtime
context->setInputShape("input", Dims4(8, 3, 224, 224));
```

### 10.2 Optimization Profile Best Practices

- **Set opt dimensions to the most common input size** -- TensorRT optimizes most aggressively for opt dimensions
- **Minimize the gap between min and max** -- some tactics only work when min=opt=max; wider ranges disable those tactics
- **Use multiple profiles** for distinct input size regimes rather than one wide-range profile
- **Each execution context must use a separate profile**
- Profile 0 is implicitly chosen for the first execution context

### 10.3 Shape Change Overhead

After shape/profile changes:
- TensorRT recomputes tensor shapes and resources
- First `enqueueV3()` after a change can be longer than subsequent calls
- CUDA graphs must be re-flushed and re-captured

### 10.4 Named Dimensions

Dimensions can be named for:
- Better error messages
- Implicit equality constraints between tensors sharing a dimension name
- ONNX parser automatically sets dimension names from the ONNX file

### 10.5 Dynamically Shaped Output

Two approaches:

1. **Computable from input dimensions**: Use `getTensorShape()` after providing input dimensions
2. **Data-dependent shapes**: Use `IOutputAllocator` with strategies:
   - Defer allocation until size is known
   - Preallocate based on `getMaxOutputSize` upper bound
   - Preallocate based on experience; return nullptr if insufficient
   - Grow buffer on demand
   - Defer initially, then recycle/grow

### 10.6 Execution Tensors vs. Shape Tensors

TensorRT uses a ping-pong execution strategy:
1. Compute tensor shapes on CPU until GPU-dependent shape is encountered
2. Stream work to GPU until unknown shape is reached, then synchronize and return to step 1

Shape tensors: must be Int32, Int64, Float, or Bool; shape determinable at build time; 64 elements max.

When TensorRT needs a shape tensor classified as an execution tensor, it copies from GPU to CPU, incurring synchronization overhead.

### 10.7 INT8 Calibration with Dynamic Shapes

- Set a calibration optimization profile via `config->setCalibrationProfile(profile)`
- Calibration runs using the profile's kOPT values
- Input data size must match opt dimensions
- `getBatchSize()` must return 1

### 10.8 Restrictions

- Conv/Deconv channel dimension must be a build-time constant
- INT8 channel dimension must be a build-time constant
- Tensor rank must be known at build time
- Only dimensions (not rank) can be dynamic

---

## 11. Tensor Core Optimization

### 11.1 Alignment Requirements

| Operation Type | Suggested Alignment (Elements) |
|---|---|
| TF32 | 4 |
| FP16 (dense) | 8 |
| FP16 (sparse) | 16 |
| INT8 | 32 |

Alignment applies to I/O channel dimensions (Conv/Deconv) and matrix dimensions K and N (MatrixMultiply: M x K times K x N).

When requirements are not met, TensorRT implicitly pads tensors to the nearest multiple. Rounding up dimensions in the model definition avoids this overhead.

### 11.2 Monitoring Tensor Core Usage

Run Nsight Systems with `--gpu-metrics-device all` and check **SM instructions/Tensor Active** row. 100% utilization is not achievable due to DRAM reads/writes, instruction stalls, and other compute activity.

---

## 12. Benchmarking and Profiling

### 12.1 trtexec Flags for Stable Measurements

```bash
trtexec --onnx=model.onnx --shapes=input:BxCxHxW --fp16 \
    --noDataTransfers --useCudaGraph --useSpinWait
```

| Flag | Purpose |
|------|---------|
| `--noDataTransfers` | Remove H2D/D2H copy noise |
| `--useCudaGraph` | Reduce enqueue overhead |
| `--useSpinWait` | More stable latency via spin-wait synchronization |

### 12.2 Per-Layer Profiling

```bash
trtexec --onnx=model.onnx --profilingVerbosity=detailed \
    --dumpLayerInfo --dumpProfile --separateProfileRun
```

### 12.3 Nsight Systems Integration

Two-step workflow to avoid profiling the build phase:

```bash
# Step 1: Build the engine
trtexec --onnx=model.onnx --profilingVerbosity=detailed --saveEngine=model.plan

# Step 2: Profile inference only
nsys profile -o profile --capture-range cudaProfilerApi \
    trtexec --loadEngine=model.plan --warmUp=0 --duration=0 --iterations=50
```

### 12.4 Synchronization Modes

| Mode | CPU Behavior | Measurement Stability |
|------|-------------|----------------------|
| **BlockingSync** (default) | Yields to other threads | Lower CPU usage, less stable |
| **SpinWait** (`--useSpinWait`) | Polls continuously | Higher CPU usage, more stable |

---

## 13. Hardware and Environment Configuration

### 13.1 Clock Management

| Mode | Behavior | Best For |
|------|----------|----------|
| **Float (default)** | Idle at low frequency, boosts under load | Average performance |
| **Locked** | `sudo nvidia-smi -lgc <freq>` | Deterministic benchmarking |

### 13.2 Thermal and Power Management

- Power throttling occurs at the power limit; monitor via `nvidia-smi dmon -s pcu`
- Thermal throttling begins around 85C
- Power consumption can depend on activation values -- always benchmark with representative data
- Ensure proper cooling, especially for passively cooled GPUs

### 13.3 PCIe and Data Transfer

- Check PCIe generation and lane width configuration
- Use pinned memory for all host-device transfers
- Overlap transfers with compute using separate CUDA streams
- Consider GPUDirect for high-throughput scenarios
- Check NUMA topology on AMD x86_64 systems -- cross-node PCIe bandwidth is significantly reduced

### 13.4 Windows: TCC vs WDDM

TCC mode is recommended for inference GPUs. WDDM mode tends to cause worse and unstable performance results.

---

## Per-Layer Optimization Quick Reference

| Layer | Best Practice |
|---|---|
| **Gather** | Use axis 0 for maximum performance. No fusions available. |
| **Reduce** | Perform reduction across last dimensions (tail reduce) for sequential memory access. |
| **RNN/Loops** | Use `ILoopLayer` API for loop fusion, unrolling, loop-invariant code motion. Avoid MatrixMultiply with recurrent data dependence along sequence dimension. |
| **Shuffle** | Identity-equivalent shuffles are automatically omitted. |
| **TopK** | Use small K values and reduce along the last dimension for optimal sequential memory access. |

---

## Key Takeaways for Engine Implementation

1. **Maximize layer fusion** by designing model architectures that align with TensorRT's fusion rules
2. **Use explicit quantization** with Q/DQ nodes -- implicit quantization is deprecated
3. **Prefer per-tensor quantization for activations, per-channel for weights** for optimal accuracy
4. **Use CUDA graphs** for enqueue-bound workloads to eliminate per-launch overhead
5. **Allocate pinned host memory** for all input/output data transfers
6. **Set optimization profile opt dimensions** to the most common input size
7. **Lock GPU clocks** during benchmarking for deterministic results
8. **Align tensor dimensions** to Tensor Core requirements (8 for FP16, 32 for INT8)
9. **Use timing cache** across builds for consistency and faster build times
10. **Profile with Nsight Systems** using a two-step (build then profile) workflow
11. **Consider weight-only INT4 quantization** for large language models to reduce memory footprint
12. **Use auxiliary streams** for within-inference parallelism but be aware of the memory trade-off
13. **Support multiple formats in custom plugins** to avoid costly reformat operations
14. **Use dynamic quantization** (FP4/MXFP8) for models where static calibration is impractical
