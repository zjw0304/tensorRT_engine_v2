# TRT Engine Benchmark Guide

## 1. How to Run Benchmarks

### 1.1 Build Benchmarks

```bash
mkdir -p build && cd build
cmake .. -DTRT_ENGINE_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 1.2 Prepare a Model

Before benchmarking, build a TensorRT engine from an ONNX model:

```cpp
#include <trt_engine/trt_engine.h>

auto& logger = trt_engine::get_logger();
trt_engine::EngineBuilder builder(logger);

trt_engine::BuilderConfig config;
config.precision = trt_engine::Precision::FP16;
config.timing_cache_path = "timing.cache";

auto data = builder.build_engine("resnet50.onnx", config);
trt_engine::EngineBuilder::save_engine(data, "resnet50_fp16.engine");
```

Or use `trtexec`:

```bash
trtexec --onnx=resnet50.onnx --saveEngine=resnet50_fp16.engine --fp16
```

### 1.3 Run a Basic Benchmark

```cpp
#include <trt_engine/trt_engine.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>

int main() {
    auto engine = trt_engine::InferenceEngine::create("resnet50_fp16.engine");
    engine->warmup(20);

    auto inputs_info = engine->get_input_info();
    size_t input_size = 1;
    for (int d : inputs_info[0].shape) {
        input_size *= (d > 0) ? d : 1;
    }

    std::vector<std::vector<float>> input_data = {
        std::vector<float>(input_size, 0.5f)
    };

    const int iterations = 1000;
    std::vector<float> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto result = engine->infer(input_data);
        if (result.success) {
            latencies.push_back(result.latency_ms);
        }
    }

    std::sort(latencies.begin(), latencies.end());

    float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    float mean = sum / latencies.size();
    float p50 = latencies[latencies.size() / 2];
    float p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    float p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "Iterations: " << latencies.size() << std::endl;
    std::cout << "Mean latency:  " << mean << " ms" << std::endl;
    std::cout << "P50 latency:   " << p50 << " ms" << std::endl;
    std::cout << "P95 latency:   " << p95 << " ms" << std::endl;
    std::cout << "P99 latency:   " << p99 << " ms" << std::endl;
    std::cout << "Throughput:    " << 1000.0f / mean << " infer/sec" << std::endl;

    return 0;
}
```

### 1.4 Using trtexec for Baseline Comparison

```bash
# FP32 baseline
trtexec --onnx=resnet50.onnx --iterations=1000 --warmUp=5000

# FP16
trtexec --onnx=resnet50.onnx --fp16 --iterations=1000 --warmUp=5000

# INT8 with calibration
trtexec --onnx=resnet50.onnx --int8 --iterations=1000 --warmUp=5000

# With CUDA graphs
trtexec --onnx=resnet50.onnx --fp16 --useCudaGraph --iterations=1000

# Stable measurements
trtexec --onnx=resnet50.onnx --fp16 --noDataTransfers --useCudaGraph --useSpinWait
```

---

## 2. Expected Performance Characteristics

### 2.1 ResNet-50 (Batch Size 1)

| GPU | FP32 | FP16 | INT8 |
|-----|------|------|------|
| V100 | ~3.5 ms | ~1.5 ms | ~0.9 ms |
| T4 | ~5.0 ms | ~1.8 ms | ~1.0 ms |
| A100 | ~2.0 ms | ~0.8 ms | ~0.5 ms |
| RTX 3090 | ~2.5 ms | ~1.0 ms | ~0.6 ms |
| L40 | ~2.0 ms | ~0.7 ms | ~0.4 ms |
| H100 | ~1.5 ms | ~0.5 ms | ~0.3 ms |

*Values are approximate GPU-only execution times (excluding H2D/D2H transfers).*

### 2.2 ResNet-50 (Batch Size 32, FP16)

| GPU | Latency (ms) | Throughput (images/sec) |
|-----|-------------|------------------------|
| V100 | ~8 ms | ~4,000 |
| T4 | ~12 ms | ~2,700 |
| A100 | ~4 ms | ~8,000 |
| RTX 3090 | ~5 ms | ~6,400 |
| H100 | ~3 ms | ~10,000+ |

### 2.3 CUDA Graph Impact

CUDA graphs are most beneficial for small, enqueue-bound models:

| Model Size | Without Graphs | With Graphs | Improvement |
|-----------|---------------|-------------|-------------|
| Small (< 10 layers) | ~0.5 ms | ~0.1 ms | ~5x |
| Medium (50-100 layers) | ~2 ms | ~1.8 ms | ~10% |
| Large (200+ layers) | ~10 ms | ~9.8 ms | ~2% |

### 2.4 Multi-Stream Throughput

Using `MultiStreamEngine` with N streams (ResNet-50, FP16, A100):

| Streams | Throughput (infer/sec) | Avg Latency |
|---------|----------------------|-------------|
| 1 | ~1,250 | ~0.8 ms |
| 2 | ~2,300 | ~0.9 ms |
| 4 | ~4,000 | ~1.0 ms |
| 8 | ~5,500 | ~1.4 ms |

Throughput scales sub-linearly due to GPU resource contention and memory bandwidth limits.

### 2.5 Dynamic Batching Throughput

Using `DynamicBatcher` (ResNet-50, FP16, A100, max_wait=5ms):

| Max Batch Size | Individual Throughput | Batched Throughput | Improvement |
|---------------|----------------------|-------------------|-------------|
| 1 | ~1,250/sec | ~1,250/sec | 1x |
| 8 | ~1,250/sec | ~5,000/sec | ~4x |
| 32 | ~1,250/sec | ~8,000/sec | ~6x |

---

## 3. Performance Tuning Tips

### 3.1 Reduce Latency

1. **Use FP16 or INT8 precision** for lower compute time.
2. **Enable CUDA graphs** for small models with high enqueue overhead.
3. **Pre-warm the engine** to avoid JIT compilation on first inference.
4. **Use pinned memory** for input/output to overlap transfers.
5. **Lock GPU clocks** to avoid frequency scaling during inference.
6. **Minimize dynamic shape changes** as they trigger re-optimization.

### 3.2 Maximize Throughput

1. **Increase batch size** -- larger batches utilize GPU parallelism better.
2. **Use MultiStreamEngine** to process multiple requests concurrently.
3. **Use DynamicBatcher** to automatically batch individual requests.
4. **Scale to multiple GPUs** with MultiGPUEngine for linear scaling.
5. **Align batch sizes to Tensor Core requirements**: multiples of 8 for FP16, 32 for INT8.

### 3.3 Reduce Memory Usage

1. **Use lower precision** (FP16 weights are 2x smaller than FP32).
2. **Reduce context_pool_size** if concurrency requirements are low.
3. **Use narrow optimization profiles** (min close to max).
4. **Limit max_aux_streams** as auxiliary streams increase memory consumption.

### 3.4 Consistent Benchmarking

1. **Lock GPU clocks**: `sudo nvidia-smi -lgc 1410`
2. **Enable persistence mode**: `sudo nvidia-smi -pm 1`
3. **Run warmup iterations** (at least 20).
4. **Use at least 1000 measurement iterations**.
5. **Use CUDA events** for GPU-only timing.
6. **Report P50, P95, P99 latencies**, not just mean.
7. **Exclude the first few iterations** from measurements.
8. **Use `--noDataTransfers --useSpinWait`** flags with trtexec.

---

## 4. Interpreting Results

### 4.1 Latency Metrics

- **P50 (median)**: Typical latency experienced by most requests. Use for SLA definition.
- **P95**: Tail latency that captures 95% of requests. Important for user experience.
- **P99**: Worst-case tail latency. Important for strict SLA requirements.
- **Mean**: Average latency. Can be skewed by outliers.

### 4.2 Throughput Metrics

- **Inferences per second**: Total completed inferences divided by elapsed time.
- **Images per second**: For batch inference, inferences/sec multiplied by batch size.
- **GPU utilization**: Check via `nvidia-smi` -- sustained >90% indicates good utilization.

### 4.3 Common Patterns

- **Latency increases linearly with batch size**: Expected. Use throughput as the primary metric for batched workloads.
- **First inference is slow**: Normal. TensorRT performs JIT optimizations on first run. Always warmup.
- **Latency varies significantly**: Lock GPU clocks and use spin-wait synchronization.
- **Throughput plateaus at high stream counts**: GPU is saturated. Adding more streams won't help.
- **INT8 not faster than FP16**: May occur on small models where kernel launch overhead dominates. Try CUDA graphs.

### 4.4 Profiling with Nsight Systems

For deeper analysis, profile with Nsight Systems:

```bash
# Build engine first
trtexec --onnx=model.onnx --fp16 --saveEngine=model.engine

# Profile inference only
nsys profile -o profile_report \
    --capture-range cudaProfilerApi \
    trtexec --loadEngine=model.engine \
    --warmUp=0 --duration=0 --iterations=50

# View report
nsys-ui profile_report.nsys-rep
```

Look for:
- **Kernel execution gaps**: Indicates CPU bottleneck or synchronization issues.
- **Memory transfer overlap**: Verify H2D/D2H transfers overlap with compute.
- **Tensor Core utilization**: Check SM Tensor Active percentage.

### 4.5 Profiling CUDA Graphs

```bash
nsys profile --cuda-graph-trace=node -o graph_report ./your_application
```

This shows per-kernel runtime within CUDA graph nodes.
