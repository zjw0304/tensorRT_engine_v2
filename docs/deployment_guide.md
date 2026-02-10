# TRT Engine Deployment Guide

## 1. System Requirements

### Minimum Requirements

| Component | Minimum Version |
|-----------|----------------|
| CUDA Toolkit | >= 11.8 |
| TensorRT | >= 8.6 |
| cuDNN | >= 8.6 |
| GPU Compute Capability | >= 7.0 (Volta+) |
| C++ Compiler | GCC >= 9 or Clang >= 10 (C++17 support) |
| CMake | >= 3.18 |
| Linux Kernel | >= 5.4 |

### Recommended

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.x |
| TensorRT | 10.x |
| GPU | A100, L40, H100 (Ampere/Ada Lovelace/Hopper) |
| OS | Ubuntu 22.04 LTS |

### Supported GPUs

| GPU | Architecture | Compute Capability |
|-----|-------------|-------------------|
| V100 | Volta | 7.0 |
| T4 | Turing | 7.5 |
| A100 | Ampere | 8.0 |
| A10/A30 | Ampere | 8.6 |
| RTX 3090 | Ampere | 8.6 |
| L40/RTX 4090 | Ada Lovelace | 8.9 |
| H100 | Hopper | 9.0 |

### Supported Platforms

- Ubuntu 20.04, 22.04, 24.04 (x86_64)
- CentOS 7, Rocky Linux 8/9 (x86_64)
- Jetson (aarch64) with JetPack 5.x+
- Windows (experimental, MSVC 2019+)

---

## 2. Build from Source

### 2.1 Prerequisites

Install CUDA Toolkit:
```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4
```

Install TensorRT:
```bash
sudo apt-get install libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev
```

### 2.2 Build Steps

```bash
# Clone the repository
git clone <repository_url> trt_engine
cd trt_engine

# Create build directory
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### 2.3 Build Options

```bash
# Enable Python bindings
cmake .. -DTRT_ENGINE_BUILD_PYTHON=ON

# Enable tests
cmake .. -DTRT_ENGINE_BUILD_TESTS=ON

# Enable benchmarks
cmake .. -DTRT_ENGINE_BUILD_BENCHMARKS=ON

# Specify TensorRT path manually
cmake .. -DTensorRT_ROOT=/usr/local/tensorrt

# Build for a specific GPU architecture only (faster build)
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
```

### 2.4 Verify Build

```bash
# Check the shared library
ls -la build/libtrt_engine.so

# Run with a test model (if available)
./build/examples/basic_inference model.onnx
```

---

## 3. Docker Setup

### 3.1 Dockerfile

```dockerfile
FROM nvcr.io/nvidia/tensorrt:24.08-py3

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
WORKDIR /workspace/trt_engine
COPY . .

# Build
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_ENGINE_BUILD_PYTHON=ON && \
    make -j$(nproc) && \
    make install

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Default command
CMD ["bash"]
```

### 3.2 Build and Run Docker Container

```bash
# Build the Docker image
docker build -t trt_engine:latest .

# Run with GPU access
docker run --gpus all --rm -it \
    -v /path/to/models:/models \
    trt_engine:latest

# Run with specific GPU
docker run --gpus '"device=0"' --rm -it \
    trt_engine:latest
```

### 3.3 Docker Compose

```yaml
version: '3.8'
services:
  trt_engine:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./models:/models
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## 4. Python Installation

### 4.1 Build Python Bindings

```bash
cd build
cmake .. -DTRT_ENGINE_BUILD_PYTHON=ON
make -j$(nproc)

# The Python module will be built as a .so file
# Add to Python path
export PYTHONPATH=/path/to/build/python:$PYTHONPATH
```

### 4.2 Using pip (when available)

```bash
pip install .
```

### 4.3 Verify Python Installation

```python
import trt_engine
print("trt_engine version:", trt_engine.__version__)
```

---

## 5. GPU Configuration

### 5.1 Clock Management

For deterministic benchmarking, lock GPU clocks:

```bash
# List supported clock frequencies
nvidia-smi -q -d SUPPORTED_CLOCKS

# Lock to a specific frequency (requires root)
sudo nvidia-smi -lgc 1410

# Reset to default
sudo nvidia-smi -rgc
```

### 5.2 Multi-Process Service (MPS)

When multiple processes share a GPU, use MPS for better resource utilization:

```bash
# Start MPS daemon
sudo nvidia-cuda-mps-control -d

# Set thread percentage (optional)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

# Stop MPS
echo quit | sudo nvidia-cuda-mps-control
```

### 5.3 Multi-Instance GPU (MIG)

For A100/H100 GPUs, partition into isolated instances:

```bash
# Enable MIG mode
sudo nvidia-smi -i 0 -mig 1

# Create instances (7 x 1g.5gb on A100)
sudo nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C

# List instances
nvidia-smi mig -i 0 -lgi

# Disable MIG
sudo nvidia-smi -i 0 -mig 0
```

### 5.4 Persistence Mode

Keep the GPU driver loaded for faster initialization:

```bash
sudo nvidia-smi -pm 1
```

### 5.5 Power and Thermal

Monitor GPU state:

```bash
# Real-time monitoring
nvidia-smi dmon -s pcu

# Detailed query
nvidia-smi -q -d TEMPERATURE,POWER

# Set power limit
sudo nvidia-smi -pl 300  # watts
```

---

## 6. Performance Tuning Checklist

### Build-Time Optimizations

- [ ] Set `precision` to FP16 or INT8 for production workloads
- [ ] Use `timing_cache_path` to cache tactic selection across builds
- [ ] Set `max_workspace_size` to a generous value (1-4 GB)
- [ ] Configure `dynamic_shapes` with tight min/opt/max ranges
- [ ] Set `opt_dims` to your most common input shape
- [ ] Consider `max_aux_streams` (1-3) for within-inference parallelism
- [ ] Use `strongly_typed = true` when working with quantized models

### Runtime Optimizations

- [ ] Call `warmup(10)` before serving requests
- [ ] Set `context_pool_size` >= expected concurrent request count
- [ ] Enable CUDA graphs for repeated same-shape inference
- [ ] Use `MultiStreamEngine` for concurrent request processing
- [ ] Use `DynamicBatcher` for many small individual requests
- [ ] Use pinned memory for input/output data transfers
- [ ] Match `device_id` to the physically closest GPU

### System-Level Optimizations

- [ ] Lock GPU clock frequency for deterministic performance
- [ ] Enable persistence mode (`nvidia-smi -pm 1`)
- [ ] Use PCIe Gen4 or Gen5 interconnect
- [ ] Pin application threads to NUMA-local CPU cores
- [ ] Disable GPU ECC for marginal performance improvement (if data integrity allows)
- [ ] Set CPU governor to `performance` mode

### Memory Optimizations

- [ ] Use lower precision to reduce engine size and memory footprint
- [ ] Share execution contexts when not running concurrently
- [ ] Limit optimization profiles to reduce memory overhead
- [ ] Monitor memory with `MemoryManager` statistics

---

## 7. Multi-GPU Setup

### 7.1 Device Enumeration

```cpp
int count = trt_engine::get_device_count();
for (int i = 0; i < count; ++i) {
    auto props = trt_engine::get_device_properties(i);
    std::cout << "GPU " << i << ": " << props.name
              << " (SM " << props.compute_capability_major
              << "." << props.compute_capability_minor << ")" << std::endl;
}
```

### 7.2 Multi-GPU Inference

```cpp
// Use all available GPUs
std::vector<int> device_ids;
for (int i = 0; i < trt_engine::get_device_count(); ++i) {
    device_ids.push_back(i);
}

trt_engine::MultiGPUEngine engine("model.engine", device_ids);
auto result = engine.infer(inputs);  // Round-robin across GPUs
```

### 7.3 Considerations

- Engine files built on one GPU architecture may not work on a different architecture.
- Build engines separately for each GPU type if using heterogeneous configurations.
- NVLink/NVSwitch provides higher bandwidth for multi-GPU communication.
- For data-parallel workloads, the library handles device selection automatically.

---

## 8. Troubleshooting

### Common Issues

**CUDA_ERROR_OUT_OF_MEMORY**
- Reduce `max_workspace_size` in `BuilderConfig`.
- Reduce `context_pool_size` in `EngineConfig`.
- Use lower precision (FP16 instead of FP32).
- Use `nvidia-smi` to check for other processes using GPU memory.

**Engine build fails with "no implementation found"**
- Ensure your GPU compute capability matches the CUDA architecture flags.
- Try increasing `max_workspace_size`.
- Check TensorRT version compatibility with your ONNX opset.

**Engine deserialization fails**
- Engine files are version-specific. Rebuild if you updated TensorRT.
- Engine files are architecture-specific. Rebuild for each GPU type.
- Verify the engine file is not corrupted.

**ONNX parse errors**
- Run `python3 -m onnxruntime.tools.check_onnx_model model.onnx` to validate.
- Use Polygraphy to diagnose: `polygraphy inspect model model.onnx`
- Check ONNX opset version compatibility.

**Slow first inference after loading**
- This is expected. TensorRT may perform JIT compilation on first run.
- Call `warmup()` to ensure the first user-facing inference is fast.

**Inconsistent latency measurements**
- Lock GPU clock frequency with `nvidia-smi -lgc`.
- Enable persistence mode with `nvidia-smi -pm 1`.
- Use CUDA events for GPU-only timing (excludes CPU overhead).
- Run warmup iterations before measuring.

**Multi-GPU initialization failure**
- Verify all device IDs exist with `nvidia-smi`.
- Check CUDA peer access: `cudaDeviceCanAccessPeer`.
- Ensure each GPU has sufficient memory for the engine.

### Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Check TensorRT version
dpkg -l | grep nvinfer

# Check CUDA version
nvcc --version

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Profile with Nsight Systems
nsys profile -o report ./your_application
```

---

## 9. Version Compatibility Matrix

| trt_engine | TensorRT | CUDA | cuDNN | ONNX Opset |
|-----------|----------|------|-------|-----------|
| 1.0.x | 8.6 - 10.x | 11.8+ | 8.6+ | 13+ |

### Important Notes

- **Engine files are NOT portable** across TensorRT versions. Rebuild when upgrading.
- **Engine files are NOT portable** across GPU architectures. Build per-architecture.
- **Calibration caches** are reusable within the same TensorRT major version.
- **Timing caches** are specific to the device, CUDA version, and TensorRT version.
- **ONNX models** are portable and should be used as the source of truth.

---

## 10. Integrating into Your Project

### CMake

```cmake
find_package(trt_engine REQUIRED)
target_link_libraries(your_target PRIVATE trt_engine::trt_engine)
```

### pkg-config

```bash
pkg-config --cflags --libs trt_engine
```

### Manual Linking

```bash
g++ -std=c++17 your_app.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -ltrt_engine -lnvinfer -lnvonnxparser -lcudart \
    -o your_app
```
