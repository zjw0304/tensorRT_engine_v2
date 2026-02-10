# TRT Engine Docker

Docker setup for building and running the TRT Engine library.

## Prerequisites

- Docker >= 20.10
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with driver >= 525

Verify GPU access in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Build the Compile Image

The compile image builds the full project including tests, benchmarks, and Python bindings:

```bash
cd /path/to/trt_engine
docker build -f docker/Dockerfile.compile -t trt-engine-compile .
```

## Build the Runtime Image

The runtime image is a lightweight image containing only the shared library, headers, and Python package. It requires the compile image to be built first:

```bash
docker build -f docker/Dockerfile.runtime -t trt-engine-runtime .
```

## Run Inference in a Container

Start an interactive shell with GPU access:

```bash
docker run --rm -it --gpus all \
    -v /path/to/models:/models \
    trt-engine-runtime bash
```

Inside the container, use the Python bindings:

```python
import trt_engine
# Load and run your TensorRT engine
```

Or use the C++ library directly via the headers and `libtrt_engine.so` installed under `/opt/trt_engine/`.

## Using Docker Compose

Build everything:

```bash
cd docker/
docker compose up compile
```

Start an interactive inference session:

```bash
docker compose run --rm inference
```

## GPU Passthrough

All containers require GPU access via `--gpus`:

```bash
# All GPUs
docker run --gpus all ...

# Specific GPU
docker run --gpus '"device=0"' ...

# Multiple GPUs
docker run --gpus '"device=0,1"' ...
```

For Docker Compose, GPU access is configured via the `deploy.resources.reservations` section and `runtime: nvidia` in `docker-compose.yml`.

## Extracting Build Artifacts

To copy compiled artifacts out of the compile image without running inference:

```bash
docker create --name trt-build trt-engine-compile
docker cp trt-build:/opt/trt_engine/ ./trt_engine_install/
docker rm trt-build
```

## Image Sizes

| Image | Contents | Approximate Size |
|-------|----------|-----------------|
| `trt-engine-compile` | Full build + library + headers | ~15 GB (TensorRT base + build tools) |
| `trt-engine-runtime` | Library + headers + Python only | ~12 GB (TensorRT base, no build tools) |

Both images use the NVIDIA TensorRT container as the base, which includes CUDA, cuDNN, and TensorRT runtimes.
