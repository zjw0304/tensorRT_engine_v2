"""
batch_inference.py
Demonstrates dynamic batching using the trt_engine DynamicBatcher.
Multiple individual requests are automatically batched for higher throughput.
"""

import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import trt_engine
except ImportError:
    print("Error: trt_engine Python module not found.")
    print("Build with -DTRT_ENGINE_BUILD_PYTHON=ON and add to PYTHONPATH.")
    sys.exit(1)


def submit_request(batcher, input_data):
    """Submit a single inference request and return the result."""
    future = batcher.submit([input_data.tolist()])
    return future.get()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.engine>")
        sys.exit(1)

    engine_path = sys.argv[1]
    max_batch_size = 16
    max_wait_ms = 10
    num_requests = 100

    # Configure logger
    logger = trt_engine.get_logger()
    logger.set_severity(trt_engine.LogSeverity.INFO)

    # Create the inference engine
    print(f"Loading engine from: {engine_path}")
    engine = trt_engine.InferenceEngine.create(engine_path)
    engine.warmup(5)

    # Query input dimensions
    input_info = engine.get_input_info()
    num_elements = 1
    for d in input_info[0].shape:
        num_elements *= max(d, 1)

    # Create the dynamic batcher
    print(f"Creating DynamicBatcher: max_batch={max_batch_size}, "
          f"wait={max_wait_ms}ms")
    batcher = trt_engine.DynamicBatcher(engine, max_batch_size, max_wait_ms)

    # Generate random input data for each request
    inputs = [np.random.randn(num_elements).astype(np.float32)
              for _ in range(num_requests)]

    # Submit requests from multiple threads to simulate concurrent clients
    print(f"\nSubmitting {num_requests} requests from multiple threads...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(submit_request, batcher, inp)
            for inp in inputs
        ]
        results = [f.result() for f in futures]

    elapsed = time.time() - start_time

    # Report results
    success_count = sum(1 for r in results if r.success)
    latencies = [r.latency_ms for r in results if r.success]

    print(f"\n=== Results ===")
    print(f"Successful: {success_count}/{num_requests}")
    print(f"Wall time:  {elapsed * 1000:.1f} ms")
    print(f"Throughput: {success_count / elapsed:.1f} infer/sec")

    if latencies:
        latencies.sort()
        print(f"Avg latency (GPU): {sum(latencies) / len(latencies):.2f} ms")
        print(f"P50 latency: {latencies[len(latencies) // 2]:.2f} ms")
        p95_idx = int(len(latencies) * 0.95)
        print(f"P95 latency: {latencies[p95_idx]:.2f} ms")


if __name__ == "__main__":
    main()
