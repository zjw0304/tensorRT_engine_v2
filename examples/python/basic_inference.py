"""
basic_inference.py
Demonstrates basic TensorRT inference using the trt_engine Python bindings.
"""

import sys
import numpy as np

try:
    import trt_engine
except ImportError:
    print("Error: trt_engine Python module not found.")
    print("Build with -DTRT_ENGINE_BUILD_PYTHON=ON and add to PYTHONPATH.")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.engine>")
        sys.exit(1)

    engine_path = sys.argv[1]

    # Configure logger
    logger = trt_engine.get_logger()
    logger.set_severity(trt_engine.LogSeverity.INFO)

    # Load the engine
    print(f"Loading engine from: {engine_path}")
    engine = trt_engine.InferenceEngine.create(engine_path)

    # Query tensor information
    input_info = engine.get_input_info()
    output_info = engine.get_output_info()

    print("\nInputs:")
    for info in input_info:
        print(f"  {info.name}: shape={info.shape}, dtype={info.dtype}")

    print("Outputs:")
    for info in output_info:
        print(f"  {info.name}: shape={info.shape}, dtype={info.dtype}")

    # Warm up
    engine.warmup(5)

    # Prepare input data
    input_shape = input_info[0].shape
    num_elements = 1
    for d in input_shape:
        num_elements *= max(d, 1)  # treat -1 (dynamic) as 1

    input_data = np.random.randn(num_elements).astype(np.float32)

    # Run inference
    result = engine.infer([input_data.tolist()])

    if result.success:
        print(f"\nInference succeeded!")
        print(f"Latency: {result.latency_ms:.2f} ms")
        print(f"Output elements: {len(result.outputs[0])}")
        output = np.array(result.outputs[0])
        print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, "
              f"mean={output.mean():.4f}")
    else:
        print(f"Inference failed: {result.error_msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
