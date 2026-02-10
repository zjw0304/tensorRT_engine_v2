"""
trt_engine - TensorRT High-Performance GPU Inference Engine
"""

__version__ = "1.0.0"

try:
    from trt_engine_python import (
        # Enums
        Precision,
        LogSeverity,
        ModelFormat,
        # Structs / configs
        DynamicShapeProfile,
        BuilderConfig,
        EngineConfig,
        DeviceConfig,
        InferenceResult,
        TensorInfo,
        DeviceProperties,
        PerformanceStats,
        GpuMetrics,
        # Classes
        ModelConverter,
        EngineBuilder,
        InferenceEngine,
        DynamicBatcher,
        MultiStreamEngine,
        MultiGPUEngine,
        PerformanceProfiler,
        Logger,
        # Functions
        get_device_count,
        get_device_properties,
        precision_to_string,
        string_to_precision,
    )

    __all__ = [
        # Enums
        "Precision",
        "LogSeverity",
        "ModelFormat",
        # Structs / configs
        "DynamicShapeProfile",
        "BuilderConfig",
        "EngineConfig",
        "DeviceConfig",
        "InferenceResult",
        "TensorInfo",
        "DeviceProperties",
        "PerformanceStats",
        "GpuMetrics",
        # Classes
        "ModelConverter",
        "EngineBuilder",
        "InferenceEngine",
        "DynamicBatcher",
        "MultiStreamEngine",
        "MultiGPUEngine",
        "PerformanceProfiler",
        "Logger",
        # Functions
        "get_device_count",
        "get_device_properties",
        "precision_to_string",
        "string_to_precision",
    ]

except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import trt_engine C++ bindings: {e}. "
        "Make sure the library is built and in your Python path.",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
