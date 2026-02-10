"""
test_python.py - Pytest tests for the trt_engine Python bindings.

Tests import, basic API, enum access, config creation, and utility functions.
Tests that require an actual GPU or TensorRT engine are skipped when not
available.
"""

import os
import sys
import pytest
import numpy as np

# Try to import the C++ bindings; skip all tests if unavailable
try:
    import trt_engine
    from trt_engine import (
        Precision,
        LogSeverity,
        ModelFormat,
        BuilderConfig,
        EngineConfig,
        DeviceConfig,
        InferenceResult,
        TensorInfo,
        DynamicShapeProfile,
        PerformanceStats,
        PerformanceProfiler,
    )
    HAS_BINDINGS = True
except ImportError:
    HAS_BINDINGS = False

skip_no_bindings = pytest.mark.skipif(
    not HAS_BINDINGS, reason="trt_engine C++ bindings not available"
)


def gpu_available() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        count = trt_engine.get_device_count()
        return count > 0
    except Exception:
        return False


def find_test_engine() -> str:
    """Find a test engine file."""
    candidates = [
        "/tmp/test_model.engine",
        "test_data/test_model.engine",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


# ── Import and version tests ────────────────────────────────────────────────


@skip_no_bindings
class TestImport:
    def test_import_module(self):
        assert trt_engine is not None

    def test_version(self):
        assert hasattr(trt_engine, "__version__")
        assert trt_engine.__version__ == "1.0.0"

    def test_all_list(self):
        assert hasattr(trt_engine, "__all__")
        assert len(trt_engine.__all__) > 0


# ── Enum tests ───────────────────────────────────────────────────────────────


@skip_no_bindings
class TestEnums:
    def test_precision_values(self):
        assert Precision.FP32 is not None
        assert Precision.FP16 is not None
        assert Precision.INT8 is not None
        assert Precision.FP8 is not None

    def test_log_severity_values(self):
        assert LogSeverity.INTERNAL_ERROR is not None
        assert LogSeverity.ERROR is not None
        assert LogSeverity.WARNING is not None
        assert LogSeverity.INFO is not None
        assert LogSeverity.VERBOSE is not None

    def test_model_format_values(self):
        assert ModelFormat.ONNX is not None
        assert ModelFormat.TENSORFLOW is not None
        assert ModelFormat.PYTORCH is not None
        assert ModelFormat.TENSORRT_ENGINE is not None
        assert ModelFormat.UNKNOWN is not None

    def test_precision_to_string(self):
        assert trt_engine.precision_to_string(Precision.FP32) == "FP32"
        assert trt_engine.precision_to_string(Precision.FP16) == "FP16"
        assert trt_engine.precision_to_string(Precision.INT8) == "INT8"
        assert trt_engine.precision_to_string(Precision.FP8) == "FP8"

    def test_string_to_precision(self):
        assert trt_engine.string_to_precision("FP32") == Precision.FP32
        assert trt_engine.string_to_precision("fp16") == Precision.FP16
        assert trt_engine.string_to_precision("INT8") == Precision.INT8

    def test_invalid_precision_string(self):
        with pytest.raises(Exception):
            trt_engine.string_to_precision("INVALID")


# ── Config tests ─────────────────────────────────────────────────────────────


@skip_no_bindings
class TestConfigs:
    def test_builder_config_defaults(self):
        config = BuilderConfig()
        assert config.precision == Precision.FP32
        assert config.max_workspace_size == 1 << 30
        assert config.enable_cuda_graph is False
        assert config.enable_dla is False
        assert len(config.dynamic_shapes) == 0

    def test_builder_config_set(self):
        config = BuilderConfig()
        config.precision = Precision.FP16
        config.max_workspace_size = 2 << 30
        config.enable_cuda_graph = True
        assert config.precision == Precision.FP16
        assert config.max_workspace_size == 2 << 30
        assert config.enable_cuda_graph is True

    def test_engine_config_defaults(self):
        config = EngineConfig()
        assert config.device_id == 0
        assert config.context_pool_size == 2
        assert config.enable_cuda_graph is False
        assert config.thread_pool_size == 2

    def test_device_config_defaults(self):
        config = DeviceConfig()
        assert config.device_id == 0
        assert config.workspace_size == 1 << 30

    def test_dynamic_shape_profile(self):
        prof = DynamicShapeProfile()
        prof.name = "input"
        prof.min_dims = [1, 3, 224, 224]
        prof.opt_dims = [4, 3, 224, 224]
        prof.max_dims = [16, 3, 224, 224]
        assert prof.name == "input"
        assert prof.min_dims == [1, 3, 224, 224]


# ── Result structs ───────────────────────────────────────────────────────────


@skip_no_bindings
class TestResultStructs:
    def test_inference_result_defaults(self):
        result = InferenceResult()
        assert result.success is False
        assert result.latency_ms == 0.0
        assert len(result.outputs) == 0
        assert result.error_msg == ""

    def test_tensor_info_defaults(self):
        ti = TensorInfo()
        assert ti.name == ""
        assert len(ti.shape) == 0
        assert ti.size_bytes == 0

    def test_performance_stats_defaults(self):
        stats = PerformanceStats()
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.total_inferences == 0


# ── PerformanceProfiler tests ────────────────────────────────────────────────


@skip_no_bindings
class TestPerformanceProfiler:
    def test_create(self):
        profiler = PerformanceProfiler()
        assert profiler.count() == 0

    def test_record_inference(self):
        profiler = PerformanceProfiler()
        profiler.record_inference(1.0)
        profiler.record_inference(2.0)
        profiler.record_inference(3.0)
        assert profiler.count() == 3

    def test_get_statistics(self):
        profiler = PerformanceProfiler()
        for i in range(1, 101):
            profiler.record_inference(float(i))

        stats = profiler.get_statistics()
        assert stats.total_inferences == 100
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        assert abs(stats.mean_ms - 50.5) < 0.01
        assert stats.throughput_fps > 0

    def test_reset(self):
        profiler = PerformanceProfiler()
        profiler.record_inference(5.0)
        assert profiler.count() == 1
        profiler.reset()
        assert profiler.count() == 0

    def test_report_text(self):
        profiler = PerformanceProfiler()
        profiler.record_inference(1.5)
        profiler.record_inference(2.5)
        text = profiler.report_text()
        assert "Performance Report" in text
        assert "Min" in text

    def test_report_json(self):
        profiler = PerformanceProfiler()
        profiler.record_inference(1.0)
        json_str = profiler.report_json()
        assert "total_inferences" in json_str
        assert "latency_ms" in json_str
        assert "throughput_fps" in json_str

    def test_empty_statistics(self):
        profiler = PerformanceProfiler()
        stats = profiler.get_statistics()
        assert stats.total_inferences == 0
        assert stats.min_ms == 0.0


# ── Logger tests ─────────────────────────────────────────────────────────────


@skip_no_bindings
class TestLogger:
    def test_singleton(self):
        logger = trt_engine.Logger.instance()
        assert logger is not None

    def test_set_severity(self):
        logger = trt_engine.Logger.instance()
        original = logger.get_severity()
        logger.set_severity(LogSeverity.VERBOSE)
        assert logger.get_severity() == LogSeverity.VERBOSE
        logger.set_severity(original)

    def test_log_calls(self):
        logger = trt_engine.Logger.instance()
        original = logger.get_severity()
        logger.set_severity(LogSeverity.VERBOSE)
        # Should not raise
        logger.info("test info from Python")
        logger.warning("test warning from Python")
        logger.set_severity(original)


# ── GPU-dependent tests ──────────────────────────────────────────────────────


@skip_no_bindings
class TestGPUFeatures:
    @pytest.mark.skipif(
        not HAS_BINDINGS or not gpu_available(),
        reason="No GPU available",
    )
    def test_device_count(self):
        count = trt_engine.get_device_count()
        assert count >= 1

    @pytest.mark.skipif(
        not HAS_BINDINGS or not gpu_available(),
        reason="No GPU available",
    )
    def test_device_properties(self):
        props = trt_engine.get_device_properties(0)
        assert props.name != ""
        assert props.total_global_memory > 0

    @pytest.mark.skipif(
        not HAS_BINDINGS or not gpu_available(),
        reason="No GPU available",
    )
    def test_gpu_metrics(self):
        # These may return 0 if NVML is not linked, but should not crash
        _ = PerformanceProfiler.gpu_utilization(0)
        _ = PerformanceProfiler.memory_used(0)
        _ = PerformanceProfiler.temperature(0)
        _ = PerformanceProfiler.power_usage(0)


# ── Utility function tests ──────────────────────────────────────────────────


class TestUtils:
    def test_load_image_missing_file(self):
        from trt_engine.utils import load_image

        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.jpg")

    def test_visualize_results(self):
        from trt_engine.utils import visualize_results

        scores = np.array([0.1, 0.7, 0.15, 0.05], dtype=np.float32)
        labels = ["cat", "dog", "bird", "fish"]
        # Should print without error
        visualize_results(scores, labels, top_k=3)

    def test_visualize_results_no_labels(self):
        from trt_engine.utils import visualize_results

        scores = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        visualize_results(scores, top_k=2)

    def test_download_model_invalid_url(self):
        from trt_engine.utils import download_model

        # Should raise on invalid URL
        with pytest.raises(Exception):
            download_model(
                "http://invalid-host-xyz-nonexistent.example.com/model.onnx",
                "/tmp/trt_test_download.onnx",
            )
