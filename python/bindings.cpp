#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <trt_engine/trt_engine.h>
#include <trt_engine/profiler.h>

#include <future>
#include <memory>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace {

// Convert a vector of float vectors to numpy arrays
py::list outputs_to_numpy(const trt_engine::InferenceResult& result) {
    py::list out;
    for (const auto& tensor : result.outputs) {
        py::array_t<float> arr(
            static_cast<py::ssize_t>(tensor.size()));
        auto buf = arr.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tensor.size()); ++i) {
            buf(i) = tensor[static_cast<size_t>(i)];
        }
        out.append(std::move(arr));
    }
    return out;
}

// Convert numpy input arrays to vector<vector<float>>
std::vector<std::vector<float>> numpy_inputs_to_vectors(const py::list& inputs) {
    std::vector<std::vector<float>> result;
    result.reserve(py::len(inputs));
    for (auto item : inputs) {
        py::array_t<float, py::array::c_style | py::array::forcecast> arr(item);
        auto buf = arr.unchecked<1>();
        std::vector<float> vec(static_cast<size_t>(buf.size()));
        for (py::ssize_t i = 0; i < buf.size(); ++i) {
            vec[static_cast<size_t>(i)] = buf(i);
        }
        result.push_back(std::move(vec));
    }
    return result;
}

// Convert a flat numpy array to a vector<float>
std::vector<float> numpy_to_vector(const py::array_t<float>& arr) {
    auto r = arr.unchecked();
    std::vector<float> vec(static_cast<size_t>(r.size()));
    for (py::ssize_t i = 0; i < r.size(); ++i) {
        vec[static_cast<size_t>(i)] = r(i);
    }
    return vec;
}

}  // namespace

PYBIND11_MODULE(trt_engine, m) {
    m.doc() = "TensorRT High-Performance GPU Inference Engine";

    // ── Enums ────────────────────────────────────────────────────────────

    py::enum_<trt_engine::Precision>(m, "Precision")
        .value("FP32", trt_engine::Precision::FP32)
        .value("FP16", trt_engine::Precision::FP16)
        .value("INT8", trt_engine::Precision::INT8)
        .value("FP8",  trt_engine::Precision::FP8)
        .export_values();

    py::enum_<trt_engine::LogSeverity>(m, "LogSeverity")
        .value("INTERNAL_ERROR", trt_engine::LogSeverity::INTERNAL_ERROR)
        .value("ERROR",          trt_engine::LogSeverity::ERROR)
        .value("WARNING",        trt_engine::LogSeverity::WARNING)
        .value("INFO",           trt_engine::LogSeverity::INFO)
        .value("VERBOSE",        trt_engine::LogSeverity::VERBOSE)
        .export_values();

    py::enum_<trt_engine::ModelFormat>(m, "ModelFormat")
        .value("ONNX",            trt_engine::ModelFormat::ONNX)
        .value("TENSORFLOW",      trt_engine::ModelFormat::TENSORFLOW)
        .value("PYTORCH",         trt_engine::ModelFormat::PYTORCH)
        .value("TENSORRT_ENGINE", trt_engine::ModelFormat::TENSORRT_ENGINE)
        .value("UNKNOWN",         trt_engine::ModelFormat::UNKNOWN)
        .export_values();

    // ── Structs ──────────────────────────────────────────────────────────

    py::class_<trt_engine::DynamicShapeProfile>(m, "DynamicShapeProfile")
        .def(py::init<>())
        .def_readwrite("name",     &trt_engine::DynamicShapeProfile::name)
        .def_readwrite("min_dims", &trt_engine::DynamicShapeProfile::min_dims)
        .def_readwrite("opt_dims", &trt_engine::DynamicShapeProfile::opt_dims)
        .def_readwrite("max_dims", &trt_engine::DynamicShapeProfile::max_dims);

    py::class_<trt_engine::BuilderConfig>(m, "BuilderConfig")
        .def(py::init<>())
        .def_readwrite("precision",          &trt_engine::BuilderConfig::precision)
        .def_readwrite("max_workspace_size", &trt_engine::BuilderConfig::max_workspace_size)
        .def_readwrite("enable_cuda_graph",  &trt_engine::BuilderConfig::enable_cuda_graph)
        .def_readwrite("enable_dla",         &trt_engine::BuilderConfig::enable_dla)
        .def_readwrite("dla_core",           &trt_engine::BuilderConfig::dla_core)
        .def_readwrite("timing_cache_path",  &trt_engine::BuilderConfig::timing_cache_path)
        .def_readwrite("max_aux_streams",    &trt_engine::BuilderConfig::max_aux_streams)
        .def_readwrite("strongly_typed",     &trt_engine::BuilderConfig::strongly_typed)
        .def_readwrite("builder_optimization_level", &trt_engine::BuilderConfig::builder_optimization_level)
        .def_readwrite("auto_timing_cache",  &trt_engine::BuilderConfig::auto_timing_cache)
        .def_readwrite("dynamic_shapes",     &trt_engine::BuilderConfig::dynamic_shapes);

    py::class_<trt_engine::EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("device_id",         &trt_engine::EngineConfig::device_id)
        .def_readwrite("context_pool_size", &trt_engine::EngineConfig::context_pool_size)
        .def_readwrite("enable_cuda_graph", &trt_engine::EngineConfig::enable_cuda_graph)
        .def_readwrite("thread_pool_size",  &trt_engine::EngineConfig::thread_pool_size);

    py::class_<trt_engine::DeviceConfig>(m, "DeviceConfig")
        .def(py::init<>())
        .def_readwrite("device_id",      &trt_engine::DeviceConfig::device_id)
        .def_readwrite("workspace_size", &trt_engine::DeviceConfig::workspace_size);

    py::class_<trt_engine::InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("outputs",    &trt_engine::InferenceResult::outputs)
        .def_readwrite("latency_ms", &trt_engine::InferenceResult::latency_ms)
        .def_readwrite("success",    &trt_engine::InferenceResult::success)
        .def_readwrite("error_msg",  &trt_engine::InferenceResult::error_msg)
        .def("get_outputs_numpy", [](const trt_engine::InferenceResult& self) {
            return outputs_to_numpy(self);
        }, "Get outputs as a list of numpy arrays");

    py::class_<trt_engine::TensorInfo>(m, "TensorInfo")
        .def(py::init<>())
        .def_readwrite("name",       &trt_engine::TensorInfo::name)
        .def_readwrite("shape",      &trt_engine::TensorInfo::shape)
        .def_readwrite("dtype",      &trt_engine::TensorInfo::dtype)
        .def_readwrite("size_bytes", &trt_engine::TensorInfo::size_bytes);

    py::class_<trt_engine::DeviceProperties>(m, "DeviceProperties")
        .def(py::init<>())
        .def_readwrite("name",                      &trt_engine::DeviceProperties::name)
        .def_readwrite("compute_capability_major",   &trt_engine::DeviceProperties::compute_capability_major)
        .def_readwrite("compute_capability_minor",   &trt_engine::DeviceProperties::compute_capability_minor)
        .def_readwrite("total_global_memory",        &trt_engine::DeviceProperties::total_global_memory)
        .def_readwrite("multi_processor_count",      &trt_engine::DeviceProperties::multi_processor_count);

    py::class_<trt_engine::PerformanceStats>(m, "PerformanceStats")
        .def(py::init<>())
        .def_readwrite("min_ms",           &trt_engine::PerformanceStats::min_ms)
        .def_readwrite("max_ms",           &trt_engine::PerformanceStats::max_ms)
        .def_readwrite("mean_ms",          &trt_engine::PerformanceStats::mean_ms)
        .def_readwrite("p50_ms",           &trt_engine::PerformanceStats::p50_ms)
        .def_readwrite("p95_ms",           &trt_engine::PerformanceStats::p95_ms)
        .def_readwrite("p99_ms",           &trt_engine::PerformanceStats::p99_ms)
        .def_readwrite("throughput_fps",   &trt_engine::PerformanceStats::throughput_fps)
        .def_readwrite("total_inferences", &trt_engine::PerformanceStats::total_inferences);

    // ── ModelConverter ────────────────────────────────────────────────────

    py::class_<trt_engine::ModelConverter>(m, "ModelConverter")
        .def_static("detect_format",       &trt_engine::ModelConverter::detect_format,
                    py::arg("path"))
        .def_static("convert",             &trt_engine::ModelConverter::convert,
                    py::arg("input_path"), py::arg("output_path"))
        .def_static("validate_onnx",       &trt_engine::ModelConverter::validate_onnx,
                    py::arg("path"))
        .def_static("optimize_onnx",       &trt_engine::ModelConverter::optimize_onnx,
                    py::arg("input_path"), py::arg("output_path"));

    // ── EngineBuilder ────────────────────────────────────────────────────

    py::class_<trt_engine::EngineBuilder>(m, "EngineBuilder")
        .def(py::init([](){ return trt_engine::EngineBuilder(trt_engine::get_logger()); }))
        .def("build_engine", &trt_engine::EngineBuilder::build_engine,
             py::arg("onnx_path"), py::arg("config"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("save_engine", &trt_engine::EngineBuilder::save_engine,
                    py::arg("engine_data"), py::arg("path"))
        .def_static("load_engine", &trt_engine::EngineBuilder::load_engine,
                    py::arg("path"));

    // ── InferenceEngine ──────────────────────────────────────────────────

    py::class_<trt_engine::InferenceEngine, std::shared_ptr<trt_engine::InferenceEngine>>(m, "InferenceEngine")
        .def_static("create",
            [](const std::string& engine_path, const trt_engine::EngineConfig& config) {
                py::gil_scoped_release release;
                return std::shared_ptr<trt_engine::InferenceEngine>(
                    trt_engine::InferenceEngine::create(engine_path, config).release());
            },
            py::arg("engine_path"),
            py::arg("config") = trt_engine::EngineConfig{})
        .def("infer",
            [](trt_engine::InferenceEngine& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                py::gil_scoped_release release;
                return self.infer(vecs);
            },
            py::arg("inputs"),
            "Run synchronous inference. inputs is a list of numpy arrays.")
        .def("infer_async",
            [](trt_engine::InferenceEngine& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                auto future = self.infer_async(vecs);
                // Wait and return result (release GIL while waiting)
                py::gil_scoped_release release;
                return future.get();
            },
            py::arg("inputs"),
            "Run async inference and wait for result.")
        .def("warmup", &trt_engine::InferenceEngine::warmup,
             py::arg("iterations") = 5,
             py::call_guard<py::gil_scoped_release>())
        .def("get_input_info",  &trt_engine::InferenceEngine::get_input_info)
        .def("get_output_info", &trt_engine::InferenceEngine::get_output_info)
        .def("set_input_shape", &trt_engine::InferenceEngine::set_input_shape,
             py::arg("name"), py::arg("dims"));

    // ── DynamicBatcher ───────────────────────────────────────────────────

    py::class_<trt_engine::DynamicBatcher>(m, "DynamicBatcher")
        .def(py::init<std::shared_ptr<trt_engine::InferenceEngine>, int, int>(),
             py::arg("engine"), py::arg("max_batch_size"), py::arg("max_wait_time_ms"))
        .def("submit",
            [](trt_engine::DynamicBatcher& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                auto future = self.submit(vecs);
                py::gil_scoped_release release;
                return future.get();
            },
            py::arg("inputs"),
            "Submit a single inference request and wait for batched result.");

    // ── MultiStreamEngine ────────────────────────────────────────────────

    py::class_<trt_engine::MultiStreamEngine>(m, "MultiStreamEngine")
        .def(py::init<const std::string&, int, const trt_engine::EngineConfig&>(),
             py::arg("engine_path"), py::arg("num_streams"),
             py::arg("config") = trt_engine::EngineConfig{},
             py::call_guard<py::gil_scoped_release>())
        .def("infer",
            [](trt_engine::MultiStreamEngine& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                py::gil_scoped_release release;
                return self.infer(vecs);
            },
            py::arg("inputs"))
        .def("submit",
            [](trt_engine::MultiStreamEngine& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                auto future = self.submit(vecs);
                py::gil_scoped_release release;
                return future.get();
            },
            py::arg("inputs"))
        .def("shutdown", &trt_engine::MultiStreamEngine::shutdown)
        .def_property_readonly("num_streams", &trt_engine::MultiStreamEngine::num_streams);

    // ── MultiGPUEngine ───────────────────────────────────────────────────

    py::class_<trt_engine::MultiGPUEngine>(m, "MultiGPUEngine")
        .def(py::init<const std::string&, const std::vector<int>&,
                       const trt_engine::EngineConfig&>(),
             py::arg("engine_path"), py::arg("device_ids"),
             py::arg("config") = trt_engine::EngineConfig{},
             py::call_guard<py::gil_scoped_release>())
        .def("infer",
            [](trt_engine::MultiGPUEngine& self, const py::list& inputs) {
                auto vecs = numpy_inputs_to_vectors(inputs);
                py::gil_scoped_release release;
                return self.infer(vecs);
            },
            py::arg("inputs"))
        .def("get_device_count", &trt_engine::MultiGPUEngine::get_device_count)
        .def("get_device_ids",   &trt_engine::MultiGPUEngine::get_device_ids);

    // ── PerformanceProfiler ──────────────────────────────────────────────

    py::class_<trt_engine::PerformanceProfiler>(m, "PerformanceProfiler")
        .def(py::init<>())
        .def("record_inference", &trt_engine::PerformanceProfiler::record_inference,
             py::arg("latency_ms"))
        .def("get_statistics",   &trt_engine::PerformanceProfiler::get_statistics)
        .def("report_text",      &trt_engine::PerformanceProfiler::report_text)
        .def("report_json",      &trt_engine::PerformanceProfiler::report_json)
        .def("reset",            &trt_engine::PerformanceProfiler::reset)
        .def("count",            &trt_engine::PerformanceProfiler::count)
        .def_static("gpu_utilization", &trt_engine::PerformanceProfiler::gpu_utilization,
                    py::arg("device_id") = 0)
        .def_static("memory_used",     &trt_engine::PerformanceProfiler::memory_used,
                    py::arg("device_id") = 0)
        .def_static("temperature",     &trt_engine::PerformanceProfiler::temperature,
                    py::arg("device_id") = 0)
        .def_static("power_usage",     &trt_engine::PerformanceProfiler::power_usage,
                    py::arg("device_id") = 0);

    // ── GpuMetrics ───────────────────────────────────────────────────────

    py::class_<trt_engine::GpuMetrics>(m, "GpuMetrics")
        .def(py::init<>())
        .def_readwrite("gpu_utilization_percent", &trt_engine::GpuMetrics::gpu_utilization_percent)
        .def_readwrite("memory_used_bytes",       &trt_engine::GpuMetrics::memory_used_bytes)
        .def_readwrite("memory_total_bytes",      &trt_engine::GpuMetrics::memory_total_bytes)
        .def_readwrite("temperature_celsius",     &trt_engine::GpuMetrics::temperature_celsius)
        .def_readwrite("power_usage_milliwatts",  &trt_engine::GpuMetrics::power_usage_milliwatts);

    // ── Logger ───────────────────────────────────────────────────────────

    py::class_<trt_engine::Logger, std::unique_ptr<trt_engine::Logger, py::nodelete>>(m, "Logger")
        .def_static("instance", &trt_engine::Logger::instance,
                    py::return_value_policy::reference)
        .def("set_severity", &trt_engine::Logger::set_severity, py::arg("severity"))
        .def("get_severity", &trt_engine::Logger::get_severity)
        .def("enable_file_output",  &trt_engine::Logger::enable_file_output, py::arg("path"))
        .def("disable_file_output", &trt_engine::Logger::disable_file_output)
        .def("error",   &trt_engine::Logger::error,   py::arg("msg"))
        .def("warning", &trt_engine::Logger::warning, py::arg("msg"))
        .def("info",    &trt_engine::Logger::info,    py::arg("msg"))
        .def("verbose", &trt_engine::Logger::verbose, py::arg("msg"));

    // ── Utility functions ────────────────────────────────────────────────

    m.def("get_device_count", &trt_engine::get_device_count,
          "Get the number of CUDA devices");
    m.def("get_device_properties", &trt_engine::get_device_properties,
          py::arg("device_id"), "Get properties of a CUDA device");
    m.def("precision_to_string", &trt_engine::precision_to_string,
          py::arg("precision"));
    m.def("string_to_precision", &trt_engine::string_to_precision,
          py::arg("s"));

    // ── Version ──────────────────────────────────────────────────────────

    m.attr("__version__") = "1.0.0";
    m.attr("VERSION_MAJOR") = TRT_ENGINE_VERSION_MAJOR;
    m.attr("VERSION_MINOR") = TRT_ENGINE_VERSION_MINOR;
    m.attr("VERSION_PATCH") = TRT_ENGINE_VERSION_PATCH;
}
