// cuda_graph_inference.cpp
// Demonstrates using CUDA graphs for reduced kernel launch overhead
// during repeated same-shape inference.

#include <trt_engine/trt_engine.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.engine>" << std::endl;
        return 1;
    }

    const std::string engine_path = argv[1];
    const int iterations = 500;

    try {
        auto& logger = trt_engine::get_logger();
        logger.set_severity(trt_engine::LogSeverity::INFO);

        // Create the inference engine
        auto engine = trt_engine::InferenceEngine::create(engine_path);
        engine->warmup(10);

        // Determine input size from engine metadata
        auto input_info = engine->get_input_info();
        size_t input_elems = 1;
        for (int d : input_info[0].shape) {
            input_elems *= (d > 0) ? d : 1;
        }

        std::vector<std::vector<float>> inputs = {
            std::vector<float>(input_elems, 0.5f)
        };

        // Benchmark WITHOUT CUDA graphs (standard inference)
        std::vector<float> latencies_standard;
        latencies_standard.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            auto result = engine->infer(inputs);
            if (result.success) {
                latencies_standard.push_back(result.latency_ms);
            }
        }

        // Benchmark WITH CUDA graphs
        // Create a dedicated context and graph executor
        trt_engine::CudaGraphExecutor graph;
        trt_engine::CudaStream stream;

        // Access the underlying engine to create a context for graph capture
        auto* trt_eng = engine->get_engine();
        trt_engine::UniqueContext ctx(trt_eng->createExecutionContext());

        // Bind I/O tensors
        int nb_io = trt_eng->getNbIOTensors();
        std::vector<trt_engine::DeviceBuffer> io_buffers;

        for (int i = 0; i < nb_io; ++i) {
            const char* name = trt_eng->getIOTensorName(i);
            nvinfer1::Dims dims = trt_eng->getTensorShape(name);
            int64_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) vol *= dims.d[d];
            nvinfer1::DataType dt = trt_eng->getTensorDataType(name);
            size_t bytes = static_cast<size_t>(vol) * trt_engine::datatype_size(dt);
            io_buffers.emplace_back(bytes);
            ctx->setTensorAddress(name, io_buffers.back().data());
        }

        // Copy input data to device
        trt_engine::copy_to_device(io_buffers[0].data(), inputs[0].data(),
                                   input_elems * sizeof(float), stream.get());

        // Capture the CUDA graph
        if (!graph.capture(ctx.get(), stream.get())) {
            std::cerr << "CUDA graph capture failed." << std::endl;
            return 1;
        }

        // Run with graph replay
        trt_engine::CudaEvent start, end;
        std::vector<float> latencies_graph;
        latencies_graph.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            start.record(stream.get());
            graph.launch(stream.get());
            end.record(stream.get());
            stream.synchronize();
            latencies_graph.push_back(
                trt_engine::CudaEvent::elapsed_time(start, end));
        }

        // Report results
        auto mean = [](const std::vector<float>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
        };

        float std_mean = mean(latencies_standard);
        float graph_mean = mean(latencies_graph);

        std::cout << "\n=== CUDA Graph Benchmark ===" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Standard mean: " << std_mean << " ms" << std::endl;
        std::cout << "Graph mean:    " << graph_mean << " ms" << std::endl;
        std::cout << "Speedup:       " << std_mean / graph_mean << "x" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
