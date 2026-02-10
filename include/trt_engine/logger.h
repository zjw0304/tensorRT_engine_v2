#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <trt_engine/types.h>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace trt_engine {

class Logger : public nvinfer1::ILogger {
public:
    static Logger& instance();

    // nvinfer1::ILogger interface
    void log(Severity severity, const char* msg) noexcept override;

    // Configuration
    void set_severity(LogSeverity severity);
    LogSeverity get_severity() const;

    void enable_file_output(const std::string& path);
    void disable_file_output();

    // Convenience logging methods
    void error(const std::string& msg);
    void warning(const std::string& msg);
    void info(const std::string& msg);
    void verbose(const std::string& msg);

private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string format_message(LogSeverity severity, const char* msg) const;
    const char* severity_color(LogSeverity severity) const;

    mutable std::mutex mutex_;
    LogSeverity        min_severity_ = LogSeverity::WARNING;
    std::ofstream      file_stream_;
    bool               file_output_enabled_ = false;
};

// Global accessor
Logger& get_logger();

}  // namespace trt_engine

// ── Error checking macros ──────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                            \
        if (status != cudaSuccess) {                                            \
            std::string msg = std::string("CUDA error at ") + __FILE__ + ":" +  \
                              std::to_string(__LINE__) + " - " +                \
                              cudaGetErrorString(status);                       \
            trt_engine::get_logger().error(msg);                                \
            throw std::runtime_error(msg);                                      \
        }                                                                       \
    } while (0)

#define TRT_CHECK(expr)                                                         \
    do {                                                                        \
        if (!(expr)) {                                                          \
            std::string msg = std::string("TensorRT check failed at ") +        \
                              __FILE__ + ":" + std::to_string(__LINE__) +        \
                              " - " + #expr;                                    \
            trt_engine::get_logger().error(msg);                                \
            throw std::runtime_error(msg);                                      \
        }                                                                       \
    } while (0)
