#include <trt_engine/logger.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace trt_engine {

Logger::Logger() = default;
Logger::~Logger() = default;

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::log(Severity severity, const char* msg) noexcept {
    LogSeverity sev = trt_severity_to_log(severity);

    // Filter by minimum severity (lower enum value = higher severity)
    if (static_cast<int>(sev) > static_cast<int>(min_severity_)) {
        return;
    }

    std::string formatted = format_message(sev, msg);

    std::lock_guard<std::mutex> lock(mutex_);

    // Console output with color
    const char* color = severity_color(sev);
    std::cerr << color << formatted << "\033[0m" << std::endl;

    // File output
    if (file_output_enabled_ && file_stream_.is_open()) {
        file_stream_ << formatted << std::endl;
        file_stream_.flush();
    }
}

void Logger::set_severity(LogSeverity severity) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_severity_ = severity;
}

LogSeverity Logger::get_severity() const {
    return min_severity_;
}

void Logger::enable_file_output(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
    file_stream_.open(path, std::ios::app);
    if (!file_stream_.is_open()) {
        std::cerr << "[TRT_ENGINE] Failed to open log file: " << path << std::endl;
        file_output_enabled_ = false;
        return;
    }
    file_output_enabled_ = true;
}

void Logger::disable_file_output() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_output_enabled_ = false;
    if (file_stream_.is_open()) {
        file_stream_.close();
    }
}

void Logger::error(const std::string& msg) {
    log(Severity::kERROR, msg.c_str());
}

void Logger::warning(const std::string& msg) {
    log(Severity::kWARNING, msg.c_str());
}

void Logger::info(const std::string& msg) {
    log(Severity::kINFO, msg.c_str());
}

void Logger::verbose(const std::string& msg) {
    log(Severity::kVERBOSE, msg.c_str());
}

std::string Logger::format_message(LogSeverity severity, const char* msg) const {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::tm tm_buf{};
    localtime_r(&time_t_now, &tm_buf);

    std::ostringstream oss;
    oss << "[" << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count()
        << "] [TRT_ENGINE] [" << severity_to_string(severity) << "] "
        << msg;
    return oss.str();
}

const char* Logger::severity_color(LogSeverity severity) const {
    switch (severity) {
        case LogSeverity::INTERNAL_ERROR: return "\033[1;31m";  // bold red
        case LogSeverity::ERROR:          return "\033[31m";    // red
        case LogSeverity::WARNING:        return "\033[33m";    // yellow
        case LogSeverity::INFO:           return "\033[0m";     // default
        case LogSeverity::VERBOSE:        return "\033[90m";    // gray
    }
    return "\033[0m";
}

Logger& get_logger() {
    return Logger::instance();
}

}  // namespace trt_engine
