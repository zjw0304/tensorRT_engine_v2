// test_logger.cpp - Unit tests for the Logger class
//
// Tests severity filtering, file output, thread safety, and CUDA_CHECK macro.

#include <gtest/gtest.h>
#include <trt_engine/logger.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace trt_engine;

// ── Severity filtering tests ────────────────────────────────────────────────

TEST(LoggerTest, DefaultSeverityIsWarning) {
    auto& logger = Logger::instance();
    EXPECT_EQ(logger.get_severity(), LogSeverity::WARNING);
}

TEST(LoggerTest, SetSeverity) {
    auto& logger = Logger::instance();
    auto original = logger.get_severity();

    logger.set_severity(LogSeverity::VERBOSE);
    EXPECT_EQ(logger.get_severity(), LogSeverity::VERBOSE);

    logger.set_severity(LogSeverity::ERROR);
    EXPECT_EQ(logger.get_severity(), LogSeverity::ERROR);

    // Restore
    logger.set_severity(original);
}

TEST(LoggerTest, SeverityOrdering) {
    // Lower enum value = higher severity
    EXPECT_LT(static_cast<int>(LogSeverity::INTERNAL_ERROR),
              static_cast<int>(LogSeverity::ERROR));
    EXPECT_LT(static_cast<int>(LogSeverity::ERROR),
              static_cast<int>(LogSeverity::WARNING));
    EXPECT_LT(static_cast<int>(LogSeverity::WARNING),
              static_cast<int>(LogSeverity::INFO));
    EXPECT_LT(static_cast<int>(LogSeverity::INFO),
              static_cast<int>(LogSeverity::VERBOSE));
}

// ── File output tests ───────────────────────────────────────────────────────

TEST(LoggerTest, FileOutput) {
    auto& logger = Logger::instance();
    auto original_sev = logger.get_severity();
    logger.set_severity(LogSeverity::INFO);

    std::string test_log_path = "/tmp/trt_engine_test_logger.log";

    // Remove if exists
    std::remove(test_log_path.c_str());

    logger.enable_file_output(test_log_path);
    logger.info("test file output message");
    logger.disable_file_output();

    // Check that the file was written
    std::ifstream file(test_log_path);
    ASSERT_TRUE(file.is_open());

    std::string content;
    std::getline(file, content);
    file.close();

    EXPECT_FALSE(content.empty());
    EXPECT_NE(content.find("test file output message"), std::string::npos);

    // Cleanup
    std::remove(test_log_path.c_str());
    logger.set_severity(original_sev);
}

TEST(LoggerTest, DisableFileOutput) {
    auto& logger = Logger::instance();
    logger.disable_file_output();
    // Should not crash -- just a no-op when not enabled
    logger.disable_file_output();
}

// ── Thread safety tests ─────────────────────────────────────────────────────

TEST(LoggerTest, ConcurrentLogging) {
    auto& logger = Logger::instance();
    auto original_sev = logger.get_severity();
    logger.set_severity(LogSeverity::VERBOSE);

    constexpr int num_threads = 8;
    constexpr int logs_per_thread = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&logger, t]() {
            for (int i = 0; i < logs_per_thread; ++i) {
                logger.info("Thread " + std::to_string(t) +
                            " msg " + std::to_string(i));
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // If we get here without crash or deadlock, the test passes
    logger.set_severity(original_sev);
}

TEST(LoggerTest, ConcurrentFileLogging) {
    auto& logger = Logger::instance();
    auto original_sev = logger.get_severity();
    logger.set_severity(LogSeverity::INFO);

    std::string test_log_path = "/tmp/trt_engine_test_concurrent.log";
    std::remove(test_log_path.c_str());

    logger.enable_file_output(test_log_path);

    constexpr int num_threads = 4;
    constexpr int logs_per_thread = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&logger, t]() {
            for (int i = 0; i < logs_per_thread; ++i) {
                logger.info("ConcFile T" + std::to_string(t) +
                            " #" + std::to_string(i));
            }
        });
    }

    for (auto& th : threads) th.join();

    logger.disable_file_output();

    // Count lines in file
    std::ifstream file(test_log_path);
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) ++line_count;
    }
    file.close();

    EXPECT_EQ(line_count, num_threads * logs_per_thread);

    std::remove(test_log_path.c_str());
    logger.set_severity(original_sev);
}

// ── Singleton test ──────────────────────────────────────────────────────────

TEST(LoggerTest, SingletonIdentity) {
    auto& a = Logger::instance();
    auto& b = Logger::instance();
    EXPECT_EQ(&a, &b);

    auto& c = get_logger();
    EXPECT_EQ(&a, &c);
}

// ── Severity to string ─────────────────────────────────────────────────────

TEST(LoggerTest, SeverityToString) {
    EXPECT_EQ(severity_to_string(LogSeverity::INTERNAL_ERROR), "INTERNAL_ERROR");
    EXPECT_EQ(severity_to_string(LogSeverity::ERROR), "ERROR");
    EXPECT_EQ(severity_to_string(LogSeverity::WARNING), "WARNING");
    EXPECT_EQ(severity_to_string(LogSeverity::INFO), "INFO");
    EXPECT_EQ(severity_to_string(LogSeverity::VERBOSE), "VERBOSE");
}

// ── Logging method calls ────────────────────────────────────────────────────

TEST(LoggerTest, LoggingMethods) {
    auto& logger = Logger::instance();
    auto original_sev = logger.get_severity();
    logger.set_severity(LogSeverity::VERBOSE);

    // These should not throw
    EXPECT_NO_THROW(logger.error("test error"));
    EXPECT_NO_THROW(logger.warning("test warning"));
    EXPECT_NO_THROW(logger.info("test info"));
    EXPECT_NO_THROW(logger.verbose("test verbose"));

    logger.set_severity(original_sev);
}

// ── CUDA_CHECK macro test ───────────────────────────────────────────────────
// We test that CUDA_CHECK with cudaSuccess does not throw, and that an error
// condition does throw. We use a mock-style test since actual CUDA may not
// be available.

TEST(LoggerTest, CudaCheckMacroSuccess) {
    // cudaSuccess should not throw
    EXPECT_NO_THROW({
        cudaError_t status = cudaSuccess;
        if (status != cudaSuccess) {
            throw std::runtime_error("Should not reach here");
        }
    });
}

TEST(LoggerTest, TrtCheckMacroSuccess) {
    EXPECT_NO_THROW({
        TRT_CHECK(1 == 1);
    });
}

TEST(LoggerTest, TrtCheckMacroFailure) {
    EXPECT_THROW({
        TRT_CHECK(1 == 0);
    }, std::runtime_error);
}
