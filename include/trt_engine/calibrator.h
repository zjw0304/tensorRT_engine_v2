#pragma once

#include <NvInfer.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>
#include <trt_engine/types.h>

#include <memory>
#include <string>
#include <vector>

namespace trt_engine {

// ── EntropyCalibratorV2 ─────────────────────────────────────────────────────
//
// INT8 calibrator using the Entropy v2 algorithm.  Reads batches of raw
// binary data (*.bin or *.raw) from a directory and feeds them to TensorRT
// for calibration.  A calibration cache is written on the first run and
// re-read on subsequent runs to avoid re-calibration.
//
class EntropyCalibratorV2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    EntropyCalibratorV2(const std::string& data_dir,
                        int batch_size,
                        const std::string& input_name,
                        const std::vector<int>& input_dims,
                        const std::string& cache_file);

    ~EntropyCalibratorV2() override = default;

    // nvinfer1::IInt8EntropyCalibrator2 interface
    int         getBatchSize() const noexcept override;
    bool        getBatch(void* bindings[], const char* names[],
                         int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void        writeCalibrationCache(const void* cache,
                                      size_t length) noexcept override;

private:
    void collect_data_files();
    bool read_data_file(const std::string& path, void* dst, size_t size);

    std::string          data_dir_;
    int                  batch_size_;
    std::string          input_name_;
    std::vector<int>     input_dims_;
    std::string          cache_file_;

    size_t               single_input_size_ = 0;   // bytes per single input
    int                  current_batch_     = 0;
    std::vector<std::string> data_files_;
    DeviceBuffer         device_buffer_;
    std::vector<char>    calibration_cache_;
};

// ── MinMaxCalibrator ────────────────────────────────────────────────────────
//
// INT8 calibrator using the MinMax algorithm.  Same data pipeline as
// EntropyCalibratorV2 but uses min/max range for scale computation.
// Recommended for NLP / transformer-based models.
//
class MinMaxCalibrator : public nvinfer1::IInt8MinMaxCalibrator {
public:
    MinMaxCalibrator(const std::string& data_dir,
                     int batch_size,
                     const std::string& input_name,
                     const std::vector<int>& input_dims,
                     const std::string& cache_file);

    ~MinMaxCalibrator() override = default;

    // nvinfer1::IInt8MinMaxCalibrator interface
    int         getBatchSize() const noexcept override;
    bool        getBatch(void* bindings[], const char* names[],
                         int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void        writeCalibrationCache(const void* cache,
                                      size_t length) noexcept override;

private:
    void collect_data_files();
    bool read_data_file(const std::string& path, void* dst, size_t size);

    std::string          data_dir_;
    int                  batch_size_;
    std::string          input_name_;
    std::vector<int>     input_dims_;
    std::string          cache_file_;

    size_t               single_input_size_ = 0;
    int                  current_batch_     = 0;
    std::vector<std::string> data_files_;
    DeviceBuffer         device_buffer_;
    std::vector<char>    calibration_cache_;
};

}  // namespace trt_engine
