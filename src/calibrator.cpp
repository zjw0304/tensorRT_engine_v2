#include <trt_engine/calibrator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace trt_engine {

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: compute total element count from a dims vector.
// ─────────────────────────────────────────────────────────────────────────────
static size_t volume(const std::vector<int>& dims) {
    size_t vol = 1;
    for (int d : dims) {
        vol *= static_cast<size_t>(d);
    }
    return vol;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: collect .bin and .raw files sorted by name.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<std::string> collect_files(const std::string& dir) {
    std::vector<std::string> files;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return files;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        // Compare lower-case
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".bin" || ext == ".raw") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: read raw bytes from a file into dst.
// ─────────────────────────────────────────────────────────────────────────────
static bool read_raw_file(const std::string& path, void* dst, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        get_logger().error("Calibrator: cannot open data file: " + path);
        return false;
    }
    file.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
    if (static_cast<size_t>(file.gcount()) != size) {
        get_logger().warning("Calibrator: data file " + path +
                             " has fewer bytes than expected (" +
                             std::to_string(file.gcount()) + " vs " +
                             std::to_string(size) + ")");
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════════
//  EntropyCalibratorV2
// ═════════════════════════════════════════════════════════════════════════════

EntropyCalibratorV2::EntropyCalibratorV2(
        const std::string& data_dir,
        int batch_size,
        const std::string& input_name,
        const std::vector<int>& input_dims,
        const std::string& cache_file)
    : data_dir_(data_dir)
    , batch_size_(batch_size)
    , input_name_(input_name)
    , input_dims_(input_dims)
    , cache_file_(cache_file) {

    // Compute the byte size of a single input sample (float32).
    single_input_size_ = volume(input_dims_) * sizeof(float);

    // Pre-allocate device buffer for a full batch.
    size_t batch_bytes = static_cast<size_t>(batch_size_) * single_input_size_;
    device_buffer_.allocate(batch_bytes);

    collect_data_files();

    get_logger().info("EntropyCalibratorV2: " +
                      std::to_string(data_files_.size()) + " data files, "
                      "batch_size=" + std::to_string(batch_size_) +
                      ", input=" + input_name_ +
                      ", sample_bytes=" + std::to_string(single_input_size_));
}

void EntropyCalibratorV2::collect_data_files() {
    data_files_ = collect_files(data_dir_);
    if (data_files_.empty()) {
        get_logger().warning("EntropyCalibratorV2: no .bin/.raw files found in " +
                             data_dir_);
    }
}

int EntropyCalibratorV2::getBatchSize() const noexcept {
    return batch_size_;
}

bool EntropyCalibratorV2::getBatch(void* bindings[],
                                    const char* names[],
                                    int nbBindings) noexcept {
    // Find the binding index that matches our input name.
    int binding_idx = -1;
    for (int i = 0; i < nbBindings; ++i) {
        if (names[i] && std::string(names[i]) == input_name_) {
            binding_idx = i;
            break;
        }
    }
    if (binding_idx < 0) {
        // Fallback: use binding 0
        binding_idx = 0;
    }

    size_t start = static_cast<size_t>(current_batch_) *
                   static_cast<size_t>(batch_size_);
    if (start >= data_files_.size()) {
        // All batches consumed.
        current_batch_ = 0;
        return false;
    }

    size_t batch_bytes = static_cast<size_t>(batch_size_) * single_input_size_;

    // Read files into a staging host buffer then upload.
    std::vector<char> host_buf(batch_bytes, 0);

    for (int i = 0; i < batch_size_; ++i) {
        size_t file_idx = start + static_cast<size_t>(i);
        if (file_idx >= data_files_.size()) break;
        read_raw_file(data_files_[file_idx],
                      host_buf.data() + static_cast<size_t>(i) * single_input_size_,
                      single_input_size_);
    }

    cudaError_t err = cudaMemcpy(device_buffer_.data(), host_buf.data(),
                                  batch_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        get_logger().error("EntropyCalibratorV2: cudaMemcpy failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }

    bindings[binding_idx] = device_buffer_.data();
    ++current_batch_;
    return true;
}

const void* EntropyCalibratorV2::readCalibrationCache(size_t& length) noexcept {
    calibration_cache_.clear();
    if (cache_file_.empty()) {
        length = 0;
        return nullptr;
    }

    std::ifstream file(cache_file_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        length = 0;
        return nullptr;
    }

    auto size = file.tellg();
    if (size <= 0) {
        length = 0;
        return nullptr;
    }
    file.seekg(0, std::ios::beg);
    calibration_cache_.resize(static_cast<size_t>(size));
    file.read(calibration_cache_.data(), size);

    length = calibration_cache_.size();
    get_logger().info("EntropyCalibratorV2: read calibration cache (" +
                      std::to_string(length) + " bytes) from " + cache_file_);
    return calibration_cache_.data();
}

void EntropyCalibratorV2::writeCalibrationCache(const void* cache,
                                                 size_t length) noexcept {
    if (cache_file_.empty()) return;

    std::ofstream file(cache_file_, std::ios::binary);
    if (!file.is_open()) {
        get_logger().error("EntropyCalibratorV2: failed to write calibration "
                           "cache to " + cache_file_);
        return;
    }
    file.write(static_cast<const char*>(cache),
               static_cast<std::streamsize>(length));
    get_logger().info("EntropyCalibratorV2: calibration cache written (" +
                      std::to_string(length) + " bytes) to " + cache_file_);
}

bool EntropyCalibratorV2::read_data_file(const std::string& path,
                                          void* dst, size_t size) {
    return read_raw_file(path, dst, size);
}

// ═════════════════════════════════════════════════════════════════════════════
//  MinMaxCalibrator
// ═════════════════════════════════════════════════════════════════════════════

MinMaxCalibrator::MinMaxCalibrator(
        const std::string& data_dir,
        int batch_size,
        const std::string& input_name,
        const std::vector<int>& input_dims,
        const std::string& cache_file)
    : data_dir_(data_dir)
    , batch_size_(batch_size)
    , input_name_(input_name)
    , input_dims_(input_dims)
    , cache_file_(cache_file) {

    single_input_size_ = volume(input_dims_) * sizeof(float);

    size_t batch_bytes = static_cast<size_t>(batch_size_) * single_input_size_;
    device_buffer_.allocate(batch_bytes);

    collect_data_files();

    get_logger().info("MinMaxCalibrator: " +
                      std::to_string(data_files_.size()) + " data files, "
                      "batch_size=" + std::to_string(batch_size_) +
                      ", input=" + input_name_ +
                      ", sample_bytes=" + std::to_string(single_input_size_));
}

void MinMaxCalibrator::collect_data_files() {
    data_files_ = collect_files(data_dir_);
    if (data_files_.empty()) {
        get_logger().warning("MinMaxCalibrator: no .bin/.raw files found in " +
                             data_dir_);
    }
}

int MinMaxCalibrator::getBatchSize() const noexcept {
    return batch_size_;
}

bool MinMaxCalibrator::getBatch(void* bindings[],
                                 const char* names[],
                                 int nbBindings) noexcept {
    int binding_idx = -1;
    for (int i = 0; i < nbBindings; ++i) {
        if (names[i] && std::string(names[i]) == input_name_) {
            binding_idx = i;
            break;
        }
    }
    if (binding_idx < 0) {
        binding_idx = 0;
    }

    size_t start = static_cast<size_t>(current_batch_) *
                   static_cast<size_t>(batch_size_);
    if (start >= data_files_.size()) {
        current_batch_ = 0;
        return false;
    }

    size_t batch_bytes = static_cast<size_t>(batch_size_) * single_input_size_;
    std::vector<char> host_buf(batch_bytes, 0);

    for (int i = 0; i < batch_size_; ++i) {
        size_t file_idx = start + static_cast<size_t>(i);
        if (file_idx >= data_files_.size()) break;
        read_raw_file(data_files_[file_idx],
                      host_buf.data() + static_cast<size_t>(i) * single_input_size_,
                      single_input_size_);
    }

    cudaError_t err = cudaMemcpy(device_buffer_.data(), host_buf.data(),
                                  batch_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        get_logger().error("MinMaxCalibrator: cudaMemcpy failed: " +
                           std::string(cudaGetErrorString(err)));
        return false;
    }

    bindings[binding_idx] = device_buffer_.data();
    ++current_batch_;
    return true;
}

const void* MinMaxCalibrator::readCalibrationCache(size_t& length) noexcept {
    calibration_cache_.clear();
    if (cache_file_.empty()) {
        length = 0;
        return nullptr;
    }

    std::ifstream file(cache_file_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        length = 0;
        return nullptr;
    }

    auto size = file.tellg();
    if (size <= 0) {
        length = 0;
        return nullptr;
    }
    file.seekg(0, std::ios::beg);
    calibration_cache_.resize(static_cast<size_t>(size));
    file.read(calibration_cache_.data(), size);

    length = calibration_cache_.size();
    get_logger().info("MinMaxCalibrator: read calibration cache (" +
                      std::to_string(length) + " bytes) from " + cache_file_);
    return calibration_cache_.data();
}

void MinMaxCalibrator::writeCalibrationCache(const void* cache,
                                              size_t length) noexcept {
    if (cache_file_.empty()) return;

    std::ofstream file(cache_file_, std::ios::binary);
    if (!file.is_open()) {
        get_logger().error("MinMaxCalibrator: failed to write calibration "
                           "cache to " + cache_file_);
        return;
    }
    file.write(static_cast<const char*>(cache),
               static_cast<std::streamsize>(length));
    get_logger().info("MinMaxCalibrator: calibration cache written (" +
                      std::to_string(length) + " bytes) to " + cache_file_);
}

bool MinMaxCalibrator::read_data_file(const std::string& path,
                                       void* dst, size_t size) {
    return read_raw_file(path, dst, size);
}

}  // namespace trt_engine
