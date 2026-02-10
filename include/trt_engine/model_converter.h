#pragma once

#include <trt_engine/logger.h>
#include <trt_engine/types.h>

#include <string>

namespace trt_engine {

// ── Supported model formats ─────────────────────────────────────────────────
enum class ModelFormat {
    ONNX,
    TENSORFLOW,
    PYTORCH,
    TENSORRT_ENGINE,
    UNKNOWN
};

inline std::string model_format_to_string(ModelFormat fmt) {
    switch (fmt) {
        case ModelFormat::ONNX:            return "ONNX";
        case ModelFormat::TENSORFLOW:      return "TensorFlow";
        case ModelFormat::PYTORCH:         return "PyTorch";
        case ModelFormat::TENSORRT_ENGINE: return "TensorRT Engine";
        case ModelFormat::UNKNOWN:         return "Unknown";
    }
    return "Unknown";
}

// ── Model converter ─────────────────────────────────────────────────────────
//
// Converts models from various frameworks to ONNX format suitable for
// TensorRT engine building.  TensorFlow and PyTorch conversions invoke
// external Python processes (tf2onnx, torch.onnx.export) via subprocess.
//
class ModelConverter {
public:
    // Detect the format of a model file from its extension / magic bytes.
    static ModelFormat detect_format(const std::string& path);

    // Convert any supported format to ONNX.
    // If the model is already ONNX it is simply validated.
    // Returns true on success.
    static bool convert(const std::string& input_path,
                        const std::string& output_path);

    // Validate that an ONNX file exists and can be opened.
    static bool validate_onnx(const std::string& path);

    // Run constant-folding / shape-inference on an ONNX model via
    // the onnxruntime / onnx Python packages.
    static bool optimize_onnx(const std::string& input_path,
                              const std::string& output_path);

    // Framework-specific converters
    static bool convert_tensorflow_to_onnx(const std::string& input_path,
                                           const std::string& output_path);

    static bool convert_pytorch_to_onnx(const std::string& input_path,
                                        const std::string& output_path);

private:
    // Execute a shell command and return the exit code.
    // stdout + stderr are captured into `output` when non-null.
    static int run_command(const std::string& cmd, std::string* output = nullptr);

    // Return the file extension in lower-case.
    static std::string get_extension(const std::string& path);
};

}  // namespace trt_engine
