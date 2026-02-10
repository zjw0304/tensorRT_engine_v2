#include <trt_engine/model_converter.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace trt_engine {

// ── Helpers ─────────────────────────────────────────────────────────────────

std::string ModelConverter::get_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

int ModelConverter::run_command(const std::string& cmd, std::string* output) {
    get_logger().verbose("Running command: " + cmd);

    if (output) {
        output->clear();
        std::string full_cmd = cmd + " 2>&1";
        FILE* pipe = popen(full_cmd.c_str(), "r");
        if (!pipe) {
            get_logger().error("Failed to open subprocess pipe");
            return -1;
        }
        std::array<char, 4096> buf;
        while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
            output->append(buf.data());
        }
        int status = pclose(pipe);
#ifdef _WIN32
        return status;
#else
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    }

    int ret = std::system(cmd.c_str());
#ifdef _WIN32
    return ret;
#else
    return WIFEXITED(ret) ? WEXITSTATUS(ret) : -1;
#endif
}

// ── Format detection ────────────────────────────────────────────────────────

ModelFormat ModelConverter::detect_format(const std::string& path) {
    std::string ext = get_extension(path);

    if (ext == ".onnx") {
        return ModelFormat::ONNX;
    }
    if (ext == ".pb" || ext == ".savedmodel") {
        return ModelFormat::TENSORFLOW;
    }
    // TensorFlow SavedModel is typically a directory
    if (fs::is_directory(path)) {
        // Check for saved_model.pb inside the directory
        if (fs::exists(fs::path(path) / "saved_model.pb")) {
            return ModelFormat::TENSORFLOW;
        }
    }
    if (ext == ".pt" || ext == ".pth" || ext == ".torchscript") {
        return ModelFormat::PYTORCH;
    }
    if (ext == ".engine" || ext == ".plan" || ext == ".trt") {
        return ModelFormat::TENSORRT_ENGINE;
    }

    return ModelFormat::UNKNOWN;
}

// ── ONNX validation ────────────────────────────────────────────────────────

bool ModelConverter::validate_onnx(const std::string& path) {
    if (!fs::exists(path)) {
        get_logger().error("ONNX file does not exist: " + path);
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        get_logger().error("Failed to open ONNX file: " + path);
        return false;
    }

    // Read the first few bytes to check for protobuf / ONNX magic.
    // ONNX files are protobuf-serialized; the first field tag is
    // typically 0x08 (field 1, varint).  We also accept any non-empty
    // binary file since a full ONNX parse happens later in the builder.
    char header[8] = {};
    file.read(header, sizeof(header));
    std::streamsize bytes_read = file.gcount();
    if (bytes_read < 4) {
        get_logger().error("ONNX file too small: " + path);
        return false;
    }

    get_logger().info("ONNX file validated: " + path +
                      " (" + std::to_string(fs::file_size(path)) + " bytes)");
    return true;
}

// ── ONNX optimisation (constant-folding via Python/onnx) ────────────────────

bool ModelConverter::optimize_onnx(const std::string& input_path,
                                    const std::string& output_path) {
    get_logger().info("Optimizing ONNX model: " + input_path + " -> " + output_path);

    // Build a one-liner Python script that loads the model, runs shape
    // inference and constant-folding via onnxruntime or plain onnx, then
    // saves the result.
    std::ostringstream py;
    py << "python3 -c \""
       << "import onnx; "
       << "from onnx import shape_inference; "
       << "model = onnx.load('" << input_path << "'); "
       << "model = shape_inference.infer_shapes(model); "
       << "try:\\n"
       << "    from onnxsim import simplify\\n"
       << "    model, ok = simplify(model)\\n"
       << "    if not ok: print('onnxsim simplify returned False')\\n"
       << "except ImportError:\\n"
       << "    print('onnxsim not available, skipping simplification')\\n"
       << "; "
       << "onnx.save(model, '" << output_path << "'); "
       << "print('Optimization complete')\"";

    std::string output;
    int ret = run_command(py.str(), &output);
    if (ret != 0) {
        get_logger().error("ONNX optimization failed (exit=" +
                           std::to_string(ret) + "): " + output);
        return false;
    }

    get_logger().info("ONNX optimization succeeded: " + output_path);
    return true;
}

// ── TensorFlow → ONNX ──────────────────────────────────────────────────────

bool ModelConverter::convert_tensorflow_to_onnx(const std::string& input_path,
                                                 const std::string& output_path) {
    get_logger().info("Converting TensorFlow model to ONNX: " + input_path);

    // Determine if we have a SavedModel directory or a frozen graph (.pb).
    std::string cmd;
    if (fs::is_directory(input_path)) {
        // SavedModel directory
        cmd = "python3 -m tf2onnx.convert"
              " --saved-model \"" + input_path + "\""
              " --output \"" + output_path + "\""
              " --opset 13";
    } else {
        // Frozen graph .pb
        cmd = "python3 -m tf2onnx.convert"
              " --graphdef \"" + input_path + "\""
              " --output \"" + output_path + "\""
              " --opset 13";
    }

    std::string output;
    int ret = run_command(cmd, &output);
    if (ret != 0) {
        get_logger().error("tf2onnx conversion failed (exit=" +
                           std::to_string(ret) + "): " + output);
        return false;
    }

    get_logger().info("TensorFlow → ONNX conversion succeeded: " + output_path);
    return true;
}

// ── PyTorch → ONNX ─────────────────────────────────────────────────────────

bool ModelConverter::convert_pytorch_to_onnx(const std::string& input_path,
                                              const std::string& output_path) {
    get_logger().info("Converting PyTorch model to ONNX: " + input_path);

    // PyTorch TorchScript (.pt / .torchscript) models can be loaded and
    // exported via a small Python script.
    std::ostringstream py;
    py << "python3 -c \""
       << "import torch; "
       << "model = torch.jit.load('" << input_path << "'); "
       << "model.eval(); "
       << "# Attempt to derive input shape from the first parameter\\n"
       << "params = list(model.parameters())\\n"
       << "if len(params) > 0:\\n"
       << "    in_features = params[0].shape[1] if params[0].dim() >= 2 else params[0].shape[0]\\n"
       << "    dummy = torch.randn(1, in_features)\\n"
       << "else:\\n"
       << "    dummy = torch.randn(1, 3, 224, 224)\\n"
       << "; "
       << "torch.onnx.export(model, dummy, '" << output_path << "', "
       << "opset_version=13, "
       << "do_constant_folding=True, "
       << "input_names=['input'], "
       << "output_names=['output'], "
       << "dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}); "
       << "print('Export complete')\"";

    std::string output;
    int ret = run_command(py.str(), &output);
    if (ret != 0) {
        get_logger().error("PyTorch → ONNX conversion failed (exit=" +
                           std::to_string(ret) + "): " + output);
        return false;
    }

    get_logger().info("PyTorch → ONNX conversion succeeded: " + output_path);
    return true;
}

// ── Generic convert entry point ─────────────────────────────────────────────

bool ModelConverter::convert(const std::string& input_path,
                              const std::string& output_path) {
    ModelFormat fmt = detect_format(input_path);
    get_logger().info("Detected model format: " + model_format_to_string(fmt));

    switch (fmt) {
        case ModelFormat::ONNX:
            // Already ONNX – validate the source and copy if needed.
            if (!validate_onnx(input_path)) {
                return false;
            }
            if (input_path != output_path) {
                try {
                    fs::copy_file(input_path, output_path,
                                  fs::copy_options::overwrite_existing);
                } catch (const fs::filesystem_error& e) {
                    get_logger().error("Failed to copy ONNX file: " +
                                       std::string(e.what()));
                    return false;
                }
            }
            return true;

        case ModelFormat::TENSORFLOW:
            return convert_tensorflow_to_onnx(input_path, output_path);

        case ModelFormat::PYTORCH:
            return convert_pytorch_to_onnx(input_path, output_path);

        case ModelFormat::TENSORRT_ENGINE:
            get_logger().error("Cannot convert a TensorRT engine to ONNX. "
                               "Please provide the original model.");
            return false;

        case ModelFormat::UNKNOWN:
        default:
            get_logger().error("Unknown model format: " + input_path);
            return false;
    }
}

}  // namespace trt_engine
