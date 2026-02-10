// test_builder.cpp - Unit tests for EngineBuilder
//
// Tests BuilderConfig creation, engine serialization/deserialization,
// and building from a sample ONNX model if available.

#include <gtest/gtest.h>
#include <trt_engine/builder.h>
#include <trt_engine/types.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── BuilderConfig tests ─────────────────────────────────────────────────────

TEST(BuilderConfigTest, DefaultValues) {
    BuilderConfig config;
    EXPECT_EQ(config.precision, Precision::FP32);
    EXPECT_EQ(config.max_workspace_size, 1ULL << 30);
    EXPECT_FALSE(config.enable_cuda_graph);
    EXPECT_FALSE(config.enable_dla);
    EXPECT_EQ(config.dla_core, 0);
    EXPECT_TRUE(config.timing_cache_path.empty());
    EXPECT_EQ(config.max_aux_streams, 0);
    EXPECT_FALSE(config.strongly_typed);
    EXPECT_EQ(config.builder_optimization_level, 3);
    EXPECT_TRUE(config.auto_timing_cache);
    EXPECT_TRUE(config.dynamic_shapes.empty());
}

TEST(BuilderConfigTest, SetPrecision) {
    BuilderConfig config;
    config.precision = Precision::FP16;
    EXPECT_EQ(config.precision, Precision::FP16);

    config.precision = Precision::INT8;
    EXPECT_EQ(config.precision, Precision::INT8);
}

TEST(BuilderConfigTest, DynamicShapes) {
    BuilderConfig config;
    DynamicShapeProfile profile;
    profile.name = "input";
    profile.min_dims = {1, 3, 224, 224};
    profile.opt_dims = {8, 3, 224, 224};
    profile.max_dims = {32, 3, 224, 224};

    config.dynamic_shapes.push_back(profile);
    EXPECT_EQ(config.dynamic_shapes.size(), 1u);
    EXPECT_EQ(config.dynamic_shapes[0].name, "input");
    EXPECT_EQ(config.dynamic_shapes[0].min_dims[0], 1);
}

TEST(BuilderConfigTest, WorkspaceSize) {
    BuilderConfig config;
    config.max_workspace_size = 2ULL << 30;  // 2GB
    EXPECT_EQ(config.max_workspace_size, 2ULL << 30);
}

// ── Engine serialization tests ──────────────────────────────────────────────

TEST(EngineBuilderTest, SaveEngineToFile) {
    std::string test_path = "/tmp/trt_engine_test_save.engine";
    std::vector<char> dummy_data = {'T', 'R', 'T', '_', 'T', 'E', 'S', 'T'};

    bool saved = EngineBuilder::save_engine(dummy_data, test_path);
    EXPECT_TRUE(saved);

    // Verify file exists and has correct size
    EXPECT_TRUE(fs::exists(test_path));
    EXPECT_EQ(fs::file_size(test_path), dummy_data.size());

    std::remove(test_path.c_str());
}

TEST(EngineBuilderTest, LoadEngineFromFile) {
    std::string test_path = "/tmp/trt_engine_test_load.engine";
    std::vector<char> original_data = {'D', 'A', 'T', 'A', '_', '1', '2', '3'};

    // Write test file
    {
        std::ofstream f(test_path, std::ios::binary);
        f.write(original_data.data(),
                static_cast<std::streamsize>(original_data.size()));
    }

    auto loaded = EngineBuilder::load_engine(test_path);
    EXPECT_EQ(loaded.size(), original_data.size());
    EXPECT_EQ(loaded, original_data);

    std::remove(test_path.c_str());
}

TEST(EngineBuilderTest, SaveAndLoadRoundTrip) {
    std::string test_path = "/tmp/trt_engine_test_roundtrip.engine";

    std::vector<char> data(1024);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<char>(i & 0xFF);
    }

    EXPECT_TRUE(EngineBuilder::save_engine(data, test_path));

    auto loaded = EngineBuilder::load_engine(test_path);
    EXPECT_EQ(data, loaded);

    std::remove(test_path.c_str());
}

TEST(EngineBuilderTest, LoadNonExistentFile) {
    auto data = EngineBuilder::load_engine("/tmp/nonexistent_engine_file_xyz.engine");
    EXPECT_TRUE(data.empty());
}

TEST(EngineBuilderTest, SaveToInvalidPath) {
    std::vector<char> data = {'t', 'e', 's', 't'};
    // Try to save to an invalid directory
    bool saved = EngineBuilder::save_engine(data,
        "/nonexistent_dir_xyz/test.engine");
    EXPECT_FALSE(saved);
}

// ── Engine builder construction ─────────────────────────────────────────────

TEST(EngineBuilderTest, Construction) {
    auto& logger = get_logger();
    EXPECT_NO_THROW({
        EngineBuilder builder(logger);
    });
}

// ── Build from ONNX (only if test model is available) ───────────────────────

TEST(EngineBuilderTest, BuildFromONNX_IfAvailable) {
    // Look for a test ONNX model in common locations
    std::vector<std::string> model_paths = {
        "/tmp/test_model.onnx",
        "test_data/test_model.onnx",
        "../test_data/test_model.onnx",
    };

    std::string model_path;
    for (const auto& p : model_paths) {
        if (fs::exists(p)) {
            model_path = p;
            break;
        }
    }

    if (model_path.empty()) {
        GTEST_SKIP() << "No test ONNX model found";
    }

    // Check CUDA is available
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto& logger = get_logger();
    EngineBuilder builder(logger);

    BuilderConfig config;
    config.precision = Precision::FP32;

    auto engine_data = builder.build_engine(model_path, config);
    EXPECT_FALSE(engine_data.empty());
}

// ── DynamicShapeProfile tests ───────────────────────────────────────────────

TEST(DynamicShapeProfileTest, DefaultConstruction) {
    DynamicShapeProfile profile;
    EXPECT_TRUE(profile.name.empty());
    EXPECT_TRUE(profile.min_dims.empty());
    EXPECT_TRUE(profile.opt_dims.empty());
    EXPECT_TRUE(profile.max_dims.empty());
}

TEST(DynamicShapeProfileTest, SetValues) {
    DynamicShapeProfile profile;
    profile.name = "images";
    profile.min_dims = {1, 3, 224, 224};
    profile.opt_dims = {4, 3, 224, 224};
    profile.max_dims = {16, 3, 224, 224};

    EXPECT_EQ(profile.name, "images");
    EXPECT_EQ(profile.min_dims.size(), 4u);
    EXPECT_EQ(profile.opt_dims.size(), 4u);
    EXPECT_EQ(profile.max_dims.size(), 4u);
}
