// test_nlp_models.cpp - Unit tests for NLP model inference with TensorRT
//
// Tests engine building, I/O info queries, and inference for BERT, DistilBERT,
// GPT-2, and T5-small models.  All tests skip gracefully if CUDA or the
// required ONNX model is unavailable.

#include <gtest/gtest.h>
#include <trt_engine/trt_engine.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace trt_engine;

// ── Helpers ─────────────────────────────────────────────────────────────────

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

static std::string find_models_dir() {
    for (const auto& p : {"models", "../models",
                          "/mnt/data2/tensorRT_engine_v2/models"}) {
        if (fs::exists(p) && fs::is_directory(p)) return p;
    }
    return "";
}

static bool model_exists(const std::string& models_dir,
                          const std::string& model_name) {
    return fs::exists(models_dir + "/" + model_name + "/model.onnx");
}

// Build (or load a cached) TensorRT engine from an ONNX file.
// Returns empty string on failure.
static std::string build_or_load_engine(
        const std::string& onnx_path,
        const std::string& engine_path,
        const std::vector<DynamicShapeProfile>& profiles,
        Precision precision = Precision::FP16) {
    if (fs::exists(engine_path)) return engine_path;

    auto& logger = get_logger();
    EngineBuilder builder(logger);

    BuilderConfig config;
    config.precision = precision;
    config.max_workspace_size = 1ULL << 30;
    config.dynamic_shapes = profiles;

    auto engine_data = builder.build_engine(onnx_path, config);
    if (engine_data.empty()) return "";

    EngineBuilder::save_engine(engine_data, engine_path);
    return engine_path;
}

// Pack a vector of int64_t values into a float vector whose raw bytes
// match the int64 representation.  The returned vector has
//   size = num_elements * sizeof(int64_t) / sizeof(float) = num_elements * 2
static std::vector<float> create_int64_as_float(
        const std::vector<int64_t>& int_data) {
    size_t float_count =
        (int_data.size() * sizeof(int64_t)) / sizeof(float);
    std::vector<float> result(float_count);
    std::memcpy(result.data(), int_data.data(),
                int_data.size() * sizeof(int64_t));
    return result;
}

// Create random input_ids in [1, vocab_size).
static std::vector<int64_t> create_input_ids(int batch, int seq_len,
                                              int vocab_size = 30522) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(1, vocab_size - 1);
    std::vector<int64_t> ids(batch * seq_len);
    for (auto& id : ids) id = dist(rng);
    return ids;
}

// Create an all-ones attention mask.
static std::vector<int64_t> create_attention_mask(int batch, int seq_len) {
    return std::vector<int64_t>(batch * seq_len, 1);
}

// Create an all-zeros token_type_ids mask.
static std::vector<int64_t> create_token_type_ids(int batch, int seq_len) {
    return std::vector<int64_t>(batch * seq_len, 0);
}

// Build standard NLP dynamic-shape profiles (input_ids + attention_mask, and
// optionally token_type_ids).
// When batch_dynamic is false, the batch dimension is fixed to 1 (for models
// exported without a dynamic batch axis).
static std::vector<DynamicShapeProfile> make_nlp_profiles(
        bool include_token_type_ids, bool batch_dynamic = true) {
    std::vector<DynamicShapeProfile> profiles;

    DynamicShapeProfile p;
    if (batch_dynamic) {
        p.min_dims = {1, 32};
        p.opt_dims = {4, 128};
        p.max_dims = {32, 512};
    } else {
        p.min_dims = {1, 32};
        p.opt_dims = {1, 128};
        p.max_dims = {1, 512};
    }

    p.name = "input_ids";
    profiles.push_back(p);

    p.name = "attention_mask";
    profiles.push_back(p);

    if (include_token_type_ids) {
        p.name = "token_type_ids";
        profiles.push_back(p);
    }

    return profiles;
}

// ── BERT fixture ────────────────────────────────────────────────────────────

class BERTModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        models_dir_ = find_models_dir();
        if (models_dir_.empty() || !model_exists(models_dir_, "bert-base"))
            GTEST_SKIP() << "BERT model not found";
        onnx_path_ = models_dir_ + "/bert-base/model.onnx";
        engine_path_ = models_dir_ + "/bert-base/model_fp16.engine";
    }

    std::string models_dir_;
    std::string onnx_path_;
    std::string engine_path_;
};

TEST_F(BERTModelTest, BuildEngine) {
    auto profiles = make_nlp_profiles(/*include_token_type_ids=*/true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty()) << "Failed to build BERT engine";
    EXPECT_TRUE(fs::exists(path));
}

TEST_F(BERTModelTest, GetIOInfo) {
    auto profiles = make_nlp_profiles(true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    ASSERT_NE(engine, nullptr);

    auto inputs = engine->get_input_info();
    EXPECT_GE(inputs.size(), 2u);

    // Verify expected input names are present.
    auto has_name = [&](const std::string& name) {
        return std::any_of(inputs.begin(), inputs.end(),
                           [&](const TensorInfo& t) { return t.name == name; });
    };
    EXPECT_TRUE(has_name("input_ids"));
    EXPECT_TRUE(has_name("attention_mask"));

    auto outputs = engine->get_output_info();
    EXPECT_GT(outputs.size(), 0u);
}

TEST_F(BERTModelTest, Inference_Batch1_Seq128) {
    auto profiles = make_nlp_profiles(true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    EngineConfig ecfg;
    auto engine = InferenceEngine::create(path, ecfg);
    ASSERT_NE(engine, nullptr);

    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});
    engine->set_input_shape("token_type_ids",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));
    auto ttid = create_int64_as_float(create_token_type_ids(batch, seq_len));

    auto result = engine->infer({ids, mask, ttid});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
    EXPECT_GT(result.latency_ms, 0.0f);
}

TEST_F(BERTModelTest, Inference_Batch8_Seq128) {
    auto profiles = make_nlp_profiles(true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 8, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});
    engine->set_input_shape("token_type_ids",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));
    auto ttid = create_int64_as_float(create_token_type_ids(batch, seq_len));

    auto result = engine->infer({ids, mask, ttid});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(BERTModelTest, Inference_VaryingSeqLen) {
    auto profiles = make_nlp_profiles(true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1;

    for (int seq_len : {32, 64, 128, 256}) {
        engine->set_input_shape("input_ids",       {batch, seq_len});
        engine->set_input_shape("attention_mask",   {batch, seq_len});
        engine->set_input_shape("token_type_ids",   {batch, seq_len});

        auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
        auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));
        auto ttid = create_int64_as_float(create_token_type_ids(batch, seq_len));

        auto result = engine->infer({ids, mask, ttid});
        EXPECT_TRUE(result.success)
            << "seq_len=" << seq_len << ": " << result.error_msg;
        EXPECT_GT(result.outputs.size(), 0u);
    }
}

TEST_F(BERTModelTest, AsyncInference) {
    auto profiles = make_nlp_profiles(true);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});
    engine->set_input_shape("token_type_ids",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));
    auto ttid = create_int64_as_float(create_token_type_ids(batch, seq_len));

    auto future = engine->infer_async({ids, mask, ttid});
    auto result = future.get();
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(BERTModelTest, FP16vsFP32) {
    auto profiles = make_nlp_profiles(true);

    std::string fp32_engine = models_dir_ + "/bert-base/model_fp32.engine";
    auto path_fp32 = build_or_load_engine(
        onnx_path_, fp32_engine, profiles, Precision::FP32);
    auto path_fp16 = build_or_load_engine(
        onnx_path_, engine_path_, profiles, Precision::FP16);

    if (path_fp32.empty() || path_fp16.empty())
        GTEST_SKIP() << "Failed to build one of the engines";

    auto engine_fp32 = InferenceEngine::create(path_fp32);
    auto engine_fp16 = InferenceEngine::create(path_fp16);

    const int batch = 1, seq_len = 64;
    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));
    auto ttid = create_int64_as_float(create_token_type_ids(batch, seq_len));

    for (auto* eng : {engine_fp32.get(), engine_fp16.get()}) {
        eng->set_input_shape("input_ids",       {batch, seq_len});
        eng->set_input_shape("attention_mask",   {batch, seq_len});
        eng->set_input_shape("token_type_ids",   {batch, seq_len});
    }

    auto r32 = engine_fp32->infer({ids, mask, ttid});
    auto r16 = engine_fp16->infer({ids, mask, ttid});

    ASSERT_TRUE(r32.success) << r32.error_msg;
    ASSERT_TRUE(r16.success) << r16.error_msg;
    ASSERT_EQ(r32.outputs.size(), r16.outputs.size());

    // Outputs should be close but not necessarily identical.
    for (size_t i = 0; i < r32.outputs.size(); ++i) {
        ASSERT_EQ(r32.outputs[i].size(), r16.outputs[i].size());
        for (size_t j = 0; j < r32.outputs[i].size(); ++j) {
            EXPECT_NEAR(r32.outputs[i][j], r16.outputs[i][j], 0.05f)
                << "output[" << i << "][" << j << "]";
        }
    }
}

// ── DistilBERT fixture ──────────────────────────────────────────────────────

class DistilBERTModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        models_dir_ = find_models_dir();
        if (models_dir_.empty() || !model_exists(models_dir_, "distilbert"))
            GTEST_SKIP() << "DistilBERT model not found";
        onnx_path_ = models_dir_ + "/distilbert/model.onnx";
        engine_path_ = models_dir_ + "/distilbert/model_fp16.engine";
    }

    std::string models_dir_;
    std::string onnx_path_;
    std::string engine_path_;
};

TEST_F(DistilBERTModelTest, BuildEngine) {
    auto profiles = make_nlp_profiles(/*include_token_type_ids=*/false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty()) << "Failed to build DistilBERT engine";
    EXPECT_TRUE(fs::exists(path));
}

TEST_F(DistilBERTModelTest, GetIOInfo) {
    auto profiles = make_nlp_profiles(false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    auto inputs = engine->get_input_info();
    EXPECT_GE(inputs.size(), 2u);

    auto has_name = [&](const std::string& name) {
        return std::any_of(inputs.begin(), inputs.end(),
                           [&](const TensorInfo& t) { return t.name == name; });
    };
    EXPECT_TRUE(has_name("input_ids"));
    EXPECT_TRUE(has_name("attention_mask"));
}

TEST_F(DistilBERTModelTest, Inference_Batch1_Seq128) {
    auto profiles = make_nlp_profiles(false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

    auto result = engine->infer({ids, mask});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(DistilBERTModelTest, Inference_Batch8_Seq128) {
    auto profiles = make_nlp_profiles(false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 8, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

    auto result = engine->infer({ids, mask});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(DistilBERTModelTest, Inference_VaryingSeqLen) {
    auto profiles = make_nlp_profiles(false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1;

    for (int seq_len : {32, 64, 128, 256}) {
        engine->set_input_shape("input_ids",       {batch, seq_len});
        engine->set_input_shape("attention_mask",   {batch, seq_len});

        auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
        auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

        auto result = engine->infer({ids, mask});
        EXPECT_TRUE(result.success)
            << "seq_len=" << seq_len << ": " << result.error_msg;
    }
}

TEST_F(DistilBERTModelTest, AsyncInference) {
    auto profiles = make_nlp_profiles(false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 30522));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

    auto future = engine->infer_async({ids, mask});
    auto result = future.get();
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

// ── GPT-2 fixture ───────────────────────────────────────────────────────────

class GPT2ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        models_dir_ = find_models_dir();
        if (models_dir_.empty() || !model_exists(models_dir_, "gpt2"))
            GTEST_SKIP() << "GPT-2 model not found";
        onnx_path_ = models_dir_ + "/gpt2/model.onnx";
        engine_path_ = models_dir_ + "/gpt2/model_fp16.engine";
    }

    std::string models_dir_;
    std::string onnx_path_;
    std::string engine_path_;
};

TEST_F(GPT2ModelTest, BuildAndInfer) {
    // GPT-2 model was exported with batch_size=1 hardcoded in output shape,
    // so we use fixed-batch profiles (batch=1, varying sequence length only).
    auto profiles = make_nlp_profiles(/*include_token_type_ids=*/false,
                                      /*batch_dynamic=*/false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty()) << "Failed to build GPT-2 engine";

    auto engine = InferenceEngine::create(path);
    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 50257));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

    auto result = engine->infer({ids, mask});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
    EXPECT_GT(result.latency_ms, 0.0f);
}

TEST_F(GPT2ModelTest, Inference_VaryingSeqLen) {
    // GPT-2 only supports batch=1 due to export constraints; test varying
    // sequence lengths instead.
    auto profiles = make_nlp_profiles(false, /*batch_dynamic=*/false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 1;

    for (int seq_len : {32, 64, 128, 256}) {
        engine->set_input_shape("input_ids",       {batch, seq_len});
        engine->set_input_shape("attention_mask",   {batch, seq_len});

        auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 50257));
        auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

        auto result = engine->infer({ids, mask});
        EXPECT_TRUE(result.success)
            << "seq_len=" << seq_len << ": " << result.error_msg;
        EXPECT_GT(result.outputs.size(), 0u);
    }
}

// ── T5-small fixture ────────────────────────────────────────────────────────

class T5SmallModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        models_dir_ = find_models_dir();
        if (models_dir_.empty() || !model_exists(models_dir_, "t5-small"))
            GTEST_SKIP() << "T5-small model not found";
        onnx_path_ = models_dir_ + "/t5-small/model.onnx";
        engine_path_ = models_dir_ + "/t5-small/model_fp16.engine";
    }

    std::string models_dir_;
    std::string onnx_path_;
    std::string engine_path_;
};

TEST_F(T5SmallModelTest, BuildAndInfer) {
    auto profiles = make_nlp_profiles(/*include_token_type_ids=*/false);
    auto path = build_or_load_engine(onnx_path_, engine_path_, profiles);
    ASSERT_FALSE(path.empty()) << "Failed to build T5-small engine";

    auto engine = InferenceEngine::create(path);
    const int batch = 1, seq_len = 128;

    engine->set_input_shape("input_ids",       {batch, seq_len});
    engine->set_input_shape("attention_mask",   {batch, seq_len});

    auto ids  = create_int64_as_float(create_input_ids(batch, seq_len, 32128));
    auto mask = create_int64_as_float(create_attention_mask(batch, seq_len));

    auto result = engine->infer({ids, mask});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

// ── ResNet-18 (CV model, float inputs) ──────────────────────────────────────

class ResNet18ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) GTEST_SKIP() << "No CUDA device";
        models_dir_ = find_models_dir();
        if (models_dir_.empty() || !model_exists(models_dir_, "resnet18"))
            GTEST_SKIP() << "ResNet-18 model not found";
        onnx_path_   = models_dir_ + "/resnet18/model.onnx";
        engine_path_ = models_dir_ + "/resnet18/model_fp16.engine";
    }

    std::string models_dir_;
    std::string onnx_path_;
    std::string engine_path_;
};

TEST_F(ResNet18ModelTest, BuildAndInfer) {
    DynamicShapeProfile p;
    p.name     = "input";
    p.min_dims = {1, 3, 224, 224};
    p.opt_dims = {4, 3, 224, 224};
    p.max_dims = {32, 3, 224, 224};

    auto path = build_or_load_engine(onnx_path_, engine_path_, {p});
    ASSERT_FALSE(path.empty()) << "Failed to build ResNet-18 engine";

    auto engine = InferenceEngine::create(path);
    const int batch = 1;

    engine->set_input_shape("input", {batch, 3, 224, 224});

    std::vector<float> img(batch * 3 * 224 * 224, 0.5f);
    auto result = engine->infer({img});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}

TEST_F(ResNet18ModelTest, BatchInference) {
    DynamicShapeProfile p;
    p.name     = "input";
    p.min_dims = {1, 3, 224, 224};
    p.opt_dims = {4, 3, 224, 224};
    p.max_dims = {32, 3, 224, 224};

    auto path = build_or_load_engine(onnx_path_, engine_path_, {p});
    ASSERT_FALSE(path.empty());

    auto engine = InferenceEngine::create(path);
    const int batch = 8;

    engine->set_input_shape("input", {batch, 3, 224, 224});

    std::vector<float> img(batch * 3 * 224 * 224, 0.5f);
    auto result = engine->infer({img});
    EXPECT_TRUE(result.success) << result.error_msg;
    EXPECT_GT(result.outputs.size(), 0u);
}
