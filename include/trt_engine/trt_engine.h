#pragma once

// ── trt_engine: Main convenience header ─────────────────────────────────
// Include this single header to access the full public API.

#define TRT_ENGINE_VERSION_MAJOR 1
#define TRT_ENGINE_VERSION_MINOR 0
#define TRT_ENGINE_VERSION_PATCH 0

#include <trt_engine/types.h>
#include <trt_engine/logger.h>
#include <trt_engine/memory.h>
#include <trt_engine/cuda_utils.h>
#include <trt_engine/model_converter.h>
#include <trt_engine/builder.h>
#include <trt_engine/calibrator.h>
#include <trt_engine/engine.h>
#include <trt_engine/cuda_graph.h>
#include <trt_engine/multi_stream.h>
#include <trt_engine/batcher.h>
#include <trt_engine/multi_gpu.h>
#include <trt_engine/profiler.h>
