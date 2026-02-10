#!/bin/bash
# run_nlp_benchmarks.sh - Run NLP benchmarks for all available models
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/../build}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/../models}"
RESULTS_DIR="$SCRIPT_DIR/results"

BENCHMARK_BIN="$BUILD_DIR/benchmarks/benchmark_nlp"

if [ ! -x "$BENCHMARK_BIN" ]; then
    echo "Error: benchmark binary not found at $BENCHMARK_BIN"
    echo "Build the project first: cmake --build build"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

PRECISION="${PRECISION:-FP16}"
BATCH_SIZES="${BATCH_SIZES:-1,4,8}"
SEQ_LENGTHS="${SEQ_LENGTHS:-64,128}"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-20}"

echo "=== NLP Benchmark Suite ==="
echo "Models dir:  $MODELS_DIR"
echo "Precision:   $PRECISION"
echo "Batch sizes: $BATCH_SIZES"
echo "Seq lengths: $SEQ_LENGTHS"
echo "Iterations:  $ITERATIONS"
echo "Warmup:      $WARMUP"
echo ""

for model in bert-base distilbert gpt2 t5-small; do
    ONNX_FILE="$MODELS_DIR/$model/model.onnx"
    if [ ! -f "$ONNX_FILE" ]; then
        echo "--- Skipping $model (model.onnx not found) ---"
        echo ""
        continue
    fi

    echo "--- Benchmarking $model ---"
    "$BENCHMARK_BIN" \
        --model "$model" \
        --models-dir "$MODELS_DIR" \
        --precision "$PRECISION" \
        --batch-sizes "$BATCH_SIZES" \
        --seq-lengths "$SEQ_LENGTHS" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --output "$RESULTS_DIR/${model}_benchmark.json"
    echo ""
done

echo "=== Results saved to $RESULTS_DIR ==="
echo "=== Done ==="
