#!/usr/bin/env python3
"""benchmark_nlp_report.py - Generate markdown report from NLP benchmark JSON results.

Usage:
    python benchmark_nlp_report.py [--results-dir DIR] [--output FILE]

Reads all *_benchmark.json files from the results directory and produces
a markdown report with comparison tables and latency distribution charts.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_results(results_dir):
    """Load all benchmark JSON files from the results directory."""
    all_results = []
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        return all_results

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_benchmark.json"):
            continue
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            r.setdefault("model", data.get("model", "unknown"))
            r.setdefault("precision", data.get("precision", "FP16"))
            all_results.append(r)

    return all_results


def make_table(headers, rows):
    """Create a markdown table from headers and row data."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def latency_bar(value, max_value, width=30):
    """Create a text-based bar for latency visualization."""
    if max_value <= 0:
        return ""
    filled = int(round((value / max_value) * width))
    filled = min(filled, width)
    return "#" * filled + "." * (width - filled)


def generate_report(results):
    """Generate a markdown report from benchmark results."""
    lines = ["# NLP Benchmark Report", ""]

    if not results:
        lines.append("No benchmark results found.")
        return "\n".join(lines)

    # --- Summary table: model x batch_size ---
    lines.append("## Throughput by Model and Batch Size")
    lines.append("")

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    all_batch_sizes = sorted(set(r["batch_size"] for r in results))
    headers = ["Model", "SeqLen"] + [f"BS={bs}" for bs in all_batch_sizes]

    rows = []
    for model in sorted(by_model.keys()):
        model_results = by_model[model]
        seq_lengths = sorted(set(r["seq_length"] for r in model_results))
        for sl in seq_lengths:
            row = [model, str(sl)]
            for bs in all_batch_sizes:
                match = [r for r in model_results
                         if r["batch_size"] == bs and r["seq_length"] == sl]
                if match:
                    row.append(f"{match[0]['throughput_ips']:.1f}")
                else:
                    row.append("-")
            rows.append(row)

    lines.append(make_table(headers, rows))
    lines.append("")
    lines.append("*Throughput in inferences/sec*")
    lines.append("")

    # --- Summary table: model x seq_length ---
    lines.append("## Latency by Model and Sequence Length")
    lines.append("")

    all_seq_lengths = sorted(set(r["seq_length"] for r in results))
    headers = ["Model", "Batch"] + [f"SL={sl}" for sl in all_seq_lengths]

    rows = []
    for model in sorted(by_model.keys()):
        model_results = by_model[model]
        batch_sizes = sorted(set(r["batch_size"] for r in model_results))
        for bs in batch_sizes:
            row = [model, str(bs)]
            for sl in all_seq_lengths:
                match = [r for r in model_results
                         if r["batch_size"] == bs and r["seq_length"] == sl]
                if match:
                    row.append(f"{match[0]['mean_latency_ms']:.2f}")
                else:
                    row.append("-")
            rows.append(row)

    lines.append(make_table(headers, rows))
    lines.append("")
    lines.append("*Mean latency in milliseconds*")
    lines.append("")

    # --- Latency percentile table ---
    lines.append("## Latency Percentiles")
    lines.append("")

    headers = ["Model", "Batch", "SeqLen", "Mean(ms)", "P50(ms)", "P95(ms)", "P99(ms)"]
    rows = []
    for r in sorted(results, key=lambda x: (x["model"], x["batch_size"], x["seq_length"])):
        rows.append([
            r["model"],
            str(r["batch_size"]),
            str(r["seq_length"]),
            f"{r['mean_latency_ms']:.2f}",
            f"{r.get('p50_latency_ms', 0):.2f}",
            f"{r.get('p95_latency_ms', 0):.2f}",
            f"{r.get('p99_latency_ms', 0):.2f}",
        ])

    lines.append(make_table(headers, rows))
    lines.append("")

    # --- Latency distribution chart (text-based) ---
    lines.append("## Latency Distribution (Mean)")
    lines.append("")
    lines.append("```")

    max_latency = max((r["mean_latency_ms"] for r in results), default=1.0)
    for r in sorted(results, key=lambda x: (x["model"], x["batch_size"], x["seq_length"])):
        label = f"{r['model']:12s} bs={r['batch_size']:<3d} sl={r['seq_length']:<4d}"
        bar = latency_bar(r["mean_latency_ms"], max_latency)
        lines.append(f"{label} | {bar} {r['mean_latency_ms']:.2f}ms")

    lines.append("```")
    lines.append("")

    # --- Tokens/sec comparison ---
    lines.append("## Tokens per Second")
    lines.append("")

    headers = ["Model", "Batch", "SeqLen", "Tokens/sec"]
    rows = []
    for r in sorted(results, key=lambda x: -x.get("tokens_per_sec", 0)):
        tps = r.get("tokens_per_sec", 0)
        rows.append([
            r["model"],
            str(r["batch_size"]),
            str(r["seq_length"]),
            f"{tps:,.0f}",
        ])

    lines.append(make_table(headers, rows))
    lines.append("")

    # --- FP32 vs FP16 comparison (if both exist) ---
    by_key = defaultdict(dict)
    for r in results:
        key = (r["model"], r["batch_size"], r["seq_length"])
        by_key[key][r["precision"]] = r

    fp_compare = []
    for key, precisions in sorted(by_key.items()):
        if "FP32" in precisions and "FP16" in precisions:
            r32 = precisions["FP32"]
            r16 = precisions["FP16"]
            speedup = r32["mean_latency_ms"] / r16["mean_latency_ms"] if r16["mean_latency_ms"] > 0 else 0
            fp_compare.append((key, r32, r16, speedup))

    if fp_compare:
        lines.append("## FP32 vs FP16 Comparison")
        lines.append("")
        headers = ["Model", "Batch", "SeqLen", "FP32(ms)", "FP16(ms)", "Speedup"]
        rows = []
        for key, r32, r16, speedup in fp_compare:
            rows.append([
                key[0], str(key[1]), str(key[2]),
                f"{r32['mean_latency_ms']:.2f}",
                f"{r16['mean_latency_ms']:.2f}",
                f"{speedup:.2f}x",
            ])
        lines.append(make_table(headers, rows))
        lines.append("")

    # --- CUDA graph comparison (if both exist) ---
    by_graph = defaultdict(dict)
    for r in results:
        key = (r["model"], r["batch_size"], r["seq_length"], r["precision"])
        cg = r.get("cuda_graph", False)
        by_graph[key][cg] = r

    cg_compare = []
    for key, variants in sorted(by_graph.items()):
        if True in variants and False in variants:
            r_no = variants[False]
            r_yes = variants[True]
            speedup = r_no["mean_latency_ms"] / r_yes["mean_latency_ms"] if r_yes["mean_latency_ms"] > 0 else 0
            cg_compare.append((key, r_no, r_yes, speedup))

    if cg_compare:
        lines.append("## CUDA Graph Speedup")
        lines.append("")
        headers = ["Model", "Batch", "SeqLen", "No Graph(ms)", "Graph(ms)", "Speedup"]
        rows = []
        for key, r_no, r_yes, speedup in cg_compare:
            rows.append([
                key[0], str(key[1]), str(key[2]),
                f"{r_no['mean_latency_ms']:.2f}",
                f"{r_yes['mean_latency_ms']:.2f}",
                f"{speedup:.2f}x",
            ])
        lines.append(make_table(headers, rows))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report from NLP benchmark results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing *_benchmark.json files")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output markdown file (default: stdout)")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    report = generate_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
