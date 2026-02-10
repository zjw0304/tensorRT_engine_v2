#!/usr/bin/env python3
"""
benchmark_report.py - Generate a markdown report from JSON benchmark results.

Reads throughput and latency JSON result files and produces a formatted
markdown table suitable for documentation.

Usage:
    python benchmark_report.py --throughput results_throughput.json \
                               --latency results_latency.json \
                               --output report.md
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_throughput_table(data: Dict[str, Any]) -> str:
    """Generate a markdown table from throughput benchmark results."""
    lines = []
    lines.append("## Throughput Benchmark Results\n")
    lines.append(f"- **Engine**: `{data.get('engine', 'N/A')}`")
    lines.append(f"- **Precision**: {data.get('precision', 'N/A')}")
    lines.append(f"- **Iterations**: {data.get('iterations', 'N/A')}")
    lines.append("")

    # Table header
    lines.append(
        "| Batch Size | CUDA Graph | Throughput (inf/s) | Images/sec | Mean Latency (ms) |"
    )
    lines.append(
        "|:----------:|:----------:|-------------------:|-----------:|------------------:|"
    )

    for r in data.get("results", []):
        cuda_graph = "Yes" if r.get("cuda_graph", False) else "No"
        lines.append(
            f"| {r['batch_size']:^10} "
            f"| {cuda_graph:^10} "
            f"| {r['throughput_ips']:>18.2f} "
            f"| {r['images_per_sec']:>10.2f} "
            f"| {r['mean_latency_ms']:>17.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_latency_table(data: Dict[str, Any]) -> str:
    """Generate a markdown table from latency benchmark results."""
    lines = []
    lines.append("## Latency Benchmark Results\n")
    lines.append(f"- **Engine**: `{data.get('engine', 'N/A')}`")
    lines.append(f"- **Batch Size**: {data.get('batch_size', 'N/A')}")
    lines.append(f"- **Iterations**: {data.get('iterations', 'N/A')}")
    lines.append("")

    # Table header
    lines.append(
        "| Mode  | Min (ms) | Max (ms) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |"
    )
    lines.append(
        "|:-----:|---------:|---------:|----------:|---------:|---------:|---------:|"
    )

    for r in data.get("results", []):
        lines.append(
            f"| {r['mode']:^5} "
            f"| {r['min_ms']:>8.4f} "
            f"| {r['max_ms']:>8.4f} "
            f"| {r['mean_ms']:>9.4f} "
            f"| {r['p50_ms']:>8.4f} "
            f"| {r['p95_ms']:>8.4f} "
            f"| {r['p99_ms']:>8.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report from benchmark JSON results"
    )
    parser.add_argument(
        "--throughput",
        type=str,
        default=None,
        help="Path to throughput benchmark JSON results",
    )
    parser.add_argument(
        "--latency",
        type=str,
        default=None,
        help="Path to latency benchmark JSON results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown file path (stdout if not specified)",
    )
    args = parser.parse_args()

    if not args.throughput and not args.latency:
        parser.error("At least one of --throughput or --latency must be provided")

    report_parts: List[str] = []
    report_parts.append("# TensorRT Engine Benchmark Report\n")

    if args.throughput:
        if not os.path.isfile(args.throughput):
            print(f"Error: throughput file not found: {args.throughput}", file=sys.stderr)
            return 1
        data = load_json(args.throughput)
        report_parts.append(format_throughput_table(data))

    if args.latency:
        if not os.path.isfile(args.latency):
            print(f"Error: latency file not found: {args.latency}", file=sys.stderr)
            return 1
        data = load_json(args.latency)
        report_parts.append(format_latency_table(data))

    report = "\n".join(report_parts)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
