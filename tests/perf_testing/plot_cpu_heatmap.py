#!/usr/bin/env python3
"""
Plot CPU performance heatmap from run_perf.py JSONL results.

- Parses the JSONL results file produced by tests/perf_testing/run_perf.py
- Filters for CPU runs (mode == "cpu")
- Aggregates performance across repeated runs for each matrix point (threads, parallel)
- Renders a heatmap with threads on the Y-axis and cores (parallel) on the X-axis

Example:
  python tests/perf_testing/plot_cpu_heatmap.py \
    --results /path/to/results.jsonl \
    --metric throughput \
    --agg mean \
    --output cpu_heatmap.png \
    --annotate

Metrics:
  - wall          : total wall time per matrix point (seconds). Lower is better.
  - avg_proc      : average per-process elapsed time (seconds) over per_proc entries.
  - median_proc   : median per-process elapsed time (seconds).
  - sum_proc      : sum of per-process elapsed times (seconds).
  - throughput    : completed processes per minute, i.e. (parallel / wall time) * 60. Higher is better.
  - throughput_per_image : throughput divided by parallel (images per minute per process). Higher is better.
  - success_rate  : fraction of successful per-proc tasks (0..1). Higher is better.

Aggregations (applied across repeated runs for the same threads/parallel):
  - mean, median, min, max

Notes:
  - By default, failed matrix-point runs (record["success"] == False) are ignored.
    Use --include-failed to include them in aggregation, in which case metric values
    for failed runs may still compute if data is present; else NaN will be used.
  - Matplotlib is required for plotting:
      pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np

# Optional import with clearer error if missing
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _compute_metric(rec: Dict[str, Any], metric: str) -> float:
    """Compute a scalar metric from a single CPU run record."""
    total_wall = float(rec.get("total_wall_s", float("nan")))
    per_proc = rec.get("per_proc", [])
    parallel = int(rec.get("parallel", 0)) or max((int(p.get("i", 0)) for p in per_proc), default=-1) + 1

    if metric == "wall":
        return total_wall
    elif metric in ("avg_proc", "median_proc", "sum_proc"):
        vals = [float(p.get("elapsed_s", float("nan"))) for p in per_proc]
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            return float("nan")
        if metric == "avg_proc":
            return float(np.nanmean(arr))
        if metric == "median_proc":
            return float(np.nanmedian(arr))
        return float(np.nansum(arr))
    elif metric == "throughput":
        # processes completed per minute
        if total_wall <= 0 or math.isnan(total_wall) or parallel <= 0:
            return float("nan")
        return float(parallel) / total_wall * 60.0
    elif metric == "throughput_per_image":
        # images per minute per process (throughput divided by parallel)
        if total_wall <= 0 or math.isnan(total_wall) or parallel <= 0:
            return float("nan")
        return (float(parallel) / total_wall / float(parallel)) * 60.0
    elif metric == "success_rate":
        if not per_proc:
            return float("nan")
        succ = [bool(p.get("success", False)) for p in per_proc]
        return float(sum(succ)) / float(len(succ))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _aggregate(values: List[float], agg: str) -> float:
    """Aggregate a list of values into one scalar using the chosen aggregator."""
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    if agg == "mean":
        return float(np.nanmean(arr))
    if agg == "median":
        return float(np.nanmedian(arr))
    if agg == "min":
        return float(np.nanmin(arr))
    if agg == "max":
        return float(np.nanmax(arr))
    raise ValueError(f"Unknown aggregation: {agg}")


def load_cpu_records(path: str, include_failed: bool) -> List[Dict[str, Any]]:
    """Load and filter CPU run records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("type") != "run":
                continue
            if obj.get("mode") != "cpu":
                continue
            if not include_failed and not obj.get("success", False):
                # Skip failed matrix points by default
                continue
            records.append(obj)
    return records


def build_matrix(records: List[Dict[str, Any]], metric: str, agg: str) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Build a 2D matrix of shape (n_threads, n_cores) with sorted unique thread and core counts.

    Returns:
      Z: 2D array (rows=threads, cols=cores)
      threads_sorted: sorted unique thread counts
      cores_sorted: sorted unique core counts (parallel)
    """
    # Collect values for each (threads, cores) pair across runs
    bucket: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    threads_set = set()
    cores_set = set()

    for rec in records:
        threads = rec.get("threads", None)
        cores = rec.get("parallel", None)
        if threads is None or cores is None:
            continue
        try:
            t = int(threads)
            c = int(cores)
        except Exception:
            continue
        val = _compute_metric(rec, metric)
        threads_set.add(t)
        cores_set.add(c)
        bucket[(t, c)].append(val)

    threads_sorted = sorted(threads_set)
    cores_sorted = sorted(cores_set)

    if not threads_sorted or not cores_sorted:
        raise RuntimeError("No CPU records with valid 'threads' and 'parallel' found.")

    # Build matrix filled with NaN
    Z = np.full((len(threads_sorted), len(cores_sorted)), np.nan, dtype=float)
    # Fill with aggregated values
    t_index = {t: i for i, t in enumerate(threads_sorted)}
    c_index = {c: j for j, c in enumerate(cores_sorted)}
    for (t, c), vals in bucket.items():
        i = t_index[t]
        j = c_index[c]
        Z[i, j] = _aggregate(vals, agg)

    return Z, threads_sorted, cores_sorted


def plot_heatmap(Z: np.ndarray,
                 threads: List[int],
                 cores: List[int],
                 metric: str,
                 agg: str,
                 output: str,
                 cmap: str = "viridis",
                 annotate: bool = False,
                 title: str | None = None) -> None:
    """Render and save the heatmap."""
    if plt is None:
        raise RuntimeError(f"matplotlib is not available: {_IMPORT_ERR}")

    fig, ax = plt.subplots(figsize=(max(6, len(cores) * 0.8), max(4, len(threads) * 0.6)))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap)

    # Ticks and labels
    ax.set_xticks(np.arange(len(cores)))
    ax.set_yticks(np.arange(len(threads)))
    ax.set_xticklabels([str(c) for c in cores])
    ax.set_yticklabels([str(t) for t in threads])
    ax.set_xlabel("Cores (parallel processes)")
    ax.set_ylabel("Threads per process")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    if metric == "wall":
        cbar.set_label("Wall time (s)")
    elif metric in ("avg_proc", "median_proc", "sum_proc"):
        cbar.set_label("Time (s)")
    elif metric == "throughput":
        cbar.set_label("Throughput (procs/min)")
    elif metric == "throughput_per_image":
        cbar.set_label("Per-image throughput (images/min)")
    elif metric == "success_rate":
        cbar.set_label("Success rate (0..1)")

    # Title
    if title is None:
        title = f"CPU performance heatmap — metric={metric}, agg={agg}"
    ax.set_title(title)

    # Annotation
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isnan(val):
                    text = "NaN"
                else:
                    if metric in ("throughput", "success_rate"):
                        text = f"{val:.2f}"
                    else:
                        text = f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="w", fontsize=8)

    fig.tight_layout()
    outdir = os.path.dirname(os.path.abspath(output))
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_heatmaps_side_by_side(Z1: np.ndarray,
                               Z2: np.ndarray,
                               threads: List[int],
                               cores: List[int],
                               metric1: str,
                               metric2: str,
                               agg: str,
                               output: str,
                               cmap: str = "viridis",
                               annotate: bool = False,
                               title: str | None = None) -> None:
    """Render two heatmaps (throughput and per-image throughput) into a single figure and save."""
    if plt is None:
        raise RuntimeError(f"matplotlib is not available: {_IMPORT_ERR}")

    # Figure sizing scales with matrix size
    fig, axes = plt.subplots(1, 2,
                             figsize=(max(10, len(cores) * 1.6), max(4, len(threads) * 0.6)),
                             constrained_layout=True)

    def _draw(ax, Z, metric):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(cores)))
        ax.set_yticks(np.arange(len(threads)))
        ax.set_xticklabels([str(c) for c in cores])
        ax.set_yticklabels([str(t) for t in threads])
        ax.set_xlabel("Cores (parallel processes)")
        ax.set_ylabel("Threads per process")

        cbar = fig.colorbar(im, ax=ax)
        if metric == "wall":
            cbar.set_label("Wall time (s)")
        elif metric in ("avg_proc", "median_proc", "sum_proc"):
            cbar.set_label("Time (s)")
        elif metric == "throughput":
            cbar.set_label("Throughput (procs/min)")
        elif metric == "throughput_per_image":
            cbar.set_label("Per-image throughput (images/min)")
        elif metric == "success_rate":
            cbar.set_label("Success rate (0..1)")

        # Axis title
        pretty = {
            "wall": "Wall time",
            "avg_proc": "Avg per-proc time",
            "median_proc": "Median per-proc time",
            "sum_proc": "Sum per-proc time",
            "throughput": "Throughput (procs/min)",
            "throughput_per_image": "Per-image throughput (images/min)",
            "success_rate": "Success rate",
        }.get(metric, metric)
        ax.set_title(pretty)

        if annotate:
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    val = Z[i, j]
                    text = "NaN" if np.isnan(val) else f"{val:.2f}"
                    ax.text(j, i, text, ha="center", va="center", color="w", fontsize=8)

    # Draw both panels
    _draw(axes[0], Z1, metric1)
    _draw(axes[1], Z2, metric2)

    # Figure title
    if title is None:
        title = f"CPU performance heatmaps — agg={agg}"
    fig.suptitle(title)

    outdir = os.path.dirname(os.path.abspath(output))
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Plot CPU performance heatmap from JSONL results.")
    ap.add_argument("--results", required=True, help="Path to results.jsonl produced by run_perf.py")
    ap.add_argument("--metric",
                    choices=["wall", "avg_proc", "median_proc", "sum_proc", "throughput", "throughput_per_image", "success_rate"],
                    default="throughput",
                    help="Metric to plot on the heatmap (default: throughput)")
    ap.add_argument("--agg",
                    choices=["mean", "median", "min", "max"],
                    default="mean",
                    help="Aggregation across repeated runs (default: mean)")
    ap.add_argument("--output", default="cpu_heatmap.png", help="Output image path (default: cpu_heatmap.png)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name (default: viridis)")
    ap.add_argument("--annotate", action="store_true", help="Annotate cells with numeric values")
    ap.add_argument("--include-failed", action="store_true",
                    help="Include failed matrix-point runs in the aggregation (default: skip)")
    ap.add_argument("--title", default=None, help="Optional custom plot title")

    args = ap.parse_args(argv)

    try:
        records = load_cpu_records(args.results, include_failed=args.include_failed)
        if not records:
            print("No CPU run records found in the results file.", file=sys.stderr)
            return 2
        # If the selected metric is throughput, produce a single multi-plot image with
        # both throughput (procs/min) and per-image throughput (images/min).
        if args.metric == "throughput":
            Z1, threads, cores = build_matrix(records, metric="throughput", agg=args.agg)
            Z2, _, _ = build_matrix(records, metric="throughput_per_image", agg=args.agg)
            plot_heatmaps_side_by_side(Z1, Z2, threads, cores,
                                       metric1="throughput", metric2="throughput_per_image",
                                       agg=args.agg, output=args.output, cmap=args.cmap,
                                       annotate=args.annotate,
                                       title=args.title or "CPU performance heatmaps — throughput & per-image")
            print(f"Saved combined heatmap to: {args.output}")
            return 0
        else:
            Z, threads, cores = build_matrix(records, metric=args.metric, agg=args.agg)
            plot_heatmap(Z, threads, cores, metric=args.metric, agg=args.agg,
                         output=args.output, cmap=args.cmap, annotate=args.annotate, title=args.title)
            print(f"Saved heatmap to: {args.output}")
            return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
