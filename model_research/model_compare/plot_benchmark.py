r"""
Plot frame (t) vs inference_ms from modelpt.py / modelengine.py CSV logs.

Usage:
    python plot_benchmark.py --csv benchmark_frames.csv --out benchmark_plot.png
    python plot_benchmark.py --csv benchmark_frames.csv --out benchmark_plot.png --max-frames 500
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: Path):
    rows_by_backend = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backend = row.get("backend", "unknown")
            try:
                frame_id = int(row["frame_id"])
                inf_ms = float(row["inference_ms"])
                rows_by_backend[backend].append((frame_id, inf_ms))
            except (KeyError, ValueError):
                continue

    data = {}
    for backend, rows in rows_by_backend.items():
        rows = sorted(rows, key=lambda x: x[0])
        frames = np.array([r[0] for r in rows], dtype=int)
        times = np.array([r[1] for r in rows], dtype=float)
        data[backend] = (frames, times)
    return data


def rolling_mean(x, y, window=15):
    if len(y) < window:
        return x, y
    cumsum = np.cumsum(np.insert(y, 0, 0))
    smooth = (cumsum[window:] - cumsum[:-window]) / window
    return x[window - 1:], smooth


def plot(data: dict, out_path: Path, max_frames: int | None, title: str):
    palette = {
        "pt": "#1f77b4",
        "engine": "#ff7f0e",
        "engine_fp32": "#2ca02c",
        "engine_fp16": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for backend, (frames, times) in data.items():
        if len(frames) == 0:
            continue

        if max_frames and len(frames) > max_frames:
            frames = frames[:max_frames]
            times = times[:max_frames]

        color = palette.get(backend, "#9467bd")
        label = backend.upper()
        ax.plot(frames, times, linewidth=1.5, alpha=0.85, color=color,
                marker="o", markersize=3, label=f"{label}")

        mean_ms = float(np.mean(times))
        ax.axhline(mean_ms, color=color, linestyle="--", alpha=0.6,
                   label=f"{label} mean: {mean_ms:.2f} ms")

    ax.set_xlabel("Frame (t)", fontsize=12)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=0)

    # Add text summary box
    summary_lines = []
    for backend, (frames, times) in data.items():
        if len(frames) == 0:
            continue
        mean_ms = float(np.mean(times))
        std_ms = float(np.std(times))
        summary_lines.append(f"{backend.upper()}: {mean_ms:.2f} ± {std_ms:.2f} ms")
    if summary_lines:
        ax.text(
            0.02, 0.98, "\n".join(summary_lines),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot frame vs inference_ms")
    ap.add_argument("--csv", required=True, type=Path, help="path to CSV log")
    ap.add_argument("--out", default="benchmark_plot.png", type=Path, help="output PNG")
    ap.add_argument("--max-frames", type=int, default=None, help="limit frames per backend")
    ap.add_argument("--title", default="YOLO Inference Time: PT vs TensorRT Engine", help="plot title")
    args = ap.parse_args()

    data = load_csv(args.csv)
    if not data:
        raise RuntimeError("No valid data found in CSV")

    print("[DATA] backends found:", list(data.keys()))
    for backend, (frames, times) in data.items():
        print(f"  {backend}: {len(frames)} frames, mean={np.mean(times):.2f}ms, std={np.std(times):.2f}ms")

    plot(data, args.out, args.max_frames, args.title)


if __name__ == "__main__":
    main()
