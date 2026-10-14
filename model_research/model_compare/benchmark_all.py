r"""
Benchmark PyTorch .pt vs TensorRT FP32 vs TensorRT FP16 on preloaded frames.
Outputs CSV with per-frame inference_ms and a line chart.

Usage:
    python benchmark_all.py --model people_counting --source C:\\cctv-dev-temp\\stairs_people.mp4 --runs 500
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


MODELS = {
    "ppe_compliance": "ppe_compliance_yolo26m",
    "helmet_and_vest": "helmet_and_vest_yolo26m",
    "face_mask": "face_mask_yolo26m",
    "hand_in_pocket": "hand_in_pocket_yolo26m",
    "holding_phone": "holding_phone_yolo26m",
    "smartphone": "smartphone_yolo26m",
    "phone_usage": "phone_usage_yolo26m",
    "people_counting": "people_counting_yolo26m",
    "security_officer_v2": "security_officer_v2_yolo26m",
}

RUNS_DETECT_DIR = Path(r"C:\cctv-insight\ai\research\runs\detect")


def resolve_pt(model_arg: str) -> Path:
    p = Path(model_arg)
    if p.suffix.lower() == ".pt" and p.is_file():
        return p
    key = model_arg.lower()
    if key in MODELS:
        cand = RUNS_DETECT_DIR / MODELS[key] / "weights" / "best.pt"
        if cand.is_file():
            return cand
    cand = RUNS_DETECT_DIR / model_arg / "weights" / "best.pt"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Model tidak ditemukan: {model_arg}")


def build_engine(pt_path: Path, imgsz: int, quantize: int | None) -> Path:
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    if engine_path.exists():
        print(f"[TRT] {suffix.upper().replace('_', '')} engine ditemukan: {engine_path.name}")
        return engine_path

    print(f"[TRT] Exporting {suffix.upper().replace('_', '')} engine dari {pt_path.name} ...")
    model = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, batch=1, workspace=2, simplify=True, verbose=False)
    if quantize == 16:
        kwargs["quantize"] = 16
    model.export(**kwargs)

    produced = pt_path.with_suffix(".engine")
    if produced.exists():
        produced.rename(engine_path)
    return engine_path


def load_frames(source: str, num_frames: int, imgsz: int) -> list[np.ndarray]:
    src = Path(source) if source else None
    if not src or not src.is_file():
        fallback = Path(r"C:\cctv-dev-temp\stairs_people.mp4")
        src = fallback if fallback.is_file() else None

    frames: list[np.ndarray] = []
    if src and src.is_file():
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka video: {src}")
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames.append(cv2.resize(frame, (imgsz, imgsz)))
        cap.release()
        print(f"[INPUT] {len(frames)} frame @ {imgsz}x{imgsz} dari {src}")
        return frames

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    frames = [img.copy() for _ in range(num_frames)]
    print(f"[INPUT] {len(frames)} synthetic frame @ {imgsz}x{imgsz}")
    return frames


def benchmark(model_path: Path, frames: list[np.ndarray], backend: str, warmup: int, device: str):
    model = YOLO(str(model_path))
    times = []

    print(f"[{backend}] Warm-up {warmup} iterasi ...")
    for i in range(warmup):
        model.predict(source=frames[i % len(frames)], verbose=False, device=device, stream=False)
        torch.cuda.synchronize()

    print(f"[{backend}] Running {len(frames)} iterasi ...")
    for idx, img in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(source=img, verbose=False, device=device, stream=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        inf_ms = float(results[0].speed.get("inference", 0))
        wall_ms = (t1 - t0) * 1000.0
        times.append((idx + 1, inf_ms, wall_ms))

    return times


def save_csv(out_csv: Path, all_times: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame_id", "backend", "inference_ms", "wall_ms"])
        for backend, times in all_times.items():
            for frame_id, inf_ms, wall_ms in times:
                writer.writerow([frame_id, backend, f"{inf_ms:.3f}", f"{wall_ms:.3f}"])
    print(f"[CSV] Saved -> {out_csv}")


def plot(out_png: Path, all_times: dict, title: str):
    palette = {"pt": "#1f77b4", "fp32": "#ff7f0e", "fp16": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6))
    summary_lines = []

    for backend, times in all_times.items():
        frames = np.array([t[0] for t in times])
        vals = np.array([t[1] for t in times])
        color = palette.get(backend, "#9467bd")
        label = backend.upper() if backend != "pt" else "PyTorch .pt"
        ax.plot(frames, vals, "o-", markersize=3, linewidth=1.5, label=label, color=color)

        mean_ms = float(np.mean(vals))
        ax.axhline(mean_ms, color=color, linestyle="--", alpha=0.6, label=f"{label} mean: {mean_ms:.2f} ms")
        summary_lines.append(f"{label}: {mean_ms:.2f} ± {np.std(vals):.2f} ms")

    ax.set_xlabel("Frame (t)", fontsize=12)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=1)

    pt_mean = float(np.mean([t[1] for t in all_times["pt"]]))
    for backend in ["fp32", "fp16"]:
        eng_mean = float(np.mean([t[1] for t in all_times[backend]]))
        speedup = pt_mean / eng_mean if eng_mean else float("inf")
        summary_lines.append(f"Speedup {backend.upper()} vs PT: {speedup:.2f}x")

    ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[PLOT] Saved -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="people_counting")
    ap.add_argument("--source", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out-csv", default="benchmark_all.csv")
    ap.add_argument("--out-png", default="benchmark_all.png")
    args = ap.parse_args()

    pt_path = resolve_pt(args.model)
    fp32_path = build_engine(pt_path, args.imgsz, quantize=None)
    fp16_path = build_engine(pt_path, args.imgsz, quantize=16)

    print(f"[MODEL] PT    = {pt_path}")
    print(f"[MODEL] FP32  = {fp32_path}")
    print(f"[MODEL] FP16  = {fp16_path}")

    frames = load_frames(args.source, args.runs, args.imgsz)

    all_times = {}

    torch.cuda.empty_cache()
    all_times["pt"] = benchmark(pt_path, frames, "PyTorch .pt", args.warmup, args.device)

    torch.cuda.empty_cache()
    all_times["fp32"] = benchmark(fp32_path, frames, "TensorRT FP32", args.warmup, args.device)

    torch.cuda.empty_cache()
    all_times["fp16"] = benchmark(fp16_path, frames, "TensorRT FP16", args.warmup, args.device)

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    save_csv(out_csv, all_times)
    plot(out_png, all_times, title=f"People Counting | PT vs FP32 vs FP16 ({args.imgsz}x{args.imgsz})")

    print("\n=== SUMMARY ===")
    for backend, times in all_times.items():
        arr = np.array([t[1] for t in times])
        label = backend.upper() if backend != "pt" else "PT"
        print(f"{label:6s}: mean={arr.mean():.2f}ms std={arr.std():.2f}ms min={arr.min():.2f}ms max={arr.max():.2f}ms")

    pt_mean = np.mean([t[1] for t in all_times["pt"]])
    for backend in ["fp32", "fp16"]:
        eng_mean = np.mean([t[1] for t in all_times[backend]])
        print(f"Speedup {backend.upper()} vs PT: {pt_mean / eng_mean:.2f}x")


if __name__ == "__main__":
    main()
