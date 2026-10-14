r"""
Benchmark PyTorch .pt vs TensorRT FP16 .engine inference on preloaded frames.
Outputs CSV with per-frame inference_ms and a line chart.

Usage:
    python benchmark_pt_vs_fp16.py --model people_counting --source C:\\cctv-dev-temp\\stairs_people.mp4 --runs 500
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
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


def build_engine(pt_path: Path, imgsz: int, force: bool = False) -> Path:
    suffix = "_fp16"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    fallback_fp32 = pt_path.with_name(f"{pt_path.stem}_fp32.engine")

    if engine_path.exists() and not force:
        print(f"[TRT] FP16 engine ditemukan: {engine_path.name}")
        return engine_path

    if fallback_fp32.exists() and not force:
        print(f"[TRT] FP16 belum ada, reuse FP32: {fallback_fp32.name}")
        return fallback_fp32

    print(f"[TRT] Exporting FP16 engine dari {pt_path.name} ...")
    model = YOLO(str(pt_path))
    try:
        model.export(format="engine", imgsz=imgsz, batch=1, simplify=True, quantize=16)
        produced = pt_path.with_suffix(".engine")
        produced.rename(engine_path)
        return engine_path
    except Exception as exc:
        print(f"[WARN] FP16 export gagal ({exc.__class__.__name__}); fallback FP32 ...")
        for junk in [pt_path.with_suffix(".engine"), pt_path.with_suffix(".onnx")]:
            if junk.exists():
                try:
                    junk.unlink()
                except OSError:
                    pass
        model.export(format="engine", imgsz=imgsz, batch=1, simplify=True)
        produced = pt_path.with_suffix(".engine")
        produced.rename(fallback_fp32)
        return fallback_fp32


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
            frame = cv2.resize(frame, (imgsz, imgsz))
            frames.append(frame)
        cap.release()
        print(f"[INPUT] {len(frames)} frame @ {imgsz}x{imgsz} dari {src}")
        return frames

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    frames = [img.copy() for _ in range(num_frames)]
    print(f"[INPUT] {len(frames)} synthetic frame @ {imgsz}x{imgsz}")
    return frames


def benchmark(
    model_path: Path,
    frames: list[np.ndarray],
    backend: str,
    warmup: int,
    device: str,
):
    model = YOLO(str(model_path))
    times: list[tuple[int, float]] = []

    # Warm-up
    print(f"[{backend}] Warm-up {warmup} iterasi ...")
    for i, img in enumerate(frames):
        model.predict(source=img, verbose=False, device=device, stream=False)
        torch.cuda.synchronize()
        if i >= warmup - 1:
            break

    # Measured
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


def save_csv(out_csv: Path, pt_times, engine_times, engine_label: str):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame_id", "backend", "inference_ms", "wall_ms"])
        for frame_id, inf_ms, wall_ms in pt_times:
            writer.writerow([frame_id, "pt", f"{inf_ms:.3f}", f"{wall_ms:.3f}"])
        for frame_id, inf_ms, wall_ms in engine_times:
            writer.writerow([frame_id, engine_label, f"{inf_ms:.3f}", f"{wall_ms:.3f}"])
    print(f"[CSV] Saved -> {out_csv}")


def plot(out_png: Path, pt_times, engine_times, engine_label: str, title: str):
    import matplotlib.pyplot as plt

    pt_frames = np.array([t[0] for t in pt_times])
    pt_vals = np.array([t[1] for t in pt_times])
    eng_frames = np.array([t[0] for t in engine_times])
    eng_vals = np.array([t[1] for t in engine_times])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pt_frames, pt_vals, "o-", markersize=3, linewidth=1.5, label="PyTorch .pt", color="#1f77b4")
    ax.plot(eng_frames, eng_vals, "o-", markersize=3, linewidth=1.5, label=engine_label.upper(), color="#d62728")

    pt_mean = float(np.mean(pt_vals))
    eng_mean = float(np.mean(eng_vals))
    ax.axhline(pt_mean, color="#1f77b4", linestyle="--", alpha=0.6, label=f"PT mean: {pt_mean:.2f} ms")
    ax.axhline(eng_mean, color="#d62728", linestyle="--", alpha=0.6, label=f"{engine_label.upper()} mean: {eng_mean:.2f} ms")

    ax.set_xlabel("Frame (t)", fontsize=12)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(left=1)

    speedup = pt_mean / eng_mean if eng_mean else float("inf")
    text = f"PT: {pt_mean:.2f} ± {np.std(pt_vals):.2f} ms\n{engine_label.upper()}: {eng_mean:.2f} ± {np.std(eng_vals):.2f} ms\nSpeedup: {speedup:.2f}x"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[PLOT] Saved -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="people_counting", help="registry key atau path .pt")
    ap.add_argument("--source", default="", help="video path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out-csv", default="benchmark_pt_fp16.csv")
    ap.add_argument("--out-png", default="benchmark_pt_fp16.png")
    args = ap.parse_args()

    pt_path = resolve_pt(args.model)
    engine_path = build_engine(pt_path, args.imgsz)
    engine_label = engine_path.stem.split("_")[-1]  # fp16 atau fp32

    print(f"[MODEL] PT      = {pt_path}")
    print(f"[MODEL] Engine  = {engine_path}")

    frames = load_frames(args.source, args.runs, args.imgsz)

    # Clear GPU memory sebelum benchmark pertama agar PT dan engine start dari kondisi sama
    import torch
    torch.cuda.empty_cache()

    pt_times = benchmark(pt_path, frames, "PT", args.warmup, args.device)

    # Clear GPU memory sebelum benchmark engine, reset CUDA state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    engine_times = benchmark(engine_path, frames, f"TensorRT {engine_label.upper()}", args.warmup, args.device)

    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    save_csv(out_csv, pt_times, engine_times, engine_label)
    plot(out_png, pt_times, engine_times, engine_label,
         title=f"People Counting | PyTorch .pt vs TensorRT {engine_label.upper()} ({args.imgsz}x{args.imgsz})")

    print("\n=== SUMMARY ===")
    pt_arr = np.array([t[1] for t in pt_times])
    eng_arr = np.array([t[1] for t in engine_times])
    print(f"PT      : mean={pt_arr.mean():.2f}ms std={pt_arr.std():.2f}ms min={pt_arr.min():.2f}ms max={pt_arr.max():.2f}ms")
    print(f"Engine  : mean={eng_arr.mean():.2f}ms std={eng_arr.std():.2f}ms min={eng_arr.min():.2f}ms max={eng_arr.max():.2f}ms")
    print(f"Speedup : {pt_arr.mean()/eng_arr.mean():.2f}x")


if __name__ == "__main__":
    main()
