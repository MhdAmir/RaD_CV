"""
Compare YOLO inference: PyTorch (.pt) vs TensorRT (.engine)

Default model : ai/research/runs/detect/ppe_compliance_yolo26m/weights/best.pt
Default video : C:/cctv-dev-temp/stairs_people.mp4 (fallback to street.mp4 / bus.jpg)

Usage (from project venv):
  python model_research/comparasion_model.py
  python model_research/comparasion_model.py --model security_officer_v2 --runs 300 --quantize 16
  python model_research/comparasion_model.py --model C:/path/to/any.pt --no-export   # engine already exists
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------- #
# Registry (relative to ai/research/runs/detect/.../weights/best.pt)
# --------------------------------------------------------------------------- #
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
FALLBACK_VIDEOS = [
    Path(r"C:\cctv-dev-temp\stairs_people.mp4"),
    Path(r"C:\cctv-dev-temp\street.mp4"),
    Path(r"C:\cctv-dev-temp\office.mp4"),
]
FALLBACK_IMAGES = [
    Path(r"C:\cctv-insight\ai\assets\bus.jpg"),  # optional local image
]


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def resolve_pt(model_arg: str) -> Path:
    """Resolve --model to an existing .pt path."""
    p = Path(model_arg)
    if p.suffix.lower() == ".pt" and p.is_file():
        return p

    # try as registry key
    key = model_arg.lower()
    if key in MODELS:
        candidate = RUNS_DETECT_DIR / MODELS[key] / "weights" / "best.pt"
        if candidate.is_file():
            return candidate

    # try as direct folder name under runs/detect
    candidate = RUNS_DETECT_DIR / model_arg / "weights" / "best.pt"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Model tidak ditemukan: {model_arg!r}\n"
        f"Gunakan salah satu registry key: {list(MODELS)}\n"
        f"Atau berikan path file .pt yang valid."
    )


def _try_export(pt_path: Path, imgsz: int, quantize: int) -> None:
    """One export attempt; raises on failure."""
    model = YOLO(str(pt_path))
    # Docs: format='engine', imgsz=..., quantize=16 (FP16) or None (FP32)
    kwargs = dict(format="engine", imgsz=imgsz, batch=1, simplify=True)
    if quantize == 16:
        kwargs["quantize"] = 16
    model.export(**kwargs)


def build_engine(pt_path: Path, imgsz: int, quantize: int, force: bool) -> Path:
    """
    Export .pt -> .engine (FP32 / FP16), dengan fallback otomatis FP16 -> FP32.

    quantize:
      0  -> FP32 (no quantize flag)
      16 -> FP16 (quantize=16). Jika gagal (butuh nvidia-modelopt),
            otomatis fallback ke FP32.
    """
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    fallback_fp32 = pt_path.with_name(f"{pt_path.stem}_fp32.engine")

    if engine_path.exists() and not force:
        print(f"[TensorRT] Engine sudah ada, skip export: {engine_path.name}")
        return engine_path

    if quantize == 16 and fallback_fp32.exists() and not force:
        # reuse existing fp32 engine if fp16 belum dibuat
        print(f"[TensorRT] FP16 belum ada, reuse FP32 engine: {fallback_fp32.name}")
        return fallback_fp32

    try:
        print(f"[TensorRT] Exporting {pt_path.name} -> quantize={quantize or 'fp32'} ...")
        _try_export(pt_path, imgsz, quantize)
        produced = pt_path.with_suffix(".engine")
        produced.rename(engine_path)
        return engine_path
    except Exception as exc:
        if quantize == 16:
            print(f"[WARN] Export FP16 gagal ({exc.__class__.__name__}); fallback ke FP32 ...")
            # bersihkan onnx/engine sisa export gagal
            for junk in [pt_path.with_suffix(".engine"), pt_path.with_suffix(".onnx")]:
                if junk.exists():
                    try:
                        junk.unlink()
                    except OSError:
                        pass
            _try_export(pt_path, imgsz, quantize=0)
            produced = pt_path.with_suffix(".engine")
            produced.rename(fallback_fp32)
            return fallback_fp32
        raise



# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_frames(source_arg: str | None, num_frames: int, imgsz: int) -> list[np.ndarray]:
    """Load up to `num_frames` frames resized to imgsz x imgsz from video or images."""
    frames: list[np.ndarray] = []

    src_path: Path | None = None
    if source_arg:
        src_path = Path(source_arg)
        if not src_path.exists():
            print(f"[WARN] source tidak ditemukan: {source_arg} -> pakai fallback video")
            src_path = None

    if src_path is None:
        for p in FALLBACK_VIDEOS:
            if p.is_file():
                src_path = p
                break

    target_size = (imgsz, imgsz)  # (w, h)

    if src_path and src_path.is_file():
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka video: {src_path}")
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        cap.release()

    if not frames:
        # fallback image
        img = None
        for p in FALLBACK_IMAGES:
            if p.is_file():
                img = cv2.imread(str(p))
                break
        if img is None:
            # last resort: deterministic synthetic image
            rng = np.random.default_rng(0)
            img = rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        img = cv2.resize(img, target_size)
        frames = [img.copy() for _ in range(num_frames)]

    print(f"[INPUT] {len(frames)} frame siap (source={src_path or 'synthetic'}, {imgsz}x{imgsz})")
    return frames


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #
def benchmark_model(
    model_path: Path,
    frames: list[np.ndarray],
    warmup: int,
    device: str,
    name: str,
) -> OrderedDict[str, float]:
    """
    Run warm-up + timed inference, return per-stage stats.

    Speed fields from ultralytics Results.speed:
      preprocess, inference, postprocess  (ms)
    """
    model = YOLO(str(model_path))

    sums = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    n = len(frames)
    conf_sum = 0.0
    det_count = 0

    for i, img in enumerate(frames):
        results = model.predict(source=img, verbose=False, device=device, stream=False)
        r = results[0]

        if i >= warmup:  # measured runs
            for k in sums:
                sums[k] += float(r.speed.get(k, 0.0))
            if r.boxes is not None and len(r.boxes) > 0:
                det_count += int(len(r.boxes))
                conf_sum += float(r.boxes.conf.mean())

    measured = n - warmup
    if measured <= 0:
        raise ValueError("runs harus > warmup")

    stats = OrderedDict()
    stats["name"] = name
    stats["runs_measured"] = measured
    stats["preprocess_ms"] = sums["preprocess"] / measured
    stats["inference_ms"] = sums["inference"] / measured
    stats["postprocess_ms"] = sums["postprocess"] / measured
    stats["total_ms"] = (
        stats["preprocess_ms"] + stats["inference_ms"] + stats["postprocess_ms"]
    )
    stats["fps"] = 1000.0 / stats["total_ms"] if stats["total_ms"] > 0 else 0.0
    stats["avg_detections_per_frame"] = det_count / measured
    stats["avg_confidence"] = conf_sum / det_count if det_count else 0.0

    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        stats["gpu_mem_mb"] = float("nan")

    return stats


def print_comparison_table(pt_stats: dict, trt_stats: dict) -> None:
    """Print side-by-side table."""
    rows = [
        ("Runs (measured)", f"{pt_stats['runs_measured']}", f"{trt_stats['runs_measured']}"),
        ("Preprocess (ms)", f"{pt_stats['preprocess_ms']:.2f}", f"{trt_stats['preprocess_ms']:.2f}"),
        ("Inference (ms)", f"{pt_stats['inference_ms']:.2f}", f"{trt_stats['inference_ms']:.2f}"),
        ("Postprocess (ms)", f"{pt_stats['postprocess_ms']:.2f}", f"{trt_stats['postprocess_ms']:.2f}"),
        ("Total (ms)", f"{pt_stats['total_ms']:.2f}", f"{trt_stats['total_ms']:.2f}"),
        ("FPS", f"{pt_stats['fps']:.1f}", f"{trt_stats['fps']:.1f}"),
        ("Avg detections", f"{pt_stats['avg_detections_per_frame']:.1f}", f"{trt_stats['avg_detections_per_frame']:.1f}"),
        ("Avg confidence", f"{pt_stats['avg_confidence']:.3f}", f"{trt_stats['avg_confidence']:.3f}"),
        ("GPU mem (MB)", f"{pt_stats.get('gpu_mem_mb', 0):.0f}", f"{trt_stats.get('gpu_mem_mb', 0):.0f}"),
    ]

    col1 = max(len(r[0]) for r in rows + [("Metric", "", "")])
    col2 = max(len(r[1]) for r in rows + [("", pt_stats["name"], "")])
    col3 = max(len(r[2]) for r in rows + [("", "", trt_stats["name"])])

    header = f"{'Metric':<{col1}}  {pt_stats['name']:>{col2}}  {trt_stats['name']:>{col3}}"
    sep = f"{'-' * col1}  {'-' * col2}  {'-' * col3}"

    print("\n" + "=" * len(header))
    print("HASIL PERBANDINGAN  .pt  vs  TensorRT")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in rows:
        print(f"{r[0]:<{col1}}  {r[1]:>{col2}}  {r[2]:>{col3}}")

    speedup_ms = pt_stats["total_ms"] / trt_stats["total_ms"] if trt_stats["total_ms"] else float("inf")
    speedup_inf = pt_stats["inference_ms"] / trt_stats["inference_ms"] if trt_stats["inference_ms"] else float("inf")
    print(sep)
    print(f"Speedup (total)     : {speedup_ms:.2f}x")
    print(f"Speedup (inference) : {speedup_inf:.2f}x")
    print("=" * len(header) + "\n")


def save_csv(pt_stats: dict, trt_stats: dict, out_csv: Path) -> None:
    """Save per-backend stats into one CSV (two rows)."""
    fieldnames = [
        "timestamp",
        "backend",
        "model_path",
        "runs_measured",
        "preprocess_ms",
        "inference_ms",
        "postprocess_ms",
        "total_ms",
        "fps",
        "avg_detections_per_frame",
        "avg_confidence",
        "gpu_mem_mb",
    ]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for stats, backend in [(pt_stats, "pytorch"), (trt_stats, "tensorrt")]:
        row = {
            "timestamp": ts,
            "backend": backend,
            "model_path": stats.get("model_path", ""),
        }
        row.update({k: v for k, v in stats.items() if k in fieldnames})
        rows.append(row)

    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Saved -> {out_csv}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare .pt vs TensorRT engine")
    p.add_argument(
        "--model",
        default="ppe_compliance",
        help="Registry key (default) atau path file .pt",
    )
    p.add_argument("--source", default="", help="Video / image path (fallback ke C:/cctv-dev-temp)")
    p.add_argument("--imgsz", type=int, default=640, help="Input size (default 640)")
    p.add_argument("--runs", type=int, default=200, help="Total iterasi (incl. warmup)")
    p.add_argument("--warmup", type=int, default=20, help="Warm-up iters (default 20)")
    p.add_argument(
        "--quantize",
        type=int,
        default=0,
        choices=[0, 16],
        help="TensorRT quantize: 0=FP32 (default, no modelopt), 16=FP16 (butuh nvidia-modelopt; auto-fallback ke FP32 bila gagal)",
    )
    p.add_argument("--device", default="0", help="CUDA device id (default 0)")
    p.add_argument("--no-export", action="store_true", help="Export engine walau sudah ada")
    p.add_argument("--no-cache", action="store_true", help="Alias untuk --no-export")
    p.add_argument("--out", default="", help="Simpan CSV ke path ini")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    force_export = args.no_export or args.no_cache

    pt_path = resolve_pt(args.model)
    print(f"[MODEL] .pt = {pt_path}")

    engine_path = build_engine(
        pt_path,
        imgsz=args.imgsz,
        quantize=args.quantize,
        force=force_export,
    )

    frames = load_frames(args.source, num_frames=args.runs, imgsz=args.imgsz)

    print(f"\n[1/2] Benchmark .pt (PyTorch) ...")
    pt_stats = benchmark_model(
        pt_path, frames, warmup=args.warmup, device=args.device, name="PyTorch .pt"
    )
    pt_stats["model_path"] = str(pt_path)

    print(f"[2/2] Benchmark TensorRT (.engine) ...")
    engine_tag = engine_path.stem.split("_")[-1].upper()  # fp16 / fp32
    trt_stats = benchmark_model(
        engine_path, frames, warmup=args.warmup, device=args.device, name=f"TensorRT {engine_tag}"
    )
    trt_stats["model_path"] = str(engine_path)

    print_comparison_table(pt_stats, trt_stats)

    if args.out:
        save_csv(pt_stats, trt_stats, Path(args.out))


if __name__ == "__main__":
    main()
