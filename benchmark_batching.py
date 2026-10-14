r"""
Minimal benchmark to compare:

A) 4 independent TensorRT inference pipelines (batch=1)
B) 1 batched TensorRT inference pipeline over 4 cameras (batch=4/8)

This script uses the existing YOLO TensorRT engine and software batching that
mimics DeepStream nvstreammux + nvinfer behavior.  DeepStream native plugins are
not installed on this Windows host, so this is the closest measurable equivalent.

Metrics recorded:
- Total FPS
- FPS per camera
- Average inference latency (wall + GPU-reported)
- GPU utilization
- GPU memory usage
- CPU utilization

Usage:
    python benchmark_batching.py --scenario independent --max-frames 500
    python benchmark_batching.py --scenario batched --batch-size 4 --max-frames 500
    python benchmark_batching.py --scenario all --max-frames 500
"""

import argparse
import csv
import json
import math
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import psutil
import pynvml
from ultralytics import YOLO

MODELS = {
    "ppe_compliance": r"ppe_compliance_yolo26m\weights\best.pt",
    "people_counting": r"people_counting_yolo26m\weights\best.pt",
}
RUNS_DETECT = Path(r"C:\cctv-insight\ai\research\runs\detect")
GST_BIN = r"C:\Users\istd.magang17\AppData\Local\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"
DEFAULT_SOURCE = r"C:\cctv-dev-temp\stairs_people.mp4"
RESULT_DIR = Path(r"C:\cctv-insight\research_amir\benchmark_results")


def resolve_pt(arg: str) -> Path:
    p = Path(arg)
    if p.suffix.lower() == ".pt" and p.is_file():
        return p
    key = arg.lower()
    if key in MODELS:
        cand = RUNS_DETECT / MODELS[key]
        if cand.is_file():
            return cand
    cand = RUNS_DETECT / arg / "weights" / "best.pt"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"Model not found: {arg}")


def get_engine(pt_path: Path, imgsz: int, quantize: int, batch: int = 1, force: bool = False) -> Path:
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}_bs{batch}.engine")
    generic_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    fp32_path = pt_path.with_name(f"{pt_path.stem}_fp32.engine")

    if not force:
        if engine_path.exists():
            print(f"[TRT] Engine found: {engine_path.name}")
            return engine_path
        if generic_path.exists():
            print(f"[TRT] Batched engine not found, using generic: {generic_path.name}")
            return generic_path
        if quantize == 16 and fp32_path.exists():
            print(f"[TRT] FP16 engine not found, using FP32: {fp32_path.name}")
            return fp32_path

    print(f"[TRT] Export engine ({'FP16' if quantize == 16 else 'FP32'}, batch={batch}) ...")
    model = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, batch=batch, simplify=True)
    if quantize == 16:
        kwargs["quantize"] = 16
    try:
        model.export(**kwargs)
        pt_path.with_suffix(".engine").replace(engine_path)
    except Exception as e:
        if quantize == 16:
            print(f"[TRT] FP16 failed ({e.__class__.__name__}), fallback FP32 ...")
            for junk in [pt_path.with_suffix(".engine"), pt_path.with_suffix(".onnx")]:
                if junk.exists():
                    try:
                        junk.unlink()
                    except OSError:
                        pass
            model.export(format="engine", imgsz=imgsz, batch=batch, simplify=True)
            pt_path.with_suffix(".engine").replace(fp32_path)
            return fp32_path
        raise
    return engine_path


class GstNvdecCapture:
    """Single GStreamer NVDEC reader (one camera)."""

    def __init__(self, source: str, width: int = 640, height: int = 640):
        self.width = width
        self.height = height
        self.source = source
        self.frame_size = width * height * 3
        self.process = None
        self.running = False
        self.frame_buffer = None
        self.lock = threading.Lock()
        is_file = Path(source).is_file() if isinstance(source, str) else False

        if is_file:
            self.pipeline_args = [
                GST_BIN, "-q",
                "filesrc", f"location={source}",
                "!", "qtdemux",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=RGB",
                "!", "fdsink", "fd=1",
            ]
        else:
            self.pipeline_args = [
                GST_BIN, "-q",
                "rtspsrc", f"location={source}", "latency=0",
                "!", "rtph264depay",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=RGB",
                "!", "fdsink", "fd=1",
            ]

        self.process = subprocess.Popen(
            self.pipeline_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=self.frame_size * 2,
        )
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        while self.running:
            data = self.process.stdout.read(self.frame_size)
            if not data or len(data) != self.frame_size:
                # Restart file sources to keep the benchmark running
                if self.running and Path(self.source).is_file():
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=2)
                    except Exception:
                        pass
                    self.process = subprocess.Popen(
                        self.pipeline_args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        bufsize=self.frame_size * 2,
                    )
                    continue
                break
            frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.frame_buffer = frame
        self.running = False

    def read(self, timeout: float = 1.0):
        start = time.time()
        while self.running and (time.time() - start) < timeout:
            with self.lock:
                if self.frame_buffer is not None:
                    frame = self.frame_buffer.copy()
                    self.frame_buffer = None
                    return True, frame
            time.sleep(0.001)
        return False, None

    def release(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
        self.thread.join(timeout=1)


class SystemMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.proc = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)

    def read(self):
        try:
            gpu_pct = pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 1e6
            gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu, 0)
        except pynvml.NVMLError:
            gpu_pct = 0
            gpu_mem = 0
            gpu_temp = 0
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_mb": self.proc.memory_info().rss / 1e6,
            "gpu_pct": gpu_pct,
            "gpu_mem_mb": gpu_mem,
            "gpu_temp_c": gpu_temp,
        }

    def close(self):
        pynvml.nvmlShutdown()


def warmup_model(model: YOLO, imgsz: int, batch: int, device: str):
    for _ in range(3):
        dummy = np.zeros((batch, imgsz, imgsz, 3), dtype=np.uint8)
        try:
            model.predict(source=dummy, verbose=False, device=device, imgsz=imgsz)
        except Exception:
            for _ in range(3):
                model.predict(source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                             verbose=False, device=device, imgsz=imgsz)


def benchmark_independent(sources, model_path, imgsz, conf, device, max_frames, log_csv=None):
    """Run N independent TensorRT pipelines, each with its own YOLO model."""
    n = len(sources)
    print(f"\n=== SCENARIO A: INDEPENDENT ({n} pipelines, batch=1) ===")
    print(f"[TRT] engine={Path(model_path).name}")

    caps = [GstNvdecCapture(s, imgsz, imgsz) for s in sources]
    models = [YOLO(str(model_path)) for _ in range(n)]
    for m in models:
        warmup_model(m, imgsz, 1, device)

    mon = SystemMonitor()
    inf_times = [deque(maxlen=100) for _ in range(n)]
    wall_inf_times = [deque(maxlen=100) for _ in range(n)]
    frame_counts = [0] * n
    start_time = time.time()
    stats_log = []
    last_report = start_time

    try:
        while sum(frame_counts) < max_frames:
            for i in range(n):
                ok, frame = caps[i].read(timeout=0.05)
                if not ok or frame is None:
                    continue
                frame_counts[i] += 1

                t0 = time.perf_counter()
                res = models[i].predict(source=frame, verbose=False, device=device,
                                        imgsz=imgsz, conf=conf)
                wall_ms = (time.perf_counter() - t0) * 1000
                gpu_ms = float(res[0].speed.get("inference", 0))
                inf_times[i].append(gpu_ms)
                wall_inf_times[i].append(wall_ms)

            now = time.time()
            if now - last_report >= 1.0:
                stats = mon.read()
                total = sum(frame_counts)
                elapsed = now - start_time
                fps_total = total / elapsed if elapsed else 0
                per_cam = [c / elapsed if elapsed else 0 for c in frame_counts]
                avg_inf = sum(sum(q) / len(q) for q in inf_times if q) / n
                avg_wall = sum(sum(q) / len(q) for q in wall_inf_times if q) / n
                print(f"[INDEP] frames={total:4d} total_fps={fps_total:5.1f} "
                      f"per_cam={[f'{x:.1f}' for x in per_cam]} "
                      f"avg_gpu_inf={avg_inf:5.1f}ms avg_wall_inf={avg_wall:5.1f}ms "
                      f"CPU {stats['cpu_pct']:3.0f}% GPU {stats['gpu_pct']:3d}% "
                      f"VRAM {stats['gpu_mem_mb']:6.0f}MB")
                stats_log.append({"time": now - start_time, "fps_total": fps_total,
                                  "per_cam": per_cam, "avg_gpu_inf": avg_inf,
                                  "avg_wall_inf": avg_wall, **stats})
                last_report = now
    finally:
        for c in caps:
            c.release()
        mon.close()

    return aggregate_metrics("independent", 1, frame_counts, start_time,
                             inf_times, wall_inf_times, mon, stats_log, log_csv)


def benchmark_batched(sources, model_path, imgsz, conf, device, max_frames,
                      batch_size=4, push_timeout_ms=40000, log_csv=None):
    """nvstreammux-style gather + nvinfer-style batched inference."""
    n = len(sources)
    print(f"\n=== SCENARIO B: BATCHED mux-n-infer ({n} sources, batch={batch_size}, timeout={push_timeout_ms}us) ===")
    print(f"[TRT] engine={Path(model_path).name}")

    caps = [GstNvdecCapture(s, imgsz, imgsz) for s in sources]
    model = YOLO(str(model_path))
    warmup_model(model, imgsz, batch_size, device)

    mon = SystemMonitor()
    batch_wall_times = deque(maxlen=100)
    batch_gpu_times = deque(maxlen=100)
    frame_counts = [0] * n
    total_inferred = 0
    start_time = time.time()
    stats_log = []
    last_report = start_time
    push_timeout_s = push_timeout_ms / 1_000_000.0

    try:
        pending = {i: None for i in range(n)}
        while total_inferred < max_frames:
            # Gather up to one fresh frame per camera
            for i in range(n):
                if pending[i] is None:
                    ok, frame = caps[i].read(timeout=0.02)
                    if ok and frame is not None:
                        pending[i] = frame

            # Collect available frames, wait for timeout or full batch
            batch_frames = []
            batch_ids = []
            deadline = time.perf_counter() + push_timeout_s
            while len(batch_frames) < batch_size:
                # Add whatever is already available
                for i in range(n):
                    if pending[i] is not None and i not in batch_ids:
                        batch_frames.append(pending[i])
                        batch_ids.append(i)
                        if len(batch_frames) == batch_size:
                            break
                if len(batch_frames) == batch_size:
                    break
                if time.perf_counter() >= deadline:
                    break
                # tiny sleep to avoid spinning too hard
                time.sleep(0.001)

            if not batch_frames:
                continue

            # Pad to full batch if dynamic batching timeout fired early
            while len(batch_frames) < batch_size:
                batch_frames.append(batch_frames[-1])
                batch_ids.append(-1)  # padding marker

            t0 = time.perf_counter()
            res = model.predict(source=np.stack(batch_frames), verbose=False,
                                device=device, imgsz=imgsz, conf=conf)
            wall_ms = (time.perf_counter() - t0) * 1000
            batch_wall_times.append(wall_ms)

            # Ultralytics reports per-image inference speed in ms
            for r in res:
                total_inferred += 1
                batch_gpu_times.append(float(r.speed.get("inference", 0)))

            for i in batch_ids:
                if i >= 0:
                    frame_counts[i] += 1
                    pending[i] = None

            now = time.time()
            if now - last_report >= 1.0:
                stats = mon.read()
                elapsed = now - start_time
                fps_total = total_inferred / elapsed if elapsed else 0
                per_cam = [frame_counts[i] / elapsed if elapsed else 0 for i in range(n)]
                avg_batch_wall = sum(batch_wall_times) / len(batch_wall_times)
                avg_gpu = sum(batch_gpu_times) / len(batch_gpu_times)
                avg_per_frame = avg_batch_wall / batch_size
                print(f"[BATCH] inferred={total_inferred:4d} batch={len(batch_ids)} total_fps={fps_total:5.1f} "
                      f"per_cam={[f'{x:.1f}' for x in per_cam]} "
                      f"batch_wall={avg_batch_wall:5.1f}ms avg_gpu_inf={avg_gpu:5.1f}ms "
                      f"per_frame_wall={avg_per_frame:5.1f}ms "
                      f"CPU {stats['cpu_pct']:3.0f}% GPU {stats['gpu_pct']:3d}% "
                      f"VRAM {stats['gpu_mem_mb']:6.0f}MB")
                stats_log.append({"time": now - start_time, "fps_total": fps_total,
                                  "per_cam": per_cam, "avg_batch_wall": avg_batch_wall,
                                  "avg_gpu_inf": avg_gpu, "avg_per_frame_wall": avg_per_frame, **stats})
                last_report = now

    finally:
        for c in caps:
            c.release()
        mon.close()

    return aggregate_metrics("batched", batch_size, frame_counts, start_time,
                             batch_gpu_times, batch_wall_times, mon, stats_log, log_csv)


def aggregate_metrics(mode, batch_size, frame_counts, start_time, inf_times, wall_times, mon, stats_log, log_csv):
    """Compute summary metrics for a scenario."""
    total_frames = sum(frame_counts)
    elapsed = time.time() - start_time
    total_fps = total_frames / elapsed if elapsed else 0
    per_cam_fps = [c / elapsed if elapsed else 0 for c in frame_counts]

    if isinstance(inf_times[0], deque):
        all_inf = [x for q in inf_times for x in q]
        all_wall = [x for q in wall_times for x in q]
    else:
        all_inf = list(inf_times)
        all_wall = list(wall_times)

    avg_gpu_inf = sum(all_inf) / len(all_inf) if all_inf else 0
    avg_wall_inf = sum(all_wall) / len(all_wall) if all_wall else 0

    # Resource averages over the log
    if stats_log:
        avg_cpu = sum(s["cpu_pct"] for s in stats_log) / len(stats_log)
        avg_gpu = sum(s["gpu_pct"] for s in stats_log) / len(stats_log)
        avg_mem = sum(s["gpu_mem_mb"] for s in stats_log) / len(stats_log)
        peak_mem = max(s["gpu_mem_mb"] for s in stats_log)
    else:
        avg_cpu = avg_gpu = avg_mem = peak_mem = 0

    row = {
        "mode": mode,
        "batch_size": batch_size,
        "total_fps": total_fps,
        "per_cam_fps": per_cam_fps,
        "avg_per_cam_fps": sum(per_cam_fps) / len(per_cam_fps) if per_cam_fps else 0,
        "avg_gpu_inf_ms": avg_gpu_inf,
        "avg_wall_inf_ms": avg_wall_inf,
        "cpu_pct": avg_cpu,
        "gpu_pct": avg_gpu,
        "gpu_mem_mb": avg_mem,
        "gpu_mem_peak_mb": peak_mem,
        "total_frames": total_frames,
        "elapsed_s": elapsed,
    }

    if log_csv:
        write_csv(log_csv, row)
    return row


def write_csv(path: str, row: dict):
    """Append a summary row to CSV."""
    file_path = Path(path)
    exists = file_path.exists()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        # Flatten per_cam list
        flat = row.copy()
        flat["per_cam_fps"] = "|".join(f"{x:.2f}" for x in flat["per_cam_fps"])
        writer = csv.DictWriter(f, fieldnames=flat.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def print_table(rows):
    print("\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    hdr = f"{'Mode':<16} {'Batch':>5} {'Total FPS':>10} {'Per-Cam FPS':>13} {'Avg Latency':>13} {'GPU %':>6} {'VRAM MB':>9} {'CPU %':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mode_label = ("Independent" if r["mode"] == "independent" else f"Batched bs={r['batch_size']}")
        per_cam = f"{r['avg_per_cam_fps']:.1f}"
        lat = r["avg_wall_inf_ms"]
        print(f"{mode_label:<16} {r['batch_size']:>5} {r['total_fps']:>10.1f} {per_cam:>13} {lat:>12.1f}ms {r['gpu_pct']:>5.0f}% {r['gpu_mem_mb']:>8.0f} {r['cpu_pct']:>5.0f}%")
    print("=" * 110)


def main():
    ap = argparse.ArgumentParser(description="Batching benchmark: independent vs batched TensorRT")
    ap.add_argument("--scenario", default="all", choices=["independent", "batched", "all"])
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=16, choices=[0, 16])
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--push-timeout-us", type=int, default=40000)
    ap.add_argument("--result-dir", default=str(RESULT_DIR))
    ap.add_argument("--export-batch-engines", action="store_true",
                    help="Pre-export batch=4/8 engines before benchmarking")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not Path(GST_BIN).exists():
        print(f"[ERROR] GStreamer not found: {GST_BIN}")
        return

    sources = [args.source] * args.streams
    print(f"[BENCH] sources={sources}")

    # Resolve base .pt
    pt_path = resolve_pt(args.model)

    # For a fair comparison we need engines with matching batch sizes.
    # Independent uses batch=1. Batched uses 4 or 8.
    if args.export_batch_engines:
        for bs in [1, 4, 8]:
            get_engine(pt_path, args.imgsz, args.quantize, batch=bs, force=True)

    engine_independent = get_engine(pt_path, args.imgsz, args.quantize, batch=1)

    rows = []

    if args.scenario in ("independent", "all"):
        log = result_dir / "independent.csv"
        rows.append(benchmark_independent(sources, engine_independent, args.imgsz, args.conf,
                                            args.device, args.max_frames, log_csv=str(log)))

    if args.scenario in ("batched", "all"):
        # Configurations requested
        configs = []
        if args.scenario == "batched":
            configs = [(args.batch_size, args.push_timeout_us)]
        else:
            configs = [(1, 40000), (4, 40000), (8, 40000)]
        for bs, timeout_us in configs:
            engine = get_engine(pt_path, args.imgsz, args.quantize, batch=bs)
            log = result_dir / f"batched_bs{bs}.csv"
            rows.append(benchmark_batched(sources, engine, args.imgsz, args.conf,
                                          args.device, args.max_frames,
                                          batch_size=bs, push_timeout_ms=timeout_us,
                                          log_csv=str(log)))

    print_table(rows)

    summary = {
        "args": vars(args),
        "results": rows,
    }
    summary_path = result_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[SUMMARY] saved {summary_path}")


if __name__ == "__main__":
    main()
