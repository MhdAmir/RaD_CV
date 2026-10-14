r"""
Run TensorRT .engine model on camera/file with FPS + inference-time overlay
and CPU / GPU / RAM usage logging.

Optimized version: uses threaded capture to reduce I/O latency.

Usage:
    python modelengine.py --source rtsp://localhost:8554/stairs-cam
    python modelengine.py --source C:\cctv-dev-temp\stairs_people.mp4
    python modelengine.py --quantize 16

Engine akan di-export otomatis jika belum ada (disimpan di samping .pt).
"""

import argparse
import csv
import queue
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
    # "ppe_compliance": r"ppe_compliance_yolo26m\weights\best.pt",
    # "helmet_and_vest": r"helmet_and_vest_yolo26m\weights\best.pt",
    # "face_mask": r"face_mask_yolo26m\weights\best.pt",
    # "hand_in_pocket": r"hand_in_pocket_yolo26m\weights\best.pt",
    # "holding_phone": r"holding_phone_yolo26m\weights\best.pt",
    # "smartphone": r"smartphone_yolo26m\weights\best.pt",
    # "phone_usage": r"phone_usage_yolo26m\weights\best.pt",
    "people_counting": r"C:\CCTV_AI_Improvement\ai\research\runs\detect\phone_usage_yolo26m\weights\best.pt",
    # "security_officer_v2": r"security_officer_v2_yolo26m\weights\best.pt",
}
RUNS_DETECT = Path(r"C:\cctv-insight\ai\research\runs\detect")


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
    raise FileNotFoundError(f"Model tidak ditemukan: {arg}")


def get_engine(pt_path: Path, imgsz: int, quantize: int, force: bool = False) -> Path:
    """Export engine jika perlu, lalu return path-nya."""
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    fp32_path = pt_path.with_name(f"{pt_path.stem}_fp32.engine")

    if not force:
        if engine_path.exists():
            print(f"[TRT] Engine ditemukan: {engine_path.name}")
            return engine_path
        if quantize == 16 and fp32_path.exists():
            print(f"[TRT] FP16 engine belum ada, reuse FP32: {fp32_path.name}")
            return fp32_path

    print(f"[TRT] Export engine ({'FP16' if quantize == 16 else 'FP32'}) ...")
    model = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, batch=1, simplify=True)
    if quantize == 16:
        kwargs["quantize"] = 16
    try:
        model.export(**kwargs)
        # .replace() menimpa file lama (rename() gagal di Windows bila target ada)
        pt_path.with_suffix(".engine").replace(engine_path)
    except Exception as e:
        if quantize == 16:
            print(f"[TRT] FP16 gagal di env ini ({e.__class__.__name__}).")
            print("[TRT] Catatan: FP16 membutuhkan nvidia-modelopt. Gunakan venv 3.12 (.venv-trt):")
            print(r"[TRT]    C:\cctv-insight\research_amir\.venv-trt\Scripts\python.exe modelengine.py ...")
            print("[TRT] Fallback otomatis ke FP32 di env ini ...")
            for junk in [pt_path.with_suffix(".engine"), pt_path.with_suffix(".onnx")]:
                if junk.exists():
                    try:
                        junk.unlink()
                    except OSError:
                        pass
            model.export(format="engine", imgsz=imgsz, batch=1, simplify=True)
            pt_path.with_suffix(".engine").replace(fp32_path)
            return fp32_path
        raise
    return engine_path


def resolve_source(src: str):
    s = src.strip()
    if s.isdigit():
        return int(s)
    if not s:
        default = Path(r"C:\cctv-dev-temp\stairs_people.mp4")
        if default.is_file():
            return str(default)
    return s or 0


class ThreadedCapture:
    """Background thread for video reading with aggressive frame dropping."""
    
    def __init__(self, source, imgsz=None, max_queue=1):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka source: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.source = source
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.imgsz = imgsz
        
        # max_queue=1 = always get latest frame, drop old ones
        self.q = queue.Queue(maxsize=max_queue)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            if self.imgsz:
                frame = cv2.resize(frame, (self.imgsz, self.imgsz))
            
            # Clear queue dan masukkan frame terbaru (low latency mode)
            while self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put(frame)
    
    def read(self, timeout=0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class DNNPreprocessor:
    """OpenCV DNN CUDA untuk preprocessing acceleration."""
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and hasattr(cv2.dnn, 'DNN_BACKEND_CUDA')
        if self.use_cuda:
            # Warm up CUDA DNN
            dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
            blob = cv2.dnn.blobFromImage(dummy)
            print("[DNN] CUDA preprocessing enabled")
    
    def preprocess(self, frame):
        """Fast BGR->RGB + resize + normalize using DNN"""
        # blobFromImage sudah handle resize + normalize + swap RB
        return cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1.0/255.0, 
            size=(640, 640), 
            mean=(0, 0, 0), 
            swapRB=True, 
            crop=False
        )


class SystemMonitor:
    def __init__(self, log_csv: str | None = None):
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.proc = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)
        self.csv_file = None
        self.writer = None
        if log_csv:
            fields = ["frame_id", "backend", "timestamp", "fps", "inference_ms",
                      "cpu_pct", "ram_mb", "gpu_pct", "gpu_mem_mb", "gpu_temp_c"]
            exists = Path(log_csv).exists()
            self.csv_file = open(log_csv, "a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
            if not exists:
                self.writer.writeheader()

    def read(self) -> dict:
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_mb": self.proc.memory_info().rss / 1e6,
            "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu,
            "gpu_mem_mb": pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 1e6,
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(self.gpu, 0),
        }

    def log_csv(self, frame_id, backend, fps, inf_ms, m):
        if self.writer:
            self.writer.writerow({
                "frame_id": frame_id,
                "backend": backend,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": f"{fps:.2f}", "inference_ms": f"{inf_ms:.2f}",
                "cpu_pct": f"{m['cpu_pct']:.1f}", "ram_mb": f"{m['ram_mb']:.1f}",
                "gpu_pct": f"{m['gpu_pct']}", "gpu_mem_mb": f"{m['gpu_mem_mb']:.1f}",
                "gpu_temp_c": f"{m['gpu_temp_c']}",
            })
            self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        pynvml.nvmlShutdown()


def put_hud(img, fps, inf_ms, stats, backend, decode_ms, queue_size):
    y, dy = 28, 24
    cv2.putText(img, f"{backend} | FPS {fps:>5.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2); y += dy
    cv2.putText(img, f"Inference {inf_ms:>5.1f} ms", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); y += dy
    cv2.putText(img, f"Decode+Wait {decode_ms:>5.1f} ms | Q:{queue_size}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2); y += dy
    cv2.putText(img, f"CPU {stats['cpu_pct']:>4.0f}%  RAM {stats['ram_mb']:>6.0f}MB", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2); y += dy
    cv2.putText(img, f"GPU {stats['gpu_pct']:>3d}%  VRAM {stats['gpu_mem_mb']:>5.0f}MB  {stats['gpu_temp_c']}C",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2)


def main():
    ap = argparse.ArgumentParser(description="Run TensorRT engine on camera/file with threaded capture")
    ap.add_argument("--model", default="people_counting", help="registry key atau path .pt / .engine")
    ap.add_argument("--source", default="", help="rtsp url | file | webcam index")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16],
                    help="0=FP32 (default), 16=FP16 (auto-fallback ke FP32)")
    ap.add_argument("--log-csv", default="", help="opsional: tulis log ke CSV")
    ap.add_argument("--no-display", action="store_true", help="headless: tidak tampilkan window")
    ap.add_argument("--fast-display", action="store_true", help="disable plotting boxes untuk FPS lebih tinggi saat preview")
    ap.add_argument("--display-skip", type=int, default=1, help="update window setiap N frame (1=real-time, 3=33fps, 5=20fps)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop setelah N frame (0 = unlimited)")
    ap.add_argument("--no-thread", action="store_true", help="disable threaded capture (legacy mode)")
    ap.add_argument("--low-latency", action="store_true", help="skip frames jika display lambat (real-time priority)")
    ap.add_argument("--dnn-preprocess", action="store_true", help="gunakan OpenCV DNN CUDA untuk preprocessing")
    ap.add_argument("--backend-tag", default="engine", help="label backend untuk log CSV")
    args = ap.parse_args()

    # terima path .engine langsung
    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    source = resolve_source(args.source)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[TRT] source={source}")

    model = YOLO(str(engine_path))
    
    if args.no_thread:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka source: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threaded = None
    else:
        # Low latency = queue size 1 (drop old frames), display_skip lebih agresif
        queue_size = 1 if args.low_latency else 2
        threaded = ThreadedCapture(source, imgsz=args.imgsz, max_queue=queue_size)
        cap = threaded

    mon = SystemMonitor(args.log_csv or None)
    dnn_pre = DNNPreprocessor(use_cuda=args.dnn_preprocess) if args.dnn_preprocess else None
    times = deque(maxlen=30)
    frame_times = deque(maxlen=60)
    decode_times = deque(maxlen=60)
    prev = time.perf_counter()
    last_log = 0.0
    frame_count = 0

    backend_label = args.backend_tag
    print("Running... tekan 'q' untuk berhenti")
    inference_times = deque(maxlen=100)
    plot_times = deque(maxlen=100)
    display_times = deque(maxlen=100)
    try:
        while True:
            t_total_start = time.perf_counter()

            t_decode_start = time.perf_counter()
            if args.no_thread:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
            else:
                # Low latency mode: shorter timeout, more aggressive skipping
                timeout = 0.01 if args.low_latency else 1.0
                frame = cap.read(timeout=timeout)
            t_decode_end = time.perf_counter()
            decode_ms = (t_decode_end - t_decode_start) * 1000

            if frame is None:
                continue

            frame_count += 1
            decode_times.append(decode_ms)

            # Inference timing
            t_infer_start = time.perf_counter()
            results = model.predict(source=frame.copy(), verbose=False, device=args.device,
                                    imgsz=args.imgsz, conf=args.conf, stream=False)
            t_infer_end = time.perf_counter()

            r = results[0]
            inf_ms = float(r.speed.get("inference", 0))
            times.append(inf_ms)
            inference_times.append((t_infer_end - t_infer_start) * 1000)

            # Plot timing
            t_plot_start = time.perf_counter()
            if args.fast_display:
                annotated = frame
            else:
                annotated = r.plot()
            t_plot_end = time.perf_counter()
            plot_times.append((t_plot_end - t_plot_start) * 1000)

            now = time.perf_counter()
            frame_times.append((now - prev) * 1000)
            prev = now
            fps = 1000.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            avg_inf = sum(times) / len(times)
            avg_decode = sum(decode_times) / len(decode_times)

            stats = mon.read()
            queue_size = threaded.q.qsize() if threaded else 0

            # Display timing — skip frame jika perlu untuk kurangi overhead
            t_display_start = time.perf_counter()
            if args.no_display:
                show = None
            elif args.low_latency:
                # Low latency: display setiap 3 frame, non-blocking
                if frame_count % 3 == 0:
                    put_hud(annotated, fps, avg_inf, stats,
                            backend="TensorRT .engine [LATENCY]",
                            decode_ms=avg_decode, queue_size=queue_size)
                    cv2.imshow("modelengine.py", annotated)
                show = cv2.waitKey(1) & 0xFF
            elif frame_count % args.display_skip == 0:
                put_hud(annotated, fps, avg_inf, stats,
                        backend="TensorRT .engine" + (" [FAST]" if args.fast_display else ""),
                        decode_ms=avg_decode, queue_size=queue_size)
                cv2.namedWindow("modelengine.py", cv2.WINDOW_NORMAL)
                cv2.imshow("modelengine.py", annotated)
                show = cv2.waitKey(1) & 0xFF
            else:
                # Non-blocking check tanpa render
                show = cv2.waitKey(1) & 0xFF
            t_display_end = time.perf_counter()
            display_times.append((t_display_end - t_display_start) * 1000)

            if now - last_log >= 1.0:
                avg_wall_inf = sum(inference_times) / len(inference_times) if inference_times else 0
                avg_plot = sum(plot_times) / len(plot_times) if plot_times else 0
                avg_display = sum(display_times) / len(display_times) if display_times else 0

                print(f"[{frame_count:5d}] FPS {fps:5.1f} | "
                      f"inf_gpu {avg_inf:5.1f}ms | wall_inf {avg_wall_inf:5.1f}ms | "
                      f"plot {avg_plot:5.1f}ms | disp {avg_display:5.1f}ms | "
                      f"CPU {stats['cpu_pct']:3.0f}% | GPU {stats['gpu_pct']:3.0f}% | T:{stats['gpu_temp_c']}C")
                mon.log_csv(frame_count, backend_label, fps, avg_inf, stats)
                last_log = now

            if (show is not None and show == ord("q")) or (args.max_frames and frame_count >= args.max_frames):
                break
    finally:
        mon.close()
        if threaded:
            threaded.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
