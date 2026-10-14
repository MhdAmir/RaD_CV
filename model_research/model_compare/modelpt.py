r"""
Run .pt model (PyTorch) on camera/file with FPS + inference-time overlay
and CPU / GPU / RAM usage logging.

Optimized version: uses threaded capture to reduce I/O latency.

Usage:
    python modelpt.py --source rtsp://localhost:8554/stairs-cam
    python modelpt.py --source C:\cctv-dev-temp\stairs_people.mp4
    python modelpt.py --source 0                     # webcam
    python modelpt.py --model security_officer_v2 --log-csv pt_log.csv
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
#     "ppe_compliance": r"ppe_compliance_yolo26m\weights\best.pt",
#     "helmet_and_vest": r"helmet_and_vest_yolo26m\weights\best.pt",
#     "face_mask": r"face_mask_yolo26m\weights\best.pt",
#     "hand_in_pocket": r"hand_in_pocket_yolo26m\weights\best.pt",
#     "holding_phone": r"holding_phone_yolo26m\weights\best.pt",
#     "smartphone": r"smartphone_yolo26m\weights\best.pt",
#     "phone_usage": r"phone_usage_yolo26m\weights\best.pt",
    "people_counting": r"C:\CCTV_AI_Improvement\ai\research\runs\detect\phone_usage_yolo26m\weights\best.pt",
    # "security_officer_v2": r"security_officer_v2_yolo26m\weights\best.pt",
}
RUNS_DETECT = Path(r"C:\cctv-insight\ai\research\runs\detect")


def resolve_model(arg: str) -> Path:
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
    """Background thread for video reading to eliminate capture latency."""
    
    def __init__(self, source, imgsz=None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka source: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.source = source
        self.is_file = isinstance(source, str) and Path(source).is_file()
        self.imgsz = imgsz
        
        self.q = queue.Queue(maxsize=2)
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
            
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
    
    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


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
    ap = argparse.ArgumentParser(description="Run .pt model on camera/file with threaded capture")
    ap.add_argument("--model", default="people_counting", help="registry key atau path .pt")
    ap.add_argument("--source", default="", help="rtsp url | file | webcam index")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--log-csv", default="", help="opsional: tulis log ke CSV")
    ap.add_argument("--no-display", action="store_true", help="headless: tidak tampilkan window")
    ap.add_argument("--fast-display", action="store_true", help="disable plotting boxes untuk FPS lebih tinggi saat preview")
    ap.add_argument("--display-skip", type=int, default=1, help="update window setiap N frame (1=real-time, 3=33fps, 5=20fps)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop setelah N frame (0 = unlimited)")
    ap.add_argument("--no-thread", action="store_true", help="disable threaded capture (legacy mode)")
    ap.add_argument("--backend-tag", default="pt", help="label backend untuk log CSV")
    args = ap.parse_args()

    pt_path = resolve_model(args.model)
    source = resolve_source(args.source)
    print(f"[PT] model={pt_path.name}")
    print(f"[PT] source={source}")

    model = YOLO(str(pt_path))
    
    if args.no_thread:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Tidak bisa buka source: {source}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        threaded = None
    else:
        threaded = ThreadedCapture(source, imgsz=args.imgsz)
        cap = threaded

    mon = SystemMonitor(args.log_csv or None)
    times = deque(maxlen=30)
    frame_times = deque(maxlen=60)
    decode_times = deque(maxlen=60)
    prev = time.perf_counter()
    last_log = 0.0
    frame_count = 0

    backend_label = args.backend_tag
    print("Running... tekan 'q' untuk berhenti")
    try:
        while True:
            t_decode_start = time.perf_counter()
            ok, frame = cap.read() if args.no_thread else (True, cap.read())
            t_decode_end = time.perf_counter()
            decode_ms = (t_decode_end - t_decode_start) * 1000

            if not ok or frame is None:
                continue

            frame_count += 1
            decode_times.append(decode_ms)

            results = model.predict(source=frame.copy(), verbose=False, device=args.device,
                                    imgsz=args.imgsz, conf=args.conf, stream=False)
            r = results[0]
            inf_ms = float(r.speed.get("inference", 0))
            times.append(inf_ms)

            # Skip plotting kalau fast-display aktif (hemat ~5-10ms CPU)
            if args.fast_display:
                annotated = frame
            else:
                annotated = r.plot()

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
            elif frame_count % args.display_skip == 0:
                put_hud(annotated, fps, avg_inf, stats,
                        backend="PyTorch .pt" + (" [FAST]" if args.fast_display else ""),
                        decode_ms=avg_decode, queue_size=queue_size)
                cv2.namedWindow("modelpt.py", cv2.WINDOW_NORMAL)
                cv2.imshow("modelpt.py", annotated)
                show = cv2.waitKey(1) & 0xFF
            else:
                # Non-blocking check tanpa render
                show = cv2.waitKey(1) & 0xFF
            t_display_end = time.perf_counter()
            # display_times.append((t_display_end - t_display_start) * 1000)

            if now - last_log >= 1.0:
                print(f"[{frame_count:5d}] FPS {fps:5.1f} | inf {avg_inf:5.1f}ms | DecodeWait {avg_decode:5.1f}ms | "
                      f"CPU {stats['cpu_pct']:4.0f}% | RAM {stats['ram_mb']:6.0f}MB | "
                      f"GPU {stats['gpu_pct']:3d}% | VRAM {stats['gpu_mem_mb']:5.0f}MB | Q:{queue_size}")
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
