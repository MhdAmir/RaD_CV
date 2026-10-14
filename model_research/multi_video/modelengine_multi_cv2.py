r"""
Multi-Stream dengan OpenCV cv2.VideoCapture (Tanpa GStreamer)

- 4 streams decode paralel dengan threading
- TensorRT inference per frame
- Display dengan timing konsisten untuk smooth playback

Usage:
  python modelengine_multi_cv2.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
import csv
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
    "helmet_and_vest": r"helmet_and_vest_yolo26m\weights\best.pt",
    "face_mask": r"face_mask_yolo26m\weights\best.pt",
    "hand_in_pocket": r"hand_in_pocket_yolo26m\weights\best.pt",
    "holding_phone": r"holding_phone_yolo26m\weights\best.pt",
    "smartphone": r"smartphone_yolo26m\weights\best.pt",
    "phone_usage": r"phone_usage_yolo26m\weights\best.pt",
    "people_counting": r"people_counting_yolo26m\weights\best.pt",
    "security_officer_v2": r"security_officer_v2_yolo26m\weights\best.pt",
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
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    fp32_path = pt_path.with_name(f"{pt_path.stem}_fp32.engine")

    if not force:
        if engine_path.exists():
            print(f"[TRT] Engine found: {engine_path.name}")
            return engine_path
        if quantize == 16 and fp32_path.exists():
            print(f"[TRT] FP16 not found, reuse FP32: {fp32_path.name}")
            return fp32_path

    print(f"[TRT] Exporting engine ({'FP16' if quantize == 16 else 'FP32'}) ...")
    model = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, batch=1, simplify=True)
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
            model.export(format="engine", imgsz=imgsz, batch=1, simplify=True)
            pt_path.with_suffix(".engine").replace(fp32_path)
            return fp32_path
        raise
    return engine_path


class Cv2Capture:
    """
    Threaded capture per stream dengan cv2.VideoCapture.
    Auto-restart untuk video files (loop).
    """
    def __init__(self, source, stream_id, width=640, height=640):
        self.source = source
        self.stream_id = stream_id
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
        # Start capture thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        """Continuous capture dengan auto-restart untuk video files."""
        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[CV2-{self.stream_id}] Failed to open {self.source}")
                time.sleep(1)
                continue
            
            # Set properties untuk performa optimal
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            frame_count = 0
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    # EOF - restart (loop video)
                    cap.release()
                    print(f"[CV2-{self.stream_id}] Loop...")
                    time.sleep(0.1)
                    break
                
                frame_count += 1
                
                # Resize jika perlu
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Update frame buffer
                with self.lock:
                    self.frame = frame.copy()
            
            cap.release()
    
    def read(self):
        """Get latest frame."""
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


class MultiStreamCapture:
    """Multi-stream manager dengan cv2.VideoCapture per stream."""
    def __init__(self, sources, width=640, height=640):
        self.captures = []
        self.latest_frames = {}
        
        for i, source in enumerate(sources):
            cap = Cv2Capture(source, i, width, height)
            self.captures.append(cap)
            self.latest_frames[i] = None
            print(f"[STREAM-{i}] {source}")
        
        # Thread untuk update latest_frames
        self.running = True
        self.thread = threading.Thread(target=self._update_latest, daemon=True)
        self.thread.start()
    
    def _update_latest(self):
        while self.running:
            for i, cap in enumerate(self.captures):
                ok, frame = cap.read()
                if ok:
                    self.latest_frames[i] = frame
            time.sleep(0.001)
    
    def get_latest_frames(self):
        return self.latest_frames.copy()
    
    def release(self):
        self.running = False
        for cap in self.captures:
            cap.stop()


class SystemMonitor:
    def __init__(self, log_csv=None):
        pynvml.nvmlInit()
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.proc = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)
        self.csv_file = None
        self.writer = None
        if log_csv:
            self.csv_file = open(log_csv, "a", newline="", encoding="utf-8")
            fields = ["timestamp", "fps", "inference_ms", "cpu_pct", "ram_mb",
                      "gpu_pct", "gpu_mem_mb", "gpu_temp_c", "streams"]
            if not Path(log_csv).exists():
                self.writer = csv.DictWriter(self.csv_file, fieldnames=fields)
                self.writer.writeheader()

    def read(self):
        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_mb": self.proc.memory_info().rss / 1e6,
            "gpu_pct": pynvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu,
            "gpu_mem_mb": pynvml.nvmlDeviceGetMemoryInfo(self.gpu).used / 1e6,
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(self.gpu, 0),
        }

    def log(self, fps, inf_ms, streams, m):
        if self.writer:
            self.writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": f"{fps:.1f}",
                "inference_ms": f"{inf_ms:.1f}",
                "streams": streams,
                "cpu_pct": f"{m['cpu_pct']:.0f}",
                "ram_mb": f"{m['ram_mb']:.1f}",
                "gpu_pct": f"{m['gpu_pct']}",
                "gpu_mem_mb": f"{m['gpu_mem_mb']:.1f}",
                "gpu_temp_c": f"{m['gpu_temp_c']}",
            })
            self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        pynvml.nvmlShutdown()


def display_thread(cap, model, args, stop_event):
    """Dedicated display thread dengan throttle FPS."""
    # Create windows
    for i in range(len(args.sources)):
        cv2.namedWindow(f"Stream {i}", cv2.WINDOW_NORMAL)
    
    last_display = {i: time.time() for i in range(len(args.sources))}
    target_fps = args.display_fps
    
    frame_counts = {i: 0 for i in range(len(args.sources))}
    
    while not stop_event.is_set():
        frames = cap.get_latest_frames()
        
        for stream_id, frame in frames.items():
            if frame is None:
                continue
            
            # Throttle display
            now = time.time()
            if (now - last_display[stream_id]) < (1.0 / target_fps):
                continue
            
            # Inference + draw
            try:
                results = model.predict(
                    source=frame,
                    verbose=False,
                    device=args.device,
                    imgsz=args.imgsz,
                    conf=args.conf
                )
                annotated = results[0].plot()
                frame_counts[stream_id] += 1
            except Exception as e:
                annotated = frame
            
            # Draw stream label
            cv2.putText(annotated, f"Stream {stream_id}: {frame_counts[stream_id]} frames",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(f"Stream {stream_id}", annotated)
            last_display[stream_id] = now
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
    
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Multi-Stream dengan OpenCV (No GStreamer)")
    ap.add_argument("--model", default="people_counting")
    ap.add_argument("--sources", nargs="+", default=[r"C:\cctv-dev-temp\stairs_people.mp4"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--display-fps", type=int, default=24, help="Target display FPS")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-csv", default="")
    args = ap.parse_args()

    # Resolve model
    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    num_streams = len(args.sources)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[CV2] Streams: {num_streams}")
    print(f"[DISPLAY] Target FPS: {args.display_fps}")

    # Load model
    model = YOLO(str(engine_path))

    # Start capture
    print("[CAPTURE] Starting cv2.VideoCapture streams...")
    cap = MultiStreamCapture(args.sources, args.imgsz, args.imgsz)
    
    time.sleep(0.5)  # Wait for first frames

    stop_event = threading.Event()
    
    # Start display thread
    d_thread = threading.Thread(target=display_thread, args=(cap, model, args, stop_event), daemon=True)
    d_thread.start()

    mon = SystemMonitor(args.log_csv or None)
    inf_times = deque(maxlen=50)
    frame_count = 0
    start_time = time.time()

    print("Running... press 'q' to stop")
    try:
        while not stop_event.is_set():
            # Count frames processed
            frames = cap.get_latest_frames()
            for stream_id, frame in frames.items():
                if frame is not None:
                    frame_count += 1
                    
                    # Quick inference timing sample
                    t_inf = time.time()
                    results = model.predict(
                        source=frame,
                        verbose=False,
                        device=args.device,
                        imgsz=args.imgsz,
                        conf=args.conf
                    )
                    inf_ms = (time.time() - t_inf) * 1000
                    inf_times.append(inf_ms)
                    break  # Sample one frame per loop
            
            # Stats per 60 frames
            if frame_count % 60 == 0 and frame_count > 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                avg_inf = sum(inf_times) / len(inf_times) if inf_times else 0
                stats = mon.read()
                
                print(f"[{frame_count:4d}] FPS ~{fps:5.1f} | inf {avg_inf:5.1f}ms | "
                      f"GPU {stats['gpu_pct']:3.0f}% | VRAM {stats['gpu_mem_mb']:5.0f}MB")
                
                mon.log(fps, avg_inf, num_streams, stats)

            if args.max_frames and frame_count >= args.max_frames:
                stop_event.set()
                break
            
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        mon.close()
        stop_event.set()
        d_thread.join(timeout=2)
        cap.release()
        print(f"\nStopped. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
