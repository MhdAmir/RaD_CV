r"""
TensorRT Multi-Stream dengan Grid Display (2x2 atau NxM)

Show semua streams dalam 1 window grid untuk monitoring.

Usage:
  python modelengine_batch_display.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
import csv
import multiprocessing as mp
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
GST_BIN = r"C:\Users\istd.magang17\AppData\Local\Programs\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe"


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
            print(f"[TRT] Engine ditemukan: {engine_path.name}")
            return engine_path
        if quantize == 16 and fp32_path.exists():
            print(f"[TRT] FP16 belum ada, reuse FP32: {fp32_path.name}")
            return fp32_path

    print(f"[TRT] Export engine ({'FP16' if quantize == 16 else 'FP32'}) ...")
    model = YOLO(str(pt_path))
    kwargs = dict(format="engine", imgsz=imgsz, batch=1, simplify=True)
    if quantize == 16:
        kwargs["quantize"] = 16
    try:
        model.export(**kwargs)
        pt_path.with_suffix(".engine").replace(engine_path)
    except Exception as e:
        if quantize == 16:
            print(f"[TRT] FP16 gagal ({e.__class__.__name__}), fallback FP32 ...")
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


def gst_reader_process(source, width, height, queue, stop_event, stream_id):
    """Dedicated process untuk GStreamer decode dengan auto-restart."""
    frame_size = width * height * 3
    is_file = Path(source).is_file() if isinstance(source, str) else False
    
    def get_args():
        if is_file:
            return [
                GST_BIN, "-q",
                "filesrc", f"location={source}",
                "!", "qtdemux",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=BGR",
                "!", "fdsink", "fd=1"
            ]
        else:
            return [
                GST_BIN, "-q",
                "rtspsrc", f"location={source}", "latency=0",
                "!", "rtph264depay",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=BGR",
                "!", "fdsink", "fd=1"
            ]
    
    total_frames = 0
    restart_count = 0
    
    while not stop_event.is_set():
        try:
            args = get_args()
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 2)
            frame_count = 0
            
            while not stop_event.is_set():
                data = proc.stdout.read(frame_size)
                if not data or len(data) != frame_size:
                    if is_file:
                        proc.terminate()
                        proc.wait(timeout=1)
                        restart_count += 1
                        print(f"[GST-{stream_id}] Loop {restart_count}...")
                        break
                    else:
                        break
                
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                frame_count += 1
                total_frames += 1
                
                try:
                    if queue.full():
                        queue.get_nowait()
                    queue.put_nowait((stream_id, frame))
                except:
                    pass
            
            if not is_file:
                break
                
        except Exception as e:
            print(f"[GST-{stream_id}] Error: {e}")
            if not stop_event.is_set():
                time.sleep(0.1)
    
    print(f"[GST-{stream_id}] Exited: {total_frames} frames, {restart_count} loops")


class MultiStreamCapture:
    def __init__(self, sources, width=640, height=640, queue_size=2):
        self.width = width
        self.height = height
        self.sources = sources
        self.streams = []
        self.queue = mp.Queue(maxsize=queue_size * len(sources))
        self.stop_event = mp.Event()
        self.latest_frames = {}  # Store latest frame per stream
        
        for i, source in enumerate(sources):
            process = mp.Process(
                target=gst_reader_process,
                args=(source, width, height, self.queue, self.stop_event, i),
                daemon=True
            )
            process.start()
            self.streams.append(process)
            self.latest_frames[i] = None
            print(f"[STREAM-{i}] {source} -> PID={process.pid}")
        
        # Thread untuk update latest_frames
        self.running = True
        self.thread = threading.Thread(target=self._update_latest, daemon=True)
        self.thread.start()
    
    def _update_latest(self):
        while self.running:
            try:
                ok, stream_id, frame = self.read(timeout=0.1)
                if ok and stream_id is not None:
                    self.latest_frames[stream_id] = frame
            except:
                time.sleep(0.001)
    
    def read(self, timeout=0.1):
        if not self.running or all(not p.is_alive() for p in self.streams):
            return False, None, None
        
        try:
            stream_id, frame = self.queue.get(timeout=timeout)
            return True, stream_id, frame
        except:
            return False, None, None
    
    def get_latest_frames(self):
        """Get dict of latest frames per stream."""
        return self.latest_frames.copy()
    
    def release(self):
        self.running = False
        self.stop_event.set()
        for p in self.streams:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        print(f"[MULTI-STREAM] Released {len(self.streams)} streams")


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
            fields = ["timestamp", "fps", "inference_ms", "decode_ms", "cpu_pct", "ram_mb",
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

    def log(self, fps, inf_ms, decode_ms, streams, m):
        if self.writer:
            self.writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": f"{fps:.1f}",
                "inference_ms": f"{inf_ms:.1f}",
                "decode_ms": f"{decode_ms:.1f}",
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


def create_grid_display(frames_dict, detections_dict, stats, fps):
    """Create 2x2 atau NxM grid dari frames."""
    num_streams = len(frames_dict)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_streams)))
    rows = int(np.ceil(num_streams / cols))
    
    # Get frame size
    sample_frame = next(iter(frames_dict.values()))
    if sample_frame is None:
        return None
    h, w = sample_frame.shape[:2]
    
    # Create grid canvas
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i in range(num_streams):
        frame = frames_dict.get(i)
        if frame is None:
            continue
        
        # Add detection boxes jika ada
        if i in detections_dict and detections_dict[i] is not None:
            boxes = detections_dict[i].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes.xyxy[0][:10]:  # Max 10 boxes
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate position
        row = i // cols
        col = i % cols
        
        # Draw frame
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame
        
        # Draw stream label
        label = f"Stream {i}"
        cv2.putText(grid, label, (col*w + 10, row*h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw global stats
    global_stats = f"FPS: {fps:.1f} | GPU: {stats['gpu_pct']}% | VRAM: {stats['gpu_mem_mb']:.0f}MB | T: {stats['gpu_temp_c']}C"
    cv2.putText(grid, global_stats, (10, grid.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return grid


def main():
    ap = argparse.ArgumentParser(description="TensorRT Multi-Stream Grid Display")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--sources", nargs="+", default=[r"C:\cctv-dev-temp\stairs_people.mp4"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log-csv", default="")
    args = ap.parse_args()

    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    num_streams = len(args.sources)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[BATCH] Streams: {num_streams}")

    if not Path(GST_BIN).exists():
        print(f"[ERROR] GStreamer tidak ditemukan: {GST_BIN}")
        return

    model = YOLO(str(engine_path))
    cap = MultiStreamCapture(args.sources, args.imgsz, args.imgsz, queue_size=2)
    
    time.sleep(0.5)

    mon = SystemMonitor(args.log_csv or None)
    inf_times = deque(maxlen=50)
    dec_times = deque(maxlen=50)
    frame_count = 0
    detections = {}
    start_time = time.time()

    print("Running... tekan 'q' untuk stop")
    try:
        while True:
            t_dec_start = time.time()
            ok, stream_id, frame = cap.read(timeout=0.5)
            dec_ms = (time.time() - t_dec_start) * 1000
            
            if not ok or frame is None:
                if all(not p.is_alive() for p in cap.streams):
                    break
                continue

            frame_count += 1
            dec_times.append(dec_ms)

            # Inference
            t_inf_start = time.time()
            results = model.predict(
                source=frame,
                verbose=False,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf
            )
            wall_inf = (time.time() - t_inf_start) * 1000

            r = results[0]
            inf_ms = float(r.speed.get("inference", 0))
            inf_times.append(inf_ms)
            
            # Store detection untuk display
            detections[stream_id] = r

            # Display grid setiap 3 frame
            if frame_count % 3 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                avg_inf = sum(inf_times) / len(inf_times)
                avg_dec = sum(dec_times) / len(dec_times)
                stats = mon.read()
                
                frames = cap.get_latest_frames()
                grid = create_grid_display(frames, detections, stats, fps)
                
                if grid is not None:
                    cv2.imshow("Multi-Stream Grid Display", grid)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Logging
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                stats = mon.read()
                print(f"[{frame_count:4d}] FPS ~{fps:5.1f} | "
                      f"dec {avg_dec:5.1f}ms | inf {avg_inf:5.1f}ms | "
                      f"GPU {stats['gpu_pct']:3.0f}% | VRAM {stats['gpu_mem_mb']:5.0f}MB")
                mon.log(fps, avg_inf, avg_dec, num_streams, stats)

            if args.max_frames and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        mon.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Total frames: {frame_count}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
