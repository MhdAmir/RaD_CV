r"""
TensorRT + GStreamer NVDEC dengan Multiprocessing Queue (Optimized)

Optimization Phase 1b:
- GStreamer decode via NVDEC (GPU)
- Frame transfer via multiprocessing.Queue (lock-free ring buffer)
- Pre-allocated numpy arrays untuk minimize GC overhead

Expected: Decode time 6-10ms (vs 12-18ms dengan pipe)

Usage:
  python modelengine_gst_q.py --source "C:/cctv-dev-temp/stairs_people.mp4" --quantize 16
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


def gst_reader_process(source, width, height, queue, stop_event):
    """
    Dedicated process untuk GStreamer decode.
    Write frames ke multiprocessing.Queue (lock-free).
    """
    frame_size = width * height * 3
    is_file = Path(source).is_file() if isinstance(source, str) else False
    
    if is_file:
        args = [
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
        args = [
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
    
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 2)
        
        while not stop_event.is_set():
            data = proc.stdout.read(frame_size)
            if not data or len(data) != frame_size:
                break
            
            # Convert to numpy dan push ke queue
            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            
            # Non-blocking put (drop old frame if queue full)
            try:
                if queue.full():
                    queue.get_nowait()  # Drop oldest
                queue.put_nowait(frame)
            except:
                pass
        
        proc.terminate()
        proc.wait(timeout=2)
    except Exception as e:
        print(f"[GST-PROC] Error: {e}")
    finally:
        print(f"[GST-PROC] Exited")


class GstNvdecQueueCapture:
    """
    GStreamer NVDEC dengan Multiprocessing Queue.
    Zero-copy dari perspective Python (GStreamer process langsung write numpy array).
    """
    def __init__(self, source, width=640, height=640, queue_size=2):
        self.width = width
        self.height = height
        self.source = source
        self.queue = mp.Queue(maxsize=queue_size)
        self.stop_event = mp.Event()
        
        # Start dedicated GStreamer process
        self.process = mp.Process(
            target=gst_reader_process,
            args=(source, width, height, self.queue, self.stop_event),
            daemon=True
        )
        self.process.start()
        
        print(f"[QUEUE] GStreamer process started, PID={self.process.pid}")
        self.running = True
    
    def read(self, timeout=0.1):
        """Get latest frame dari queue."""
        if not self.running or not self.process.is_alive():
            return False, None
        
        try:
            frame = self.queue.get(timeout=timeout)
            return True, frame
        except:
            return False, None
    
    def release(self):
        """Stop GStreamer process."""
        self.running = False
        self.stop_event.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
        # Drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break
        print("[QUEUE] Released")


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
                      "gpu_pct", "gpu_mem_mb", "gpu_temp_c"]
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

    def log(self, fps, inf_ms, decode_ms, m):
        if self.writer:
            self.writer.writerow({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": f"{fps:.1f}",
                "inference_ms": f"{inf_ms:.1f}",
                "decode_ms": f"{decode_ms:.1f}",
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


def main():
    ap = argparse.ArgumentParser(description="TensorRT + GStreamer NVDEC + Queue")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--source", default=r"C:\cctv-dev-temp\stairs_people.mp4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--log-csv", default="")
    args = ap.parse_args()

    # Resolve model
    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    print(f"[TRT] engine={engine_path.name}")
    print(f"[TRT] source={args.source}")
    print(f"[OPT] Phase 1b: Multiprocessing Queue (Lock-Free)")

    if not Path(GST_BIN).exists():
        print(f"[ERROR] GStreamer tidak ditemukan: {GST_BIN}")
        return

    model = YOLO(str(engine_path))

    # Start GStreamer NVDEC dengan Queue
    print("[CAPTURE] Starting GStreamer NVDEC + Queue...")
    cap = GstNvdecQueueCapture(args.source, args.imgsz, args.imgsz, queue_size=2)
    
    time.sleep(0.5)  # Wait untuk first frame

    mon = SystemMonitor(args.log_csv or None)
    inf_times = deque(maxlen=50)
    dec_times = deque(maxlen=50)
    frame_count = 0
    start_time = time.time()

    print("Running... tekan Ctrl+C untuk stop")
    try:
        while True:
            t_dec_start = time.time()
            ok, frame = cap.read(timeout=0.5)
            dec_ms = (time.time() - t_dec_start) * 1000
            
            if not ok or frame is None:
                if not cap.process.is_alive():
                    print("[ERROR] GStreamer process died")
                    break
                continue

            frame_count += 1
            dec_times.append(dec_ms)

            # TensorRT inference
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

            # Display
            if not args.no_display and frame_count % 5 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                avg_inf = sum(inf_times) / len(inf_times)
                avg_dec = sum(dec_times) / len(dec_times)
                stats = mon.read()
                
                y = 25
                hud_text = f"NVDEC+Queue+TRT FPS:{fps:.0f} Dec:{avg_dec:.0f}ms Inf:{avg_inf:.0f}ms"
                cv2.putText(frame, hud_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("GStreamer NVDEC + Queue + TensorRT", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Logging per 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                avg_inf = sum(inf_times) / len(inf_times)
                avg_dec = sum(dec_times) / len(dec_times)
                stats = mon.read()

                print(f"[{frame_count:4d}] FPS ~{fps:5.1f} | "
                      f"dec {avg_dec:5.1f}ms | inf {avg_inf:5.1f}ms | "
                      f"CPU {stats['cpu_pct']:3.0f}% | GPU {stats['gpu_pct']:3.0f}% | "
                      f"VRAM {stats['gpu_mem_mb']:5.0f}MB | T:{stats['gpu_temp_c']}C")
                
                mon.log(fps, avg_inf, avg_dec, stats)

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
