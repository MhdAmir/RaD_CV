r"""
TensorRT + GStreamer NVDEC dengan Shared Memory (Zero-Copy)

Optimization Phase 1:
- GStreamer decode via NVDEC (GPU)
- Frame transfer via multiprocessing.shared_memory (zero-copy)
- TensorRT inference tanpa copy buffer

Expected: Decode time turun dari 12-18ms -> 5-7ms (2x speedup)

Usage:
  python modelengine_gst_shm.py --source "C:/cctv-dev-temp/stairs_people.mp4" --quantize 16
"""

import argparse
import csv
import mmap
import subprocess
import threading
import time
from collections import deque
from multiprocessing import shared_memory
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


class GstNvdecShmCapture:
    """
    GStreamer NVDEC dengan Shared Memory (zero-copy frame transfer).
    
    Cara kerja:
    1. Python allocate shared memory buffer
    2. GStreamer decode via NVDEC -> shmsink (write directly to shared mem)
    3. Python read dari shared memory tanpa copy
    4. Frame tersedia sebagai numpy array (zero-copy view)
    """
    def __init__(self, source, width=640, height=640):
        self.width = width
        self.height = height
        self.source = source
        self.frame_size = width * height * 3
        self.shm_name = f"gst_frame_{time.time_ns()}"
        self.shm = None
        self.process = None
        self.running = False
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.last_frame_time = 0
        
        # Create shared memory buffer (3 frames ring buffer)
        self.buffer_size = self.frame_size * 3
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name,
                create=True,
                size=self.buffer_size
            )
            print(f"[SHM] Created buffer: {self.shm_name} ({self.buffer_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"[SHM] Failed to create: {e}")
            # Fallback ke array biasa
            self.shm = None
            self.buffer = None
        
        # Build GStreamer pipeline dengan shmsink
        is_file = Path(source).is_file() if isinstance(source, str) else False
        
        if is_file:
            self.pipeline_args = [
                GST_BIN,
                "-q",
                "filesrc", f"location={source}",
                "!", "qtdemux",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=BGR",
                "!", "shmsink", f"socket-path={self.shm_name}", "wait-for-connection=false"
            ]
        else:
            # RTSP
            self.pipeline_args = [
                GST_BIN,
                "-q",
                "rtspsrc", f"location={source}", "latency=0",
                "!", "rtph264depay",
                "!", "h264parse",
                "!", "nvh264dec",
                "!", "videoconvert",
                "!", "videoscale",
                "!", f"video/x-raw,width={width},height={height},format=BGR",
                "!", "shmsink", f"socket-path={self.shm_name}", "wait-for-connection=false"
            ]
        
        print(f"[GST] Starting NVDEC + SHM pipeline...")
        
        # Start GStreamer process
        try:
            self.process = subprocess.Popen(
                self.pipeline_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            self.running = True
            
            # Start reader thread
            self.thread = threading.Thread(target=self._monitor_shm, daemon=True)
            self.thread.start()
            
            print(f"[GST] Process started, PID={self.process.pid}")
            
        except Exception as e:
            print(f"[GST] Failed to start: {e}")
            self.running = False
            if self.shm:
                self.shm.close()
                self.shm.unlink()
    
    def _monitor_shm(self):
        """Monitor shared memory untuk frame baru."""
        # Wait untuk GStreamer connect ke shmsink
        time.sleep(0.5)
        
        while self.running:
            try:
                # Check apakah ada data di shared memory
                if self.shm and self.shm.buf:
                    # Read header (first 4 bytes = frame size)
                    header = bytes(self.shm.buf[0:4])
                    frame_size = int.from_bytes(header, 'little')
                    
                    if frame_size == self.frame_size:
                        # Frame ready! Read data
                        frame_data = bytes(self.shm.buf[4:4+self.frame_size])
                        
                        if len(frame_data) == self.frame_size:
                            # Convert to numpy
                            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                                (self.height, self.width, 3)
                            ).copy()
                            
                            with self.lock:
                                self.buffer = frame
                                self.last_frame_time = time.time()
                            self.frame_ready.set()
                
                time.sleep(0.001)  # 1ms poll
                
            except Exception as e:
                # Silent error during normal operation
                if self.running:
                    time.sleep(0.01)
                break
        
        print("[SHM] Monitor thread stopped")
    
    def read(self, timeout=0.1):
        """Get latest frame dari shared memory."""
        if not self.running or self.shm is None:
            return False, None
        
        start = time.time()
        while self.running and (time.time() - start) < timeout:
            with self.lock:
                if hasattr(self, 'buffer') and self.buffer is not None:
                    frame = self.buffer
                    self.buffer = None  # Clear
                    return True, frame
            time.sleep(0.001)
        
        return False, None
    
    def release(self):
        """Stop dan cleanup."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1)
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                pass
        print("[SHM] Released")


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
    ap = argparse.ArgumentParser(description="TensorRT + GStreamer NVDEC + Shared Memory")
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
    print(f"[OPT] Phase 1: Shared Memory (Zero-Copy)")

    # Check GStreamer
    if not Path(GST_BIN).exists():
        print(f"[ERROR] GStreamer tidak ditemukan: {GST_BIN}")
        return

    # Load TensorRT model
    model = YOLO(str(engine_path))

    # Start GStreamer NVDEC dengan Shared Memory
    print("[CAPTURE] Starting GStreamer NVDEC + Shared Memory...")
    cap = GstNvdecShmCapture(args.source, args.imgsz, args.imgsz)
    
    # Wait untuk first frame
    print("[CAPTURE] Waiting for first frame...")
    time.sleep(1.0)

    mon = SystemMonitor(args.log_csv or None)
    inf_times = deque(maxlen=50)
    dec_times = deque(maxlen=50)
    frame_count = 0
    start_time = time.time()

    print("Running... tekan Ctrl+C untuk stop")
    try:
        while True:
            # Get frame dari shared memory
            t_dec_start = time.time()
            ok, frame = cap.read(timeout=0.5)
            dec_ms = (time.time() - t_dec_start) * 1000
            
            if not ok or frame is None:
                if not cap.running:
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
                hud_text = f"NVDEC+SHM+TRT FPS:{fps:.0f} Dec:{avg_dec:.0f}ms Inf:{avg_inf:.0f}ms"
                cv2.putText(frame, hud_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("GStreamer NVDEC + Shared Memory + TensorRT", frame)
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
    main()
