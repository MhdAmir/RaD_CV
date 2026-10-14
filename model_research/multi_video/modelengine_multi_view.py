r"""
Multi-Stream dengan Individual Windows (1 window per stream)

Setiap stream punya window sendiri dengan bounding boxes.

Usage:
  python modelengine_multi_view.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
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
    "people_counting": r"people_counting_yolo26m\weights\best.pt",
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


def gst_reader_process(source, width, height, queue, stop_event, stream_id):
    """GStreamer decode process dengan auto-restart."""
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
    
    print(f"[GST-{stream_id}] Exited: {total_frames} frames")


class MultiStreamCapture:
    def __init__(self, sources, width=640, height=640, queue_size=1):
        self.width = width
        self.height = height
        self.sources = sources
        self.streams = []
        self.queue = mp.Queue(maxsize=queue_size * len(sources))
        self.stop_event = mp.Event()
        self.latest_frames = {}
        self.latest_results = {}  # Store inference results
        
        for i, source in enumerate(sources):
            process = mp.Process(
                target=gst_reader_process,
                args=(source, width, height, self.queue, self.stop_event, i),
                daemon=True
            )
            process.start()
            self.streams.append(process)
            self.latest_frames[i] = None
            self.latest_results[i] = None
            print(f"[STREAM-{i}] {source}")
        
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
        if not self.running:
            return False, None, None
        try:
            stream_id, frame = self.queue.get(timeout=timeout)
            return True, stream_id, frame
        except:
            return False, None, None
    
    def get_latest_frames(self):
        return self.latest_frames.copy()
    
    def release(self):
        self.running = False
        self.stop_event.set()
        for p in self.streams:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()


def display_thread(cap, model, args, stop_event):
    """Dedicated thread untuk display (non-blocking inference)."""
    cv2.namedWindow("Stream 0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Stream 1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Stream 2", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Stream 3", cv2.WINDOW_NORMAL)
    
    last_display = {i: time.time() for i in range(len(args.sources))}
    target_fps = 30  # Target display FPS
    
    while not stop_event.is_set():
        frames = cap.get_latest_frames()
        
        for stream_id, frame in frames.items():
            if frame is None:
                continue
            
            # Throttle display to target FPS
            now = time.time()
            if (now - last_display[stream_id]) < (1.0 / target_fps):
                continue
            
            # Inference + draw (bisa di-skip untuk lebih smooth)
            try:
                results = model.predict(
                    source=frame,
                    verbose=False,
                    device=args.device,
                    imgsz=args.imgsz,
                    conf=args.conf
                )
                annotated = results[0].plot()
            except:
                annotated = frame
            
            cv2.imshow(f"Stream {stream_id}", annotated)
            last_display[stream_id] = now
        
        # Minimal waitKey untuk event processing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
    
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Multi-Stream dengan Windows")
    ap.add_argument("--model", default="people_counting")
    ap.add_argument("--sources", nargs="+", default=[r"C:\cctv-dev-temp\stairs_people.mp4"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--display-fps", type=int, default=24, help="Target display FPS (default 24)")
    args = ap.parse_args()

    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    num_streams = len(args.sources)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[VIEW] Streams: {num_streams}")
    print(f"[DISPLAY] Target FPS: {args.display_fps}")

    model = YOLO(str(engine_path))
    cap = MultiStreamCapture(args.sources, args.imgsz, args.imgsz, queue_size=1)
    
    time.sleep(0.5)

    stop_event = threading.Event()
    
    # Start display thread
    d_thread = threading.Thread(target=display_thread, args=(cap, model, args, stop_event), daemon=True)
    d_thread.start()

    frame_count = 0
    start_time = time.time()

    print("Running... press 'q' to stop")
    try:
        while not stop_event.is_set():
            ok, stream_id, frame = cap.read(timeout=0.1)
            
            if not ok:
                if all(not p.is_alive() for p in cap.streams):
                    break
                continue

            frame_count += 1

            # Stats
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[{frame_count:4d}] FPS ~{fps:5.1f}")

            if args.max_frames and frame_count >= args.max_frames:
                stop_event.set()
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        d_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nStopped. Total frames: {frame_count}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
