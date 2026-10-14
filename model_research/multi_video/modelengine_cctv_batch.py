r"""
CCTV Batch Inference - Optimized Multi-Stream Monitoring (FIXED)

- TRUE BATCH inference (4 frames → 1 predict call)
- GStreamer NVDEC GPU decode
- Single window grid display
- All threads in main process (no multiprocessing issues)

Usage:
  python modelengine_cctv_batch.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16 --batch-size 4
"""

import argparse
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

MODELS = {
    "ppe_compliance": r"ppe_compliance_yolo26m\weights\best.pt",
    "security_officer_v2": r"security_officer_v2_yolo26m\weights\best.pt",
    "helmet_and_vest": r"helmet_and_vest_yolo26m\weights\best.pt",
    "face_mask": r"face_mask_yolo26m\weights\best.pt",
    "people_counting": r"people_counting_yolo26m\weights\best.pt",
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


def get_engine(pt_path: Path, imgsz: int, quantize: int, batch_size: int = 1, force: bool = False) -> Path:
    suffix = "_fp16" if quantize == 16 else "_fp32"
    engine_path = pt_path.with_name(f"{pt_path.stem}{suffix}_bs{batch_size}.engine")
    bs1_engine = pt_path.with_name(f"{pt_path.stem}{suffix}.engine")
    
    if batch_size > 1:
        if engine_path.exists() and not force:
            print(f"[TRT] Engine found: {engine_path.name}")
            return engine_path
        print(f"[TRT] Engine not found, using batch=1: {bs1_engine.name}")
        return bs1_engine
    
    if engine_path.exists():
        return engine_path
    if bs1_engine.exists():
        print(f"[TRT] Engine found: {bs1_engine.name}")
        return bs1_engine
    
    print(f"[TRT] Engine not found: {bs1_engine.name}")
    return bs1_engine


def build_gst_pipeline(source):
    """Build GStreamer pipeline for NVDEC GPU decode."""
    ext = Path(source).suffix.lower()
    
    if ext in ['.mp4', '.mkv', '.avi', '.mov']:
        pipeline = (
            f'filesrc location="{source}" '
            f'! qtdemux ! h264parse ! nvv4l2decoder enable-max-performance=1 '
            f'! nvvidconv ! video/x-raw,format=NV12,width=640,height=640 '
            f'! appsink name=sink emit-signals=true sync=false drop=true max-buffers=1'
        )
    elif source.startswith('rtsp://'):
        pipeline = (
            f'rtspsrc location={source} latency=0 '
            f'! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 '
            f'! nvvidconv ! video/x-raw,format=NV12,width=640,height=640 '
            f'! appsink name=sink emit-signals=true sync=false drop=true max-buffers=1'
        )
    else:
        pipeline = (
            f'uridecodebin uri=file://{source} '
            f'! nvv4l2decoder enable-max-performance=1 '
            f'! nvvidconv ! video/x-raw,format=NV12,width=640,height=640 '
            f'! appsink name=sink emit-signals=true sync=false drop=true max-buffers=1'
        )
    
    return pipeline


class StreamDecoder(threading.Thread):
    """Dedicated thread untuk decode per stream dengan GStreamer NVDEC."""
    
    def __init__(self, source, stream_id, width, height, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.source = source
        self.stream_id = stream_id
        self.width = width
        self.height = height
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        
        self.frame_count = 0
        self.fps = 0.0
        self.start_time = time.time()
    
    def run(self):
        # Try GStreamer first
        gst_pipeline = build_gst_pipeline(self.source)
        print(f"[DEC-{self.stream_id}] GST pipeline created")
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        use_gst = cap.isOpened()
        
        if not use_gst:
            print(f"[DEC-{self.stream_id}] GStreamer failed, fallback to cv2")
            cap = cv2.VideoCapture(self.source)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[DEC-{self.stream_id}] STARTED ({'GStreamer' if use_gst else 'cv2'})")
        
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.release()
                time.sleep(0.1)
                if use_gst:
                    cap = cv2.VideoCapture(build_gst_pipeline(self.source), cv2.CAP_GSTREAMER)
                else:
                    cap = cv2.VideoCapture(self.source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            self.frame_count += 1
            
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Put frame (non-blocking)
            try:
                self.frame_queue.put((self.stream_id, frame.copy()), block=False)
            except:
                pass  # Drop if full
            
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
        
        cap.release()
        print(f"[DEC-{self.stream_id}] STOPPED - {self.frame_count} frames @ {self.fps:.1f} FPS")
    
    def get_stats(self):
        return self.frame_count, self.fps


class BatchInference(threading.Thread):
    """Dedicated thread untuk batch inference."""
    
    def __init__(self, num_streams, model, device, imgsz, conf, input_queue, output_queue, stop_event):
        super().__init__(daemon=True)
        self.num_streams = num_streams
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        
        self.batch_count = 0
        self.infer_count = 0
        self.fps = 0.0
        self.start_time = time.time()
    
    def run(self):
        print(f"[BATCH-INF] STARTED (batch_size={self.num_streams})")
        
        frame_buffer = {}
        last_infer_time = time.time()
        infer_interval = 1.0 / 30
        
        while not self.stop_event.is_set():
            # Collect frames (non-blocking)
            try:
                stream_id, frame = self.input_queue.get_nowait()
                frame_buffer[stream_id] = frame
            except Empty:
                pass
            
            # Check if we have all streams
            if len(frame_buffer) >= self.num_streams:
                now = time.time()
                if (now - last_infer_time) >= infer_interval:
                    batch_streams = sorted(frame_buffer.keys())
                    batch_frames = [frame_buffer[sid] for sid in batch_streams]
                    frame_buffer.clear()
                    
                    try:
                        # BATCH INFERENCE
                        results = self.model.predict(
                            source=batch_frames,
                            verbose=False,
                            device=self.device,
                            imgsz=self.imgsz,
                            conf=self.conf,
                            batch=len(batch_frames)
                        )
                        
                        for stream_id, result in zip(batch_streams, results):
                            annotated = result.plot()
                            cv2.putText(annotated, f"Cam {stream_id}",
                                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            try:
                                self.output_queue.put((stream_id, annotated), block=False)
                            except:
                                pass
                        
                        self.batch_count += 1
                        self.infer_count += len(batch_streams)
                        last_infer_time = now
                        
                        elapsed = now - self.start_time
                        if elapsed > 0:
                            self.fps = self.infer_count / elapsed
                    
                    except Exception as e:
                        print(f"[BATCH-INF] Error: {e}")
                        frame_buffer.clear()
            
            time.sleep(0.001)
        
        print(f"[BATCH-INF] STOPPED - {self.batch_count} batches, {self.infer_count} inferences @ {self.fps:.1f} FPS")
    
    def get_stats(self):
        return self.batch_count, self.infer_count, self.fps


class DisplayWorker(threading.Thread):
    """Dedicated thread untuk display grid."""
    
    def __init__(self, num_streams, cell_size, output_queue, stop_event):
        super().__init__(daemon=True)
        self.num_streams = num_streams
        self.cell_size = cell_size
        self.output_queue = output_queue
        self.stop_event = stop_event
        
        self.rows = 2 if num_streams <= 4 else 3
        self.cols = (num_streams + 1) // 2
        self.grid_height = self.rows * cell_size[1]
        self.grid_width = self.cols * cell_size[0]
        
        self.latest_frames = {i: None for i in range(num_streams)}
    
    def run(self):
        grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        
        cv2.namedWindow("CCTV Batch", cv2.WINDOW_NORMAL)
        cv2.imshow("CCTV Batch", grid)
        
        print(f"[DISPLAY] Grid: {self.rows}x{self.cols} ({self.grid_width}x{self.grid_height})")
        
        last_display = time.time()
        display_interval = 1.0 / 30
        
        while not self.stop_event.is_set():
            updated = False
            try:
                while True:
                    stream_id, frame = self.output_queue.get_nowait()
                    self.latest_frames[stream_id] = frame
                    updated = True
            except:
                pass
            
            if updated:
                for i in range(self.num_streams):
                    if self.latest_frames[i] is not None:
                        row = i // self.cols
                        col = i % self.cols
                        y1, y2 = row * self.cell_size[1], (row + 1) * self.cell_size[1]
                        x1, x2 = col * self.cell_size[0], (col + 1) * self.cell_size[0]
                        
                        frame = cv2.resize(self.latest_frames[i], (self.cell_size[0], self.cell_size[1]))
                        grid[y1:y2, x1:x2] = frame
                
                now = time.time()
                if (now - last_display) >= display_interval:
                    cv2.imshow("CCTV Batch", grid)
                    last_display = now
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break
            
            time.sleep(0.001)
        
        cv2.destroyWindow("CCTV Batch")
        print("[DISPLAY] STOPPED")


def main():
    ap = argparse.ArgumentParser(description="CCTV Batch Inference (FIXED)")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--cell-size", type=int, default=480)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        batch_size = args.batch_size if args.batch_size > 0 else len(args.sources)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize, batch_size=batch_size)

    num_streams = len(args.sources)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[CCTV] Streams: {num_streams}")
    print(f"[BATCH] Size: {num_streams}")

    # Load model
    model = YOLO(str(engine_path))

    # Create queues
    frame_queue = Queue(maxsize=num_streams * 2)
    result_queue = Queue(maxsize=num_streams * 2)
    
    stop_event = threading.Event()

    # Start decoders
    decoders = []
    for i in range(num_streams):
        dec = StreamDecoder(args.sources[i], i, args.imgsz, args.imgsz, frame_queue, stop_event)
        decoders.append(dec)
        dec.start()
        print(f"[DEC-{i}] {args.sources[i]}")
    
    time.sleep(0.5)

    # Start batch inference
    batch_inf = BatchInference(num_streams, model, args.device, args.imgsz, args.conf,
                                frame_queue, result_queue, stop_event)
    batch_inf.start()
    
    time.sleep(0.5)

    # Start display
    display = DisplayWorker(num_streams, (args.cell_size, args.cell_size), result_queue, stop_event)
    display.start()

    print(f"[CCTV] All {num_streams} streams started. Press 'q' to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            
            dec_stats = [d.get_stats() for d in decoders]
            batch_stats = batch_inf.get_stats()
            
            dec_fps = [f"{s[1]:.1f}" for s in dec_stats]
            total_frames = sum(s[0] for s in dec_stats)
            
            print(f"[STATS] Decode FPS: {dec_fps} | Batch: {batch_stats[0]} | "
                  f"Inferences: {batch_stats[1]} | Inf FPS: {batch_stats[2]:.1f}")
            
            if args.max_frames and total_frames >= args.max_frames:
                break
    
    except KeyboardInterrupt:
        stop_event.set()
    
    print("\n[CLEANUP] Stopping...")
    
    for dec in decoders:
        dec.stop_event.set()
    
    batch_inf.stop_event.set()
    display.stop_event.set()
    
    batch_inf.join(timeout=2)
    display.join(timeout=2)
    
    cv2.destroyAllWindows()
    print("\n[CCTV] Stopped.")


if __name__ == "__main__":
    main()
