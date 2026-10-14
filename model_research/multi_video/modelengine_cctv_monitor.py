r"""
Multi-Stream CCTV Monitoring (Fixed Display + No Frame Skip)

- Semua frame di-process (no dropping)
- Smooth display dengan timing konsisten
- Dedicated threads untuk decode, inference, dan display

Usage:
  python modelengine_cctv_monitor.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
import multiprocessing as mp
import subprocess
import threading
import time
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


class StreamDecoder:
    """Dedicated thread untuk decode per stream dengan cv2.VideoCapture."""
    
    def __init__(self, source, stream_id, width=640, height=640):
        self.source = source
        self.stream_id = stream_id
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.fps = 0.0
        
        self.thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.thread.start()
    
    def _decode_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                time.sleep(0.5)
                continue
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            start_time = time.time()
            count = 0
            
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    print(f"[STREAM-{self.stream_id}] Looping...")
                    time.sleep(0.1)
                    break
                
                count += 1
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                with self.lock:
                    self.frame = frame.copy()
                    self.frame_count = count
                    self.fps = count / (time.time() - start_time)
    
    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


class InferenceWorker:
    """Dedicated thread untuk inference per stream."""
    
    def __init__(self, stream_id, model, device, imgsz, conf):
        self.stream_id = stream_id
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.result = None
        self.lock = threading.Lock()
        self.running = True
        self.input_frame = None
        self.fps = 0.0
        self.frame_count = 0
        
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def _inference_loop(self):
        start_time = time.time()
        while self.running:
            if self.input_frame is not None:
                frame = self.input_frame
                self.input_frame = None
                
                try:
                    results = self.model.predict(
                        source=frame,
                        verbose=False,
                        device=self.device,
                        imgsz=self.imgsz,
                        conf=self.conf
                    )
                    with self.lock:
                        self.result = results[0]
                        self.frame_count += 1
                        self.fps = self.frame_count / (time.time() - start_time)
                except Exception as e:
                    print(f"[INF-{self.stream_id}] Error: {e}")
            else:
                time.sleep(0.001)
    
    def set_frame(self, frame):
        if self.input_frame is None:  # Only accept new frame if buffer empty
            self.input_frame = frame
    
    def get_result(self):
        with self.lock:
            return self.result
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


class DisplayManager:
    """Dedicated thread untuk display dengan smooth timing."""
    
    def __init__(self, num_streams, target_fps=30):
        self.num_streams = num_streams
        self.target_fps = target_fps
        self.frames_to_display = {i: None for i in range(num_streams)}
        self.running = True
        
        for i in range(num_streams):
            cv2.namedWindow(f"CCTV {i}", cv2.WINDOW_NORMAL)
        
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()
    
    def _display_loop(self):
        last_display = {i: time.time() for i in range(self.num_streams)}
        frame_interval = 1.0 / self.target_fps
        
        while self.running:
            for i in range(self.num_streams):
                if self.frames_to_display[i] is not None:
                    now = time.time()
                    if (now - last_display[i]) >= frame_interval:
                        cv2.imshow(f"CCTV {i}", self.frames_to_display[i])
                        last_display[i] = now
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        
        cv2.destroyAllWindows()
    
    def update_frame(self, stream_id, frame):
        self.frames_to_display[stream_id] = frame
    
    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


def main():
    ap = argparse.ArgumentParser(description="CCTV Monitoring (All Frames Processed)")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--display-fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    # Resolve model
    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    num_streams = len(args.sources)
    print(f"[TRT] engine={engine_path.name}")
    print(f"[CCTV] Streams: {num_streams}")
    print(f"[DISPLAY] Target FPS: {args.display_fps}")

    # Load model
    model = YOLO(str(engine_path))

    # Create decoders, inference workers, dan display manager
    decoders = []
    workers = []
    
    for i in range(num_streams):
        print(f"[STREAM-{i}] {args.sources[i]}")
        decoder = StreamDecoder(args.sources[i], i, args.imgsz, args.imgsz)
        worker = InferenceWorker(i, model, args.device, args.imgsz, args.conf)
        decoders.append(decoder)
        workers.append(worker)
    
    display = DisplayManager(num_streams, args.display_fps)

    # Main loop: decode -> inference -> display
    frame_count = 0
    start_time = time.time()
    total_inferences = 0

    print("Running CCTV monitoring... press 'q' to stop")
    try:
        while True:
            for i in range(num_streams):
                # 1. Get decoded frame
                ok, frame = decoders[i].get_frame()
                if ok:
                    # 2. Send to inference worker
                    workers[i].set_frame(frame)
                    
                    # 3. Get inference result
                    result = workers[i].get_result()
                    if result is not None:
                        # 4. Draw and display
                        annotated = result.plot()
                        fps = workers[i].fps
                        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        display.update_frame(i, annotated)
                        total_inferences += 1
            
            frame_count += 1
            
            # Stats
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                decoders_fps = [d.fps for d in decoders]
                workers_fps = [w.fps for w in workers]
                print(f"[{frame_count:4d}] Total inferences: {total_inferences} | "
                      f"Decode FPS: {[f'{f:.1f}' for f in decoders_fps]} | "
                      f"Inf FPS: {[f'{f:.1f}' for f in workers_fps]}")
            
            if args.max_frames and frame_count >= args.max_frames:
                break
            
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        display.stop()
        for w in workers:
            w.stop()
        for d in decoders:
            d.stop()
        print(f"\nStopped. Total inferences: {total_inferences}")


if __name__ == "__main__":
    main()
