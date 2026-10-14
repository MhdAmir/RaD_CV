r"""
CCTV Real-time Monitoring (NO FRAME SKIP - Every Frame Processed)

- SETIAP frame di-process (decode → inference → display)
- TIDAK ADA frame dropping/skipping
- Display langsung muncul dengan frame pertama
- Architecture: 1 thread per stream (decode+inference+display dalam 1 loop)

Usage:
  python modelengine_cctv_realtime.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
import multiprocessing as mp
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pynvml
from ultralytics import YOLO

MODELS = {
    "ppe_compliance": r"ppe_compliance_yolo26m\weights\best.pt",
    "security_officer_v2": r"security_officer_v2_yolo26m\weights\best.pt",
    "helmet_and_vest": r"helmet_and_vest_yolo26m\weights\best.pt",
    "face_mask": r"face_mask_yolo26m\weights\best.pt",
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


def stream_process_loop(source, stream_id, width, height, engine_path, device, imgsz, conf, stop_event):
    """
    Dedicated process per stream: decode → inference → display (NO SKIP).
    Setiap frame LANGSUNG di-process dan ditampilkan.
    """
    # Load model INSIDE child process (cannot pickle YOLO model)
    from ultralytics import YOLO
    model = YOLO(str(engine_path))
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[STREAM-{stream_id}] FAILED to open {source}")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Always get latest frame
    
    # Create window IMMEDIATELY
    window_name = f"CCTV {stream_id}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, np.zeros((height, width, 3), dtype=np.uint8))
    
    frame_count = 0
    infer_count = 0
    start_time = time.time()
    
    print(f"[STREAM-{stream_id}] STARTED")
    
    while not stop_event.is_set():
        # 1. Decode frame
        ok, frame = cap.read()
        if not ok:
            # EOF - restart (loop video)
            cap.release()
            time.sleep(0.1)
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_count += 1
        
        # 2. Resize if needed
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        # 3. Inference (EVERY frame - NO SKIP)
        try:
            results = model.predict(
                source=frame,
                verbose=False,
                device=device,
                imgsz=imgsz,
                conf=conf
            )
            result = results[0]
            
            # 4. Draw boxes
            annotated = result.plot()
            infer_count += 1
        except Exception as e:
            annotated = frame
        
        # 5. Draw stats
        elapsed = time.time() - start_time
        fps_decode = frame_count / elapsed if elapsed > 0 else 0
        fps_infer = infer_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(annotated, f"D:{fps_decode:.0f} I:{fps_infer:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 6. Display IMMEDIATELY (NO THROTTLE)
        cv2.imshow(window_name, annotated)
        
        # 7. Check quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    # Cleanup
    cap.release()
    cv2.destroyWindow(window_name)
    print(f"[STREAM-{stream_id}] STOPPED - Frames: {frame_count}, Inferences: {infer_count}")


def main():
    ap = argparse.ArgumentParser(description="CCTV Real-time (NO FRAME SKIP)")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
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
    print(f"[MODE] NO FRAME SKIP - Every frame processed")
    print(f"[MODEL] Loading {engine_path.name}...")

    # Create stop event
    stop_event = mp.Event()

    # Start ONE PROCESS PER STREAM (decode → inference → display in same process)
    # Model loaded INSIDE each process (cannot pickle YOLO)
    processes = []
    for i in range(num_streams):
        p = mp.Process(
            target=stream_process_loop,
            args=(
                args.sources[i],
                i,
                args.imgsz,
                args.imgsz,
                engine_path,
                args.device,
                args.imgsz,
                args.conf,
                stop_event
            ),
            daemon=True
        )
        p.start()
        processes.append(p)
    
    print(f"[CCTV] All {num_streams} streams started. Press 'q' in any window to stop.")
    
    # Wait for all processes
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
            
            # Check if user pressed q in main thread
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    except KeyboardInterrupt:
        stop_event.set()
    
    # Wait for processes to finish
    for p in processes:
        p.join(timeout=2)
    
    cv2.destroyAllWindows()
    print("\n[CCTV] Stopped.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
