r"""
CCTV Real-time Monitoring - Single Window Grid View

- SEMUA streams dalam 1 window (grid layout)
- SETIAP frame di-process (no skip)
- Auto-adjust grid berdasarkan jumlah streams
- Dedicated thread untuk display

Usage:
  python modelengine_cctv_single_window.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
"""

import argparse
import multiprocessing as mp
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue

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


def calc_grid(n):
    """Calculate grid layout (rows, cols) for n streams."""
    if n <= 1:
        return 1, 1
    elif n <= 4:
        return 2, 2
    elif n <= 6:
        return 2, 3
    elif n <= 9:
        return 3, 3
    elif n <= 12:
        return 3, 4
    else:
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        return rows, cols


def stream_process_loop(source, stream_id, width, height, engine_path, device, imgsz, conf, stop_event, output_queue):
    """
    Dedicated process per stream: decode → inference → put result to queue.
    """
    from ultralytics import YOLO
    model = YOLO(str(engine_path))
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[STREAM-{stream_id}] FAILED to open {source}")
        return
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    infer_count = 0
    start_time = time.time()
    
    print(f"[STREAM-{stream_id}] STARTED")
    
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            cap.release()
            time.sleep(0.1)
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        
        frame_count += 1
        
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        try:
            results = model.predict(
                source=frame,
                verbose=False,
                device=device,
                imgsz=imgsz,
                conf=conf
            )
            result = results[0]
            annotated = result.plot()
            infer_count += 1
        except Exception as e:
            annotated = frame
        
        elapsed = time.time() - start_time
        fps_decode = frame_count / elapsed if elapsed > 0 else 0
        fps_infer = infer_count / elapsed if elapsed > 0 else 0
        
        # Draw stats on frame
        cv2.putText(annotated, f"Cam {stream_id} | D:{fps_decode:.0f} I:{fps_infer:.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Put to queue for display thread
        if not output_queue.full():
            output_queue.put((stream_id, annotated))
    
    cap.release()
    print(f"[STREAM-{stream_id}] STOPPED - Frames: {frame_count}, Inferences: {infer_count}")


def display_loop(num_streams, stop_event, frame_queues, cell_size=(640, 640)):
    """
    Dedicated thread untuk display: collect all frames → stitch to grid → show.
    """
    rows, cols = calc_grid(num_streams)
    grid_height = rows * cell_size[1]
    grid_width = cols * cell_size[0]
    
    # Initialize grid with black
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Create window
    cv2.namedWindow("CCTV Monitoring", cv2.WINDOW_NORMAL)
    cv2.imshow("CCTV Monitoring", grid)
    
    # Store latest frames
    latest_frames = {i: None for i in range(num_streams)}
    
    last_display_time = time.time()
    display_fps_target = 30
    display_interval = 1.0 / display_fps_target
    
    print(f"[DISPLAY] Grid: {rows}x{cols} ({grid_width}x{grid_height})")
    
    while not stop_event.is_set():
        # Collect frames from all queues (non-blocking)
        updated = False
        for i in range(num_streams):
            try:
                while not frame_queues[i].empty():
                    stream_id, frame = frame_queues[i].get_nowait()
                    latest_frames[stream_id] = frame
                    updated = True
            except:
                pass
        
        # Update grid if any frame changed
        if updated:
            for i in range(num_streams):
                if latest_frames[i] is not None:
                    row = i // cols
                    col = i % cols
                    y1, y2 = row * cell_size[1], (row + 1) * cell_size[1]
                    x1, x2 = col * cell_size[0], (col + 1) * cell_size[0]
                    
                    frame = latest_frames[i]
                    if frame.shape[0] != cell_size[1] or frame.shape[1] != cell_size[0]:
                        frame = cv2.resize(frame, (cell_size[0], cell_size[1]))
                    
                    grid[y1:y2, x1:x2] = frame
            
            # Display throttled to target FPS
            now = time.time()
            if (now - last_display_time) >= display_interval:
                cv2.imshow("CCTV Monitoring", grid)
                last_display_time = now
        
        # Check quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
        
        time.sleep(0.001)
    
    cv2.destroyWindow("CCTV Monitoring")
    print("[DISPLAY] Stopped")


def main():
    ap = argparse.ArgumentParser(description="CCTV Single Window Grid")
    ap.add_argument("--model", default="ppe_compliance")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--quantize", type=int, default=0, choices=[0, 16])
    ap.add_argument("--cell-size", type=int, default=640, help="Size of each grid cell")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if args.model.lower().endswith(".engine") and Path(args.model).is_file():
        engine_path = Path(args.model)
    else:
        pt_path = resolve_pt(args.model)
        engine_path = get_engine(pt_path, args.imgsz, args.quantize)

    num_streams = len(args.sources)
    rows, cols = calc_grid(num_streams)
    
    print(f"[TRT] engine={engine_path.name}")
    print(f"[CCTV] Streams: {num_streams} ({rows}x{cols} grid)")
    print(f"[MODE] NO FRAME SKIP - Every frame processed")
    print(f"[DISPLAY] Single window grid")

    # Create queues (one per stream)
    frame_queues = [mp.Queue(maxsize=5) for _ in range(num_streams)]
    
    # Create stop event
    stop_event = mp.Event()

    # Start display thread (in main process - OpenCV needs this)
    display_thread = threading.Thread(
        target=display_loop,
        args=(num_streams, stop_event, frame_queues, (args.cell_size, args.cell_size)),
        daemon=True
    )
    display_thread.start()
    
    # Give display thread time to create window
    time.sleep(0.5)

    # Start stream processes
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
                stop_event,
                frame_queues[i]
            ),
            daemon=True
        )
        p.start()
        processes.append(p)
        print(f"[STREAM-{i}] {args.sources[i]}")
    
    print(f"[CCTV] All {num_streams} streams started. Press 'q' to stop.")

    # Wait for processes
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    
    # Cleanup
    for p in processes:
        p.join(timeout=2)
    
    display_thread.join(timeout=2)
    cv2.destroyAllWindows()
    print("\n[CCTV] Stopped.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
