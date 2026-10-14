"""
RTMPose untuk video CCTV (via rtmlib, tanpa mmcv/mmpose/pytorch).

Instalasi:
    pip install rtmlib
    # rtmlib ikut memasang onnxruntime versi CPU -> ganti dengan versi GPU:
    pip uninstall -y onnxruntime onnxruntime-gpu
    pip install --force-reinstall --no-deps onnxruntime-gpu
    # Cek: pip list | findstr -i onnx   -> hanya onnxruntime-gpu
    # Untuk --trt: pip install tensorrt-cu12  (atau TensorRT dari NVIDIA, folder lib di PATH)

Contoh pemakaian:
    # File video, GPU + TensorRT FP16
    python rtmpose_cctv.py --source rekaman.mp4 --preset akurat --device cuda --trt --out hasil.mp4

    # RTSP dari CCTV/NVR (pakai main stream, bukan sub stream)
    python rtmpose_cctv.py --source "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101" --show

    # Webcam
    python rtmpose_cctv.py --source 0 --show

    # RTMO one-stage: kecepatan konstan berapa pun jumlah orang
    python rtmpose_cctv.py --source rekaman.mp4 --preset rtmo-m --device cuda --trt

Output:
    - Video dengan skeleton (--out)
    - File keypoints per frame (--kpt-out, format JSON Lines):
      {"frame": 12, "t": 0.48, "people": [{"bbox": [...], "kpts": [[x, y, score], ...17 titik]}]}
"""

import argparse
import importlib
import json
import os
import time

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"

# Kombinasi detektor orang + pose estimator.
# pose_input_size ditulis (lebar, tinggi).
PRESETS = {
    # Cepat, cocok untuk CPU atau banyak kamera
    "ringan": dict(
        det=BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_input_size=(416, 416),
        pose=BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
    ),
    # Seimbang, titik awal yang disarankan
    "seimbang": dict(
        det=BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose=BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
    ),
    # Lebih akurat untuk orang kecil/jauh dan gambar blur (butuh GPU agar real-time)
    "akurat": dict(
        det=BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose=BASE + "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip",
        pose_input_size=(288, 384),
    ),
    # --- RTMO: one-stage (tanpa detektor terpisah), kecepatan KONSTAN berapa pun
    # jumlah orangnya. Lisensi Apache-2.0. Input 640x640.
    # rtmlib memilih bobot RTMO lewat `mode`, jadi `pose` cukup diisi "rtmo".
    "rtmo-s": dict(pose="rtmo", mode="lightweight"),   # COCO AP 68.6
    "rtmo-m": dict(pose="rtmo", mode="balanced"),      # COCO AP 72.6
    "rtmo-l": dict(pose="rtmo", mode="performance"),   # COCO AP 74.8
}

# Urutan 17 keypoint COCO (indeks dipakai saat menghitung fitur nanti)
COCO_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def prepare_gpu_libs():
    """Muat DLL CUDA/cuDNN (dari paket pip nvidia-*) dan TensorRT sebelum sesi dibuat.

    onnxruntime-gpu >= 1.21 punya preload_dlls() yang mencari DLL di paket pip
    nvidia-cublas/cudnn/... Kalau CUDA & cuDNN terpasang di sistem dan sudah di PATH,
    langkah ini tidak berpengaruh apa-apa.
    """
    add_tensorrt_dll_path()
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as e:
        print(f"[GPU] preload_dlls gagal (diabaikan): {e}")


def report_providers(body):
    """Cetak provider yang benar-benar aktif untuk tiap model."""
    for name in ("det_model", "pose_model"):
        m = getattr(body, name, None)
        sess = getattr(m, "session", None)
        if sess is None or not hasattr(sess, "get_providers"):
            continue
        active = sess.get_providers()
        print(f"[GPU] {name}: provider aktif = {active[0]}")
        if active[0] == "CPUExecutionProvider":
            print(f"[GPU] PERINGATAN: {name} jalan di CPU! DLL CUDA/cuDNN tidak ditemukan "
                  f"atau versinya tidak cocok dengan onnxruntime-gpu (lihat log di atas).")


def add_tensorrt_dll_path():
    """Windows: daftarkan folder DLL TensorRT dari paket pip (tensorrt-cu12/cu13) ke PATH,
    supaya TensorrtExecutionProvider bisa menemukan nvinfer_*.dll."""
    if os.name != "nt":
        return
    for mod in ("tensorrt_libs", "tensorrt"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        d = os.path.dirname(m.__file__)
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass


def enable_tensorrt(body, fp16=True, cache_dir="trt_cache", device_id=0):
    """Ganti sesi onnxruntime milik detektor & pose model dengan TensorRT EP.

    rtmlib secara default hanya memakai CUDAExecutionProvider, jadi sesi dibuat ulang
    dengan urutan provider: TensorRT -> CUDA -> CPU (fallback otomatis kalau ada op
    yang tidak didukung TensorRT).

    Engine TensorRT dibangun saat inferensi PERTAMA (bisa 1-5 menit per model),
    lalu disimpan di cache_dir sehingga run berikutnya langsung cepat.
    """
    import onnxruntime as ort
    from rtmlib.tools.file import download_checkpoint

    add_tensorrt_dll_path()
    os.makedirs(cache_dir, exist_ok=True)

    trt_opts = {
        "device_id": device_id,
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
        "trt_builder_optimization_level": 3,
        "trt_max_workspace_size": 2 * 1024 ** 3,
    }
    providers = [
        ("TensorrtExecutionProvider", trt_opts),
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]

    for name in ("det_model", "pose_model"):
        m = getattr(body, name, None)
        if m is None:
            continue
        path = m.onnx_model
        if not os.path.exists(path):
            path = download_checkpoint(path)   # ambil dari cache lokal rtmlib
        m.session = ort.InferenceSession(path, providers=providers)
        if m.session.get_providers()[0] == "CUDAExecutionProvider":
            print(f"[TRT] {name}: TensorRT tidak termuat, jalan di CUDA "
                  f"(cek DLL TensorRT di PATH, versinya harus cocok dengan CUDA onnxruntime).")


def parse_args():
    p = argparse.ArgumentParser(description="RTMPose untuk video CCTV")
    p.add_argument("--source", required=True,
                   help="path video, URL RTSP, atau indeks kamera (mis. 0)")
    p.add_argument("--preset", default="seimbang", choices=PRESETS.keys())
    p.add_argument("--device", default="cpu", help="cpu atau cuda")
    p.add_argument("--backend", default="onnxruntime",
                   help="onnxruntime, openvino, atau opencv")
    p.add_argument("--kpt-thr", type=float, default=0.3,
                   help="ambang confidence untuk menggambar keypoint")
    p.add_argument("--out", default="", help="simpan video hasil (mis. hasil.mp4)")
    p.add_argument("--kpt-out", default="keypoints.jsonl",
                   help="simpan keypoints per frame ('' untuk mematikan)")
    p.add_argument("--show", action="store_true", help="tampilkan jendela preview")
    p.add_argument("--max-frames", type=int, default=0, help="0 = semua frame")
    p.add_argument("--trt", action="store_true",
                   help="pakai TensorRT (butuh --device cuda)")
    p.add_argument("--fp32", action="store_true",
                   help="matikan FP16 di TensorRT (lebih presisi, lebih lambat)")
    p.add_argument("--trt-cache", default="trt_cache",
                   help="folder cache engine TensorRT")
    p.add_argument("--skip", type=int, default=1,
                   help="proses 1 dari tiap N frame (mis. 2 = setengah FPS sumber)")
    return p.parse_args()


def bbox_from_kpts(kpts, scores, thr=0.3):
    """Perkiraan bounding box dari keypoint yang cukup yakin."""
    valid = kpts[scores > thr]
    if len(valid) == 0:
        valid = kpts
    x1, y1 = valid.min(axis=0)
    x2, y2 = valid.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Tidak bisa membuka sumber video: {args.source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Sumber: {w}x{h} @ {fps_src:.1f} FPS | preset={args.preset} device={args.device}")

    if "cuda" in args.device:
        prepare_gpu_libs()

    # Model diunduh otomatis saat pertama kali dijalankan
    model = Body(**PRESETS[args.preset], backend=args.backend, device=args.device)

    if args.trt:
        if "cuda" not in args.device:
            raise SystemExit("--trt butuh --device cuda")
        enable_tensorrt(model, fp16=not args.fp32, cache_dir=args.trt_cache)
    report_providers(model)

    # Warm-up: memicu pembangunan engine TensorRT (lama hanya di run pertama)
    # supaya tidak terhitung di FPS.
    print("Warm-up model..." + (" (build engine TensorRT pertama kali bisa beberapa menit)"
                                 if args.trt else ""))
    t_w = time.time()
    dummy = np.zeros((max(h, 64), max(w, 64), 3), dtype=np.uint8)
    model(dummy)   # top-down: detektor | RTMO: seluruh model
    if not getattr(model, "one_stage", False):
        # top-down: pose model hanya jalan kalau ada orang, jadi dipanggil manual
        model.pose_model(dummy, bboxes=[[0, 0, dummy.shape[1], dummy.shape[0]]])
    print(f"Warm-up selesai dalam {time.time() - t_w:.1f} s")

    skip = max(1, args.skip)
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps_src / skip, (w, h))

    kpt_file = open(args.kpt_out, "w") if args.kpt_out else None

    read_idx, frame_idx, t_start = -1, 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_idx += 1
            if read_idx % skip:
                continue

            t0 = time.time()
            # keypoints: (N, 17, 2) koordinat piksel, scores: (N, 17) confidence
            keypoints, scores = model(frame)
            infer_ms = (time.time() - t0) * 1000

            if kpt_file is not None:
                people = []
                for k, s in zip(keypoints, scores):
                    people.append({
                        "bbox": bbox_from_kpts(k, s),
                        "kpts": np.concatenate([k, s[:, None]], axis=1).round(2).tolist(),
                    })
                # "frame" = indeks frame asli di video (tetap benar walau pakai --skip)
                kpt_file.write(json.dumps({"frame": read_idx,
                                           "t": round(read_idx / fps_src, 3),
                                           "people": people}) + "\n")

            if writer is not None or args.show:
                vis = draw_skeleton(frame.copy(), keypoints, scores,
                                    kpt_thr=args.kpt_thr, openpose_skeleton=False)
                cv2.putText(vis, f"{len(keypoints)} orang | {infer_ms:.0f} ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("RTMPose CCTV", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
            if frame_idx % 50 == 0:
                fps = frame_idx / (time.time() - t_start)
                print(f"frame {frame_idx} | rata-rata {fps:.1f} FPS | terakhir {infer_ms:.0f} ms")
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if kpt_file is not None:
            kpt_file.close()
        if args.show:
            cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"Selesai: {frame_idx} frame dalam {total:.1f} s ({frame_idx / max(total, 1e-6):.1f} FPS)")


if __name__ == "__main__":
    main()