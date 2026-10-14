# CCTV Insight — Model Research Scripts Documentation

Complete guide untuk menjalankan semua scripts di `C:\cctv-insight\research_amir\model_research\`.

---

## 📁 Folder Structure

```
model_research/
├── gstream/                  # GStreamer NVDEC experiments
│   ├── modelengine_gst.py
│   ├── modelengine_gst_q.py
│   └── modelengine_gst_shm.py
│
├── model_compare/            # TensorRT vs PyTorch comparison
│   ├── comparasion_model.py
│   ├── modelengine.py
│   └── modelpt.py
│
└── multi_video/              # Multi-stream processing
    ├── modelengine_batch.py
    ├── modelengine_batch_display.py
    ├── modelengine_multi_cv2.py
    └── modelengine_multi_view.py
```

---

## 🔧 Prerequisites

### 1. Setup Environment PATH (untuk GStreamer scripts)
```powershell
$env:PATH = "C:\Users\istd.magang17\AppData\Local\Programs\gstreamer\1.0\msvc_x86_64\bin;" + $env:PATH
```

### 2. Python Environment
- **GStreamer scripts**: Gunakan `ai\.venv` (punya torch CUDA)
- **Compare scripts**: Gunakan `ai\.venv` (punya torch CUDA)

### 3. Test Video Files
```
C:\cctv-dev-temp\
├── stairs_people.mp4
├── road.mp4
├── office.mp4
├── plant.mp4
└── street.mp4
```

---

## 📜 Script Documentation

---

### 📁 `gstream/` — GStreamer NVDEC Experiments

---

#### 1. `modelengine_gst.py`
**Fungsi**: Single stream dengan GStreamer NVDEC GPU decode + TensorRT inference

**Cara Jalankan**:
```powershell
# Basic (video file)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\gstream\modelengine_gst.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --no-display `
    --max-frames 300

# Dengan RTSP stream
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\gstream\modelengine_gst.py" `
    --source "rtsp://localhost:8554/stairs-cam" `
    --quantize 16 `
    --no-display

# Dengan display window
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\gstream\modelengine_gst.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16
```

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--source` | stairs_people.mp4 | Video file atau RTSP URL |
| `--quantize` | 0 (FP32) | 0=FP32, 16=FP16 |
| `--no-display` | false | Disable window display |
| `--max-frames` | 0 (unlimited) | Stop setelah N frames |
| `--log-csv` | "" | Log path ke CSV |

---

#### 2. `modelengine_gst_q.py`
**Fungsi**: GStreamer NVDEC dengan Multiprocessing Queue (optimized transfer)

**Cara Jalankan**:
```powershell
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\gstream\modelengine_gst_q.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --no-display `
    --max-frames 300
```

**Kelebihan**: Lebih stabil dari `modelengine_gst.py` untuk pipe transfer.

---

#### 3. `modelengine_gst_shm.py`
**Fungsi**: GStreamer NVDEC dengan Shared Memory (zero-copy experiment)

**Cara Jalankan**:
```powershell
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\gstream\modelengine_gst_shm.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --no-display `
    --max-frames 300
```

**Note**: Experimental — shared memory di Windows butuh setup khusus.

---

### 📁 `model_compare/` — TensorRT vs PyTorch Benchmark

---

#### 4. `comparasion_model.py`
**Fungsi**: Benchmark comparison TensorRT (.engine) vs PyTorch (.pt)

**Cara Jalankan**:
```powershell
# Default comparison (PPE model)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\comparasion_model.py" `
    --model ppe_compliance `
    --runs 200 `
    --warmup 20

# Model lain
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\comparasion_model.py" `
    --model security_officer_v2 `
    --runs 300 `
    --quantize 0 `
    --out "results.csv"

# Custom model path
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\comparasion_model.py" `
    --model "C:/path/to/custom.pt" `
    --runs 100
```

**Output**: Tabel perbandingan FPS, inference time, speedup factor.

**Available Models**:
- `ppe_compliance`
- `security_officer_v2`
- `helmet_and_vest`
- `face_mask`
- `hand_in_pocket`
- `phone_usage`
- `people_counting`

---

#### 5. `modelengine.py`
**Fungsi**: Single stream TensorRT inference dengan monitoring (CPU/GPU/RAM stats)

**Cara Jalankan**:
```powershell
# Basic (headless benchmark)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelengine.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --no-display `
    --max-frames 300

# Dengan display window
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelengine.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16

# Low latency mode (aggressive frame dropping)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelengine.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --low-latency `
    --fast-display

# Dengan profiling detail
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelengine.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --quantize 16 `
    --no-display `
    --log-csv "benchmark.csv"
```

**Options**:
| Option | Description |
|--------|-------------|
| `--low-latency` | Skip frames untuk real-time priority |
| `--fast-display` | Skip bounding box plotting |
| `--display-skip N` | Update window setiap N frames |
| `--dnn-preprocess` | Gunakan OpenCV DNN CUDA untuk preprocessing |

---

#### 6. `modelpt.py`
**Fungsi**: Single stream PyTorch (.pt) inference dengan monitoring

**Cara Jalankan**:
```powershell
# Basic
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelpt.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --no-display `
    --max-frames 300

# Dengan display
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelpt.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4"

# Low latency
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\model_compare\modelpt.py" `
    --source "C:/cctv-dev-temp/stairs_people.mp4" `
    --low-latency `
    --fast-display
```

**Use Case**: Compare performance dengan `modelengine.py` (TensorRT).

---

### 📁 `multi_video/` — Multi-Stream Processing

---

#### 7. `modelengine_batch.py`
**Fungsi**: Multi-stream batch processing (headless, production-ready)

**Cara Jalankan**:
```powershell
# 4 streams
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_batch.py" `
    --sources `
    "C:/cctv-dev-temp/stairs_people.mp4" `
    "C:/cctv-dev-temp/road.mp4" `
    "C:/cctv-dev-temp/office.mp4" `
    "C:/cctv-dev-temp/plant.mp4" `
    --quantize 16 `
    --no-display `
    --max-frames 500

# 8 streams (maximize GPU)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_batch.py" `
    --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 cam5.mp4 cam6.mp4 cam7.mp4 cam8.mp4 `
    --quantize 16 `
    --no-display `
    --log-csv "8streams.csv"

# RTSP streams (CCTV live)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_batch.py" `
    --sources `
    "rtsp://camera1-ip/stream" `
    "rtsp://camera2-ip/stream" `
    "rtsp://camera3-ip/stream" `
    "rtsp://camera4-ip/stream" `
    --quantize 16
```

**Expected Output**:
```
[  60] FPS ~  9.6 | dec   1.5ms | inf   6.9ms | CPU  38% | GPU  57% | VRAM 1085MB | S0:10 | S1:22 | S2:27 | S3:1
[ 120] FPS ~ 17.0 | dec   1.2ms | inf   8.0ms | CPU  46% | GPU  66% | VRAM 1085MB | S0:18 | S1:50 | S2:51 | S3:1
```

---

#### 8. `modelengine_batch_display.py`
**Fungsi**: Multi-stream dengan Grid Display (semua streams dalam 1 window)

**Cara Jalankan**:
```powershell
# 2x2 grid (4 streams)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_batch_display.py" `
    --sources `
    "C:/cctv-dev-temp/stairs_people.mp4" `
    "C:/cctv-dev-temp/road.mp4" `
    "C:/cctv-dev-temp/office.mp4" `
    "C:/cctv-dev-temp/plant.mp4" `
    --quantize 16

# Tekan 'q' di window untuk stop
```

**Display**: 1 window dengan layout NxM grid, bounding boxes + stats overlay.

---

#### 9. `modelengine_multi_cv2.py`
**Fungsi**: Multi-stream dengan OpenCV cv2.VideoCapture (tanpa GStreamer)

**Cara Jalankan**:
```powershell
# 4 streams (simple, no GStreamer)
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_multi_cv2.py" `
    --sources `
    "C:/cctv-dev-temp/stairs_people.mp4" `
    "C:/cctv-dev-temp/road.mp4" `
    "C:/cctv-dev-temp/office.mp4" `
    "C:/cctv-dev-temp/plant.mp4" `
    --quantize 16 `
    --display-fps 24

# 2 streams test
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_multi_cv2.py" `
    --sources `
    "C:/cctv-dev-temp/stairs_people.mp4" `
    "C:/cctv-dev-temp/road.mp4" `
    --quantize 16 `
    --display-fps 20
```

**Display**: 4 windows terpisah (1 per stream).

**Kelebihan**: No GStreamer dependency, lebih simple setup.

---

#### 10. `modelengine_multi_view.py`
**Fungsi**: Multi-stream dengan Individual Windows + GStreamer NVDEC

**Cara Jalankan**:
```powershell
# 4 streams dengan GStreamer NVDEC
$env:PATH = "C:\Users\istd.magang17\AppData\Local\Programs\gstreamer\1.0\msvc_x86_64\bin;" + $env:PATH
& "C:\cctv-insight\ai\.venv\Scripts\python.exe" "C:\cctv-insight\research_amir\model_research\multi_video\modelengine_multi_view.py" `
    --sources `
    "C:/cctv-dev-temp/stairs_people.mp4" `
    "C:/cctv-dev-temp/road.mp4" `
    "C:/cctv-dev-temp/office.mp4" `
    "C:/cctv-dev-temp/plant.mp4" `
    --quantize 16 `
    --display-fps 24
```

**Display**: 4 windows terpisah dengan bounding boxes.

---

## 📊 Quick Reference Table

| Script | Display | Streams | GStreamer | Use Case |
|--------|---------|---------|-----------|----------|
| `modelengine_gst.py` | Optional | 1 | ✅ | Single stream NVDEC test |
| `modelengine_gst_q.py` | Optional | 1 | ✅ | Queue optimization test |
| `modelengine_gst_shm.py` | Optional | 1 | ✅ | Shared memory experiment |
| `comparasion_model.py` | No | 1 | ❌ | TRT vs PT benchmark |
| `modelengine.py` | Optional | 1 | ❌ | Single stream monitoring |
| `modelpt.py` | Optional | 1 | ❌ | PyTorch baseline |
| `modelengine_batch.py` | No | 2-8 | ✅ | Production multi-stream |
| `modelengine_batch_display.py` | ✅ Grid | 2-8 | ✅ | Monitoring grid view |
| `modelengine_multi_cv2.py` | ✅ Windows | 2-8 | ❌ | Simple multi-stream |
| `modelengine_multi_view.py` | ✅ Windows | 2-8 | ✅ | NVDEC multi-stream |

---

## 🎯 Common Use Cases

### 1. Benchmark TensorRT vs PyTorch
```powershell
python model_compare\comparasion_model.py --model ppe_compliance --runs 300
```

### 2. Production 4-Camera CCTV
```powershell
python multi_video\modelengine_batch.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16 --no-display
```

### 3. Monitoring dengan Grid View
```powershell
python multi_video\modelengine_batch_display.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
```

### 4. Simple Multi-Stream (No GStreamer)
```powershell
python multi_video\modelengine_multi_cv2.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16 --display-fps 24
```

### 5. Maximize GPU Utilization (8 streams)
```powershell
python multi_video\modelengine_batch.py --sources cam1.mp4 ... cam8.mp4 --quantize 16 --no-display --log-csv 8streams.csv
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError: tensorrt** | Gunakan `ai\.venv` bukan `.venv-trt` |
| **Invalid CUDA device** | Pastikan pakai `ai\.venv` (punya torch CUDA) |
| **GStreamer pipeline error** | Check PATH: `$env:PATH = "C:\...\gstreamer\1.0\msvc_x86_64\bin;" + $env:PATH` |
| **Video patah-patah** | Tambah `--display-fps 20` atau kurangi streams |
| **All streams exit cepat** | Video files habis — pakai RTSP atau script auto-loop |
| **GPU utilization rendah** | Tambah streams (4 → 8) |

---

**Last Updated**: 2026-08-26  
**Maintained By**: AI Research Team
