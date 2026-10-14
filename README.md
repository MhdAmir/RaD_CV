# RaD_CV — CCTV Insight Research

Kumpulan script riset computer vision untuk CCTV: deteksi orang/HP, klasifikasi,
pose, benchmark PyTorch vs TensorRT, dan multi-stream inference.

## Struktur Folder

```
research_amir/
├── algorithm_research/     # Algoritma deteksi perilaku
│   ├── use_phone/          #   deteksi pakai HP berbasis pose (YOLO-Pose / RTMPose + gates)
│   └── cascade/            #   eksperimen Haar cascade (masker) & YOLO pakai HP
├── model_research/         # Eksperimen model
│   ├── model_compare/      #   RF-DETR, YOLO .pt vs TensorRT FP32/FP16, export engine
│   ├── multi_video/        #   inference multi-stream / multi-kamera
│   ├── gstream/            #   decode GPU via GStreamer NVDEC
│   ├── classification/     #   DINOv3 classifier (train & inference)
│   ├── detection-classification/  # YOLO + DINO / MobileNet (2 tahap)
│   ├── keypoints/          #   RTMPose keypoint
│   └── depth/              #   YOLO + depth
├── benchmarks/             # Script benchmark batching + hasil CSV/PNG + report
├── tools/                  # Script utilitas (mis. rapikan dataset)
├── docs/how-to-run.md      # Dokumentasi detail opsi tiap script
└── requirements/           # Daftar package per venv
```

Yang **tidak** ikut di repo (lihat `.gitignore`): venv, model/engine
(`.pt`, `.pth`, `.onnx`, `.trt`, `.keras`, `.bin`), gambar, video, dan folder
hasil (`out/`, `output/`, `output_768/`, `trt_cache/`).

## Prasyarat

- Windows + GPU NVIDIA (driver & CUDA terbaru)
- Python **3.14** (venv utama) dan Python **3.12** (venv tambahan)
- (Opsional) GStreamer, hanya untuk script di `model_research/gstream/`

## Setup Awal

```powershell
git clone https://github.com/MhdAmir/RaD_CV.git
cd RaD_CV

# Venv utama (Python 3.14) — dipakai hampir semua script
py -3.14 -m venv .venv-trt
.venv-trt\Scripts\activate
pip install -r requirements\requirements-trt.txt
```

Venv tambahan, buat kalau memang perlu:

| Venv | Python | Requirements | Dipakai untuk |
|------|--------|--------------|---------------|
| `.venv-trt` | 3.14 | `requirements-trt.txt` | **Utama**: RF-DETR, YOLO, TensorRT, RTMPose, Keras, SAHI |
| `.venv-py312` | 3.12 | `requirements-py312.txt` | Export YOLO ke TensorRT **FP16** (butuh `nvidia-modelopt`) |
| `.venv-infm` | 3.12 | `requirements-infm.txt` | Eksperimen Roboflow `inference-models` |

```powershell
py -3.12 -m venv .venv-py312
.venv-py312\Scripts\pip install -r requirements\requirements-py312.txt
```

> Package `torch` versi CUDA (`+cu126` / `+cu130`) tidak ada di PyPI biasa. Kalau
> `pip install` gagal di torch, install torch dulu dari
> https://pytorch.org/get-started/locally/ lalu ulangi `pip install -r ...`.

## Siapkan Model & Video

Model dan video tidak di-commit, jadi siapkan sendiri:

- **Video uji** → taruh di `C:\cctv-dev-temp\` (mis. `stairs_people.mp4`, `use_phone_5.MOV`)
- **Model YOLO** → `C:\cctv-insight\ai\models\` (mis. `yolo26m-pose.pt`)
- **Engine/ONNX RF-DETR** → `out/` atau `output/` (dibuat ulang lewat script export, lihat di bawah)
- **Lifter 3D** → `models/lifter/pretrained_h36m_detectron_coco.bin`

Path model/video ada sebagai konstanta di **bagian atas tiap script**
(mis. `ENGINE_PATH = ...`). Kalau lokasinya beda, ubah di situ atau pakai
argumen `--source` / `--model` kalau script-nya punya.

## Cara Menjalankan

Aktifkan venv dulu: `.venv-trt\Scripts\activate`. Semua contoh dijalankan dari root repo.

**Deteksi pakai HP (pose-based)**
```powershell
python algorithm_research\use_phone\research_pose.py --source C:\cctv-dev-temp\DEV_Gerbang_Depan.mp4 --display
```
Hasilnya (video + CSV log) masuk ke `algorithm_research/use_phone/research_output/`.

**RF-DETR TensorRT FP16 (auto-export `.pth` → engine kalau belum ada)**
```powershell
python model_research\model_compare\rfdetr-model.py
```

**Benchmark PyTorch vs TensorRT**
```powershell
python model_research\model_compare\benchmark_all.py --model people_counting --source C:\cctv-dev-temp\stairs_people.mp4 --runs 500
```

**Multi-stream (4 kamera, grid display)**
```powershell
python model_research\multi_video\modelengine_batch_display.py --sources cam1.mp4 cam2.mp4 cam3.mp4 cam4.mp4 --quantize 16
```

**Keypoint RTMPose**
```powershell
python model_research\keypoints\rtmpose-l.py --source C:\cctv-dev-temp\stairs_people.mp4
```

**Benchmark batching**
```powershell
python benchmarks\benchmark_batching.py --scenario all --max-frames 500
```

Opsi lengkap tiap script: jalankan `python <script> --help`, atau lihat
[docs/how-to-run.md](docs/how-to-run.md). Hasil benchmark ada di
[benchmarks/BENCHMARK_REPORT.md](benchmarks/BENCHMARK_REPORT.md).

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: tensorrt` / `rfdetr` | Pastikan venv `.venv-trt` aktif |
| Export FP16 YOLO gagal | Pakai `.venv-py312` (punya `nvidia-modelopt`) |
| `FileNotFoundError` model/video | Cek konstanta path di atas script |
| Error pipeline GStreamer | Tambahkan folder `gstreamer\1.0\msvc_x86_64\bin` ke `PATH` |
