import cv2
import time
import numpy as np
import tensorrt as trt

from pathlib import Path
from cuda.bindings import runtime


# ============================================================
# CONFIG
# ============================================================

# Boleh .pth (checkpoint) atau .trt (engine).
# Kalau .pth -> otomatis di-export ke FP16 TensorRT engine di OUT_DIR.
MODEL_PATH = r"C:\Users\istd.magang17\Downloads\hasil\runs\sh17_rfdetr_medium\checkpoint_best_total.pth"

OUT_DIR = Path("./out")

INPUT_SIZE = 576
CONF_THRESHOLD = 0.5

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# AUTO EXPORT .pth -> FP16 .trt
# ============================================================

def convert_onnx_to_fp16(onnx_path, onnx_fp16_path):
    """
    FP32 ONNX -> FP16 ONNX (termasuk I/O tensor).

    TensorRT 11 menghapus BuilderFlag.FP16 (semua network STRONGLY_TYPED),
    jadi rfdetr.export(format="tensorrt") diam-diam jatuh ke FP32.
    Precision harus ada di ONNX-nya sendiri.
    """

    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))

    # keep_io_types=True tidak menyisipkan Cast di input untuk graph ini,
    # jadi input/output ikut FP16 (dtype dibaca dari engine saat inference).
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=False
    )

    # Cast(to=float32) bawaan export tidak diubah converter,
    # dan di strongly-typed network bikin Conv/MatMul FP32 x FP16 bentrok.
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16

    onnx.save(model, str(onnx_fp16_path))


def build_engine(onnx_path, engine_path):
    """
    Build TensorRT engine tanpa precision flag
    (precision mengikuti dtype di ONNX).
    """

    from polygraphy.backend.trt import (
        CreateConfig,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )

    engine = engine_from_network(
        network_from_onnx_path(str(onnx_path)),
        config=CreateConfig()
    )

    save_engine(engine, path=str(engine_path))


def ensure_fp16_engine(model_path, out_dir):
    """
    Return path ke FP16 TensorRT engine.

    - .trt / .engine : langsung dipakai.
    - .pth           : .pth -> ONNX -> FP16 ONNX -> FP16 .trt di out_dir
                       (tiap tahap di-skip kalau file-nya sudah ada).

    https://rfdetr.roboflow.com/latest/exports/tensorrt/
    """

    model_path = Path(model_path)

    if model_path.suffix.lower() in (".trt", ".engine"):
        return model_path

    if model_path.suffix.lower() != ".pth":
        raise ValueError(
            f"Unsupported model file: {model_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # e.g. sh17_rfdetr_medium_checkpoint_best_total
    name = f"{model_path.parent.name}_{model_path.stem}"

    onnx_path = out_dir / f"{name}.onnx"
    onnx_fp16_path = out_dir / f"{name}_fp16.onnx"
    engine_path = out_dir / f"{name}_fp16.trt"

    if engine_path.is_file():
        print(f"[INFO] FP16 engine sudah ada: {engine_path}")
        return engine_path

    # --------------------------------------------------------
    # 1. .pth -> ONNX (FP32)
    # --------------------------------------------------------

    if not onnx_path.is_file():

        print(f"[INFO] Exporting {model_path.name} -> ONNX...")

        # Import di sini supaya rfdetr hanya dibutuhkan saat export
        from rfdetr import RFDETRMedium

        model = RFDETRMedium(
            pretrain_weights=str(model_path)
        )

        onnx_path = Path(
            model.export(
                format="onnx",
                output_dir=str(out_dir),
                output_name=name,
                verbose=False
            )
        )

    # --------------------------------------------------------
    # 2. ONNX -> FP16 ONNX
    # --------------------------------------------------------

    if not onnx_fp16_path.is_file():

        print("[INFO] Converting ONNX -> FP16...")

        convert_onnx_to_fp16(
            onnx_path,
            onnx_fp16_path
        )

    # --------------------------------------------------------
    # 3. FP16 ONNX -> TensorRT engine
    # --------------------------------------------------------

    print("[INFO] Building FP16 TensorRT engine (~1-2 menit)...")

    build_engine(
        onnx_fp16_path,
        engine_path
    )

    print(f"[INFO] Engine saved: {engine_path}")

    return engine_path


ENGINE_PATH = ensure_fp16_engine(
    MODEL_PATH,
    OUT_DIR
)


# ============================================================
# CUDA ERROR CHECK
# ============================================================

def check_cuda(result, operation="CUDA operation"):
    """
    Check CUDA Runtime API return value.

    cuda.bindings runtime functions may return:
        (error,)
    or:
        (error, value)
    """

    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"{operation} failed: {err}"
        )


# ============================================================
# LOAD TENSORRT ENGINE
# ============================================================

print("[INFO] Loading TensorRT engine...")

with open(ENGINE_PATH, "rb") as f:
    engine_data = f.read()

trt_runtime = trt.Runtime(TRT_LOGGER)

engine = trt_runtime.deserialize_cuda_engine(
    engine_data
)

if engine is None:
    raise RuntimeError(
        "Failed to load TensorRT engine"
    )

context = engine.create_execution_context()

if context is None:
    raise RuntimeError(
        "Failed to create TensorRT execution context"
    )

print("[INFO] Engine loaded.")


# ============================================================
# TENSOR NAMES
# ============================================================

input_name = "input"
dets_name = "dets"
labels_name = "labels"


# ============================================================
# PRINT TENSOR INFORMATION
# ============================================================

print()
print("[INFO] Tensor information")

input_shape = tuple(
    engine.get_tensor_shape(input_name)
)

dets_shape = tuple(
    engine.get_tensor_shape(dets_name)
)

labels_shape = tuple(
    engine.get_tensor_shape(labels_name)
)

# Ikuti resolusi engine (NCHW)
INPUT_SIZE = int(input_shape[-1])

input_trt_dtype = engine.get_tensor_dtype(
    input_name
)

dets_trt_dtype = engine.get_tensor_dtype(
    dets_name
)

labels_trt_dtype = engine.get_tensor_dtype(
    labels_name
)

print(
    f"  Input  : {input_shape} "
    f"{input_trt_dtype}"
)

print(
    f"  Dets   : {dets_shape} "
    f"{dets_trt_dtype}"
)

print(
    f"  Labels : {labels_shape} "
    f"{labels_trt_dtype}"
)

print()


# ============================================================
# CONVERT TRT DTYPE -> NUMPY DTYPE
# ============================================================

input_dtype = np.dtype(
    trt.nptype(input_trt_dtype)
)

dets_dtype = np.dtype(
    trt.nptype(dets_trt_dtype)
)

labels_dtype = np.dtype(
    trt.nptype(labels_trt_dtype)
)

print("[INFO] NumPy dtypes")

print(
    f"  Input  : {input_dtype}"
)

print(
    f"  Dets   : {dets_dtype}"
)

print(
    f"  Labels : {labels_dtype}"
)

print()


# ============================================================
# BUFFER SIZES
# ============================================================

input_nbytes = (
    int(np.prod(input_shape))
    * input_dtype.itemsize
)

dets_nbytes = (
    int(np.prod(dets_shape))
    * dets_dtype.itemsize
)

labels_nbytes = (
    int(np.prod(labels_shape))
    * labels_dtype.itemsize
)

print("[INFO] Buffer sizes")

print(
    f"  Input  : {input_nbytes / 1024:.2f} KB"
)

print(
    f"  Dets   : {dets_nbytes / 1024:.2f} KB"
)

print(
    f"  Labels : {labels_nbytes / 1024:.2f} KB"
)

print()


# ============================================================
# HOST MEMORY
# ============================================================

h_input = np.empty(
    input_shape,
    dtype=input_dtype
)

h_dets = np.empty(
    dets_shape,
    dtype=dets_dtype
)

h_labels = np.empty(
    labels_shape,
    dtype=labels_dtype
)


# ============================================================
# CUDA STREAM
# ============================================================

print("[INFO] Creating CUDA stream...")

stream_result = runtime.cudaStreamCreate()

check_cuda(
    stream_result,
    "cudaStreamCreate"
)

stream = stream_result[1]

print("[INFO] CUDA stream created.")


# ============================================================
# GPU MEMORY
# ============================================================

print("[INFO] Allocating CUDA memory...")

input_result = runtime.cudaMalloc(
    input_nbytes
)

check_cuda(
    input_result,
    "cudaMalloc input"
)

d_input = input_result[1]


dets_result = runtime.cudaMalloc(
    dets_nbytes
)

check_cuda(
    dets_result,
    "cudaMalloc dets"
)

d_dets = dets_result[1]


labels_result = runtime.cudaMalloc(
    labels_nbytes
)

check_cuda(
    labels_result,
    "cudaMalloc labels"
)

d_labels = labels_result[1]

print("[INFO] CUDA memory allocated.")


# ============================================================
# SET TENSOR ADDRESSES
# ============================================================

print("[INFO] Setting TensorRT tensor addresses...")

if not context.set_tensor_address(
    input_name,
    int(d_input)
):
    raise RuntimeError(
        f"Failed to set tensor address: {input_name}"
    )

if not context.set_tensor_address(
    dets_name,
    int(d_dets)
):
    raise RuntimeError(
        f"Failed to set tensor address: {dets_name}"
    )

if not context.set_tensor_address(
    labels_name,
    int(d_labels)
):
    raise RuntimeError(
        f"Failed to set tensor address: {labels_name}"
    )

print("[INFO] Tensor addresses configured.")


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(frame):

    original_h, original_w = frame.shape[:2]

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        frame,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Normalize 0-1
    # --------------------------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )

    # --------------------------------------------------------
    # Convert to engine dtype
    # --------------------------------------------------------

    image = np.ascontiguousarray(
        image,
        dtype=input_dtype
    )

    return (
        image,
        original_w,
        original_h
    )


# ============================================================
# INFERENCE
# ============================================================

def inference(frame):

    image, original_w, original_h = preprocess(
        frame
    )

    # --------------------------------------------------------
    # Copy image into host buffer
    # --------------------------------------------------------

    np.copyto(
        h_input,
        image
    )

    # --------------------------------------------------------
    # CPU -> GPU
    # --------------------------------------------------------

    result = runtime.cudaMemcpyAsync(
        int(d_input),
        h_input.ctypes.data,
        input_nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
        stream
    )

    check_cuda(
        result,
        "cudaMemcpyAsync H2D"
    )

    # --------------------------------------------------------
    # TensorRT inference
    # --------------------------------------------------------

    start = time.perf_counter()

    success = context.execute_async_v3(
        stream_handle=int(stream)
    )

    if not success:
        raise RuntimeError(
            "TensorRT execute_async_v3() failed"
        )

    # --------------------------------------------------------
    # GPU -> CPU : Dets
    # --------------------------------------------------------

    result = runtime.cudaMemcpyAsync(
        h_dets.ctypes.data,
        int(d_dets),
        dets_nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )

    check_cuda(
        result,
        "cudaMemcpyAsync D2H dets"
    )

    # --------------------------------------------------------
    # GPU -> CPU : Labels
    # --------------------------------------------------------

    result = runtime.cudaMemcpyAsync(
        h_labels.ctypes.data,
        int(d_labels),
        labels_nbytes,
        runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )

    check_cuda(
        result,
        "cudaMemcpyAsync D2H labels"
    )

    # --------------------------------------------------------
    # Synchronize
    # --------------------------------------------------------

    result = runtime.cudaStreamSynchronize(
        stream
    )

    check_cuda(
        result,
        "cudaStreamSynchronize"
    )

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    return (
        h_dets.copy(),
        h_labels.copy(),
        latency,
        original_w,
        original_h
    )


# ============================================================
# POSTPROCESS
# ============================================================

def postprocess(
    dets,
    labels,
    original_w,
    original_h
):

    detections = []

    dets = dets[0]
    labels = labels[0]

    num_detections = min(
        dets.shape[0],
        labels.shape[0]
    )

    for i in range(num_detections):

        # ----------------------------------------------------
        # Class scores
        # ----------------------------------------------------

        # labels = logits -> sigmoid jadi probabilitas
        scores = 1.0 / (
            1.0 + np.exp(-labels[i].astype(np.float32))
        )

        class_id = int(
            np.argmax(scores)
        )

        confidence = float(
            scores[class_id]
        )

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        if confidence < CONF_THRESHOLD:
            continue

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        box = dets[i]

        cx, cy, w, h = box

        cx = float(cx)
        cy = float(cy)
        w = float(w)
        h = float(h)

        # ----------------------------------------------------
        # RF-DETR normalized coordinates
        # ----------------------------------------------------

        x1 = (
            cx - w / 2
        ) * original_w

        y1 = (
            cy - h / 2
        ) * original_h

        x2 = (
            cx + w / 2
        ) * original_w

        y2 = (
            cy + h / 2
        ) * original_h

        # ----------------------------------------------------
        # Clamp coordinates
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                original_w - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                original_h - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                original_w - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                original_h - 1,
                y2
            )
        )

        # ----------------------------------------------------
        # Save detection
        # ----------------------------------------------------

        detections.append(
            (
                x1,
                y1,
                x2,
                y2,
                class_id,
                confidence
            )
        )

    return detections


# ============================================================
# CAMERA
# ============================================================

print("[INFO] Opening camera...")

cap = cv2.VideoCapture("C:/cctv-dev-temp/use_phone_5.MOV")
# cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Cannot open camera"
    )


# ============================================================
# FPS
# ============================================================

fps = 0.0

prev_time = time.perf_counter()


# ============================================================
# MAIN LOOP
# ============================================================

print("[INFO] Starting inference...")
print("[INFO] Press Q or ESC to exit.")
print()


try:

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            print(
                "[WARNING] Failed to read camera frame."
            )

            break

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        (
            dets,
            labels,
            latency,
            width,
            height
        ) = inference(
            frame
        )

        # ----------------------------------------------------
        # Postprocess
        # ----------------------------------------------------

        detections = postprocess(
            dets,
            labels,
            width,
            height
        )

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        for (
            x1,
            y1,
            x2,
            y2,
            class_id,
            confidence
        ) in detections:

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Bounding box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label

            label = (
                f"class {class_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        delta_time = (
            current_time -
            prev_time
        )

        prev_time = current_time

        if delta_time > 0:

            current_fps = (
                1.0 / delta_time
            )

            if fps > 0:

                fps = (
                    0.9 * fps +
                    0.1 * current_fps
                )

            else:

                fps = current_fps

        # ----------------------------------------------------
        # Info
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"TensorRT: {latency:.1f} ms",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {len(detections)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "RF-DETR Medium TensorRT FP16",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("q") or
            key == 27
        ):
            break


finally:

    # ========================================================
    # CAMERA CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    # ========================================================
    # CUDA CLEANUP
    # ========================================================

    print()
    print("[INFO] Cleaning up CUDA resources...")

    try:
        result = runtime.cudaFree(
            int(d_input)
        )
        check_cuda(
            result,
            "cudaFree input"
        )
    except Exception as e:
        print(
            f"[WARNING] cudaFree input: {e}"
        )

    try:
        result = runtime.cudaFree(
            int(d_dets)
        )
        check_cuda(
            result,
            "cudaFree dets"
        )
    except Exception as e:
        print(
            f"[WARNING] cudaFree dets: {e}"
        )

    try:
        result = runtime.cudaFree(
            int(d_labels)
        )
        check_cuda(
            result,
            "cudaFree labels"
        )
    except Exception as e:
        print(
            f"[WARNING] cudaFree labels: {e}"
        )

    try:
        result = runtime.cudaStreamDestroy(
            stream
        )
        check_cuda(
            result,
            "cudaStreamDestroy"
        )
    except Exception as e:
        print(
            f"[WARNING] cudaStreamDestroy: {e}"
        )

    print("[INFO] Done.")