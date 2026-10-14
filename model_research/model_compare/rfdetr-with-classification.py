"""RF-DETR human detection (TensorRT) + Keras phone classification.

Pipeline per frame:
    1. TensorRT RF-DETR (576x576) detects humans on the full frame.
    2. Each human crop is resized to 180x180 and classified by the
       Keras model (feature_extraction_with_augmentation.keras).
    3. Result is drawn: rectangle around human + "Phone"/"No Phone" label.

Usage:
    python rfdetr-with-classification.py
    Press Q or ESC to exit.
"""
import os
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import tensorrt as trt

from cuda.bindings import runtime

import keras
import keras_hub  # noqa: F401  (required to deserialize the .keras model)


# ============================================================
# CONFIG
# ============================================================

HUMAN_ENGINE_PATH = r"C:\cctv-insight\research_amir\model_research\model_compare\output\rfdetr-medium_fp16.trt"
KERAS_MODEL_PATH = r"C:\cctv-insight\research_amir\model_research\model_compare\feature_extraction_letterbox.keras"

HUMAN_INPUT_SIZE = 576
CONF_THRESHOLD_HUMAN = 0.5
CONF_THRESHOLD_PHONE = 0.5

HUMAN_CLASS_ID = 1

MIN_HUMAN_CROP_SIZE = 50
MAX_HUMANS_PER_FRAME = 5

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# CUDA ERROR CHECK
# ============================================================

def check_cuda(result, operation="CUDA operation"):
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"{operation} failed: {err}"
        )


# ============================================================
# LOAD TENSORRT HUMAN DETECTION ENGINE
# ============================================================

print("[INFO] Loading TensorRT human detection engine...")

with open(HUMAN_ENGINE_PATH, "rb") as f:
    engine_data = f.read()

trt_runtime = trt.Runtime(TRT_LOGGER)

engine = trt_runtime.deserialize_cuda_engine(engine_data)

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
# LOAD KERAS CLASSIFICATION MODEL
# ============================================================

print("[INFO] Loading Keras classification model...")

keras_model = keras.models.load_model(KERAS_MODEL_PATH)

print("[INFO] Keras model loaded.")


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
        (HUMAN_INPUT_SIZE, HUMAN_INPUT_SIZE),
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
    original_h,
    target_class_id=HUMAN_CLASS_ID,
    conf_threshold=CONF_THRESHOLD_HUMAN
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

        scores = labels[i]

        class_id = int(
            np.argmax(scores)
        )

        # ----------------------------------------------------
        # Class filtering
        # ----------------------------------------------------

        if class_id != target_class_id:
            continue

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        confidence = float(
            scores[class_id]
        )

        if confidence < conf_threshold:
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
# KERAS CLASSIFICATION
# ============================================================

def classify_crops(crops):
    """Classify a list of human crops, return list of (class, prob1)."""

    if not crops:
        return []

    batch = np.stack(crops, axis=0)

    probs = keras_model.predict(
        batch,
        verbose=0
    ).ravel()

    return [
        (1 if prob1 >= CONF_THRESHOLD_PHONE else 0, float(prob1))
        for prob1 in probs
    ]


# ============================================================
# CAMERA
# ============================================================

print("[INFO] Opening camera...")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:/cctv-dev-temp/use_phone_2.MOV")


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

print("[INFO] Starting detection + classification...")
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
        # Stage 1: TensorRT human detection
        # ----------------------------------------------------

        (
            dets,
            labels,
            latency_detect,
            width,
            height
        ) = inference(
            frame
        )

        humans = postprocess(
            dets,
            labels,
            width,
            height,
            target_class_id=HUMAN_CLASS_ID,
            conf_threshold=CONF_THRESHOLD_HUMAN
        )

        # Sort humans by confidence, take top N
        humans.sort(key=lambda x: x[5], reverse=True)
        humans = humans[:MAX_HUMANS_PER_FRAME]

        # ----------------------------------------------------
        # Stage 2: Keras classification on human crops
        # ----------------------------------------------------

        crops = []
        valid_humans = []

        for (
            hx1,
            hy1,
            hx2,
            hy2,
            h_class_id,
            h_conf
        ) in humans:

            hx1_i = int(hx1)
            hy1_i = int(hy1)
            hx2_i = int(hx2)
            hy2_i = int(hy2)

            crop_w = hx2_i - hx1_i
            crop_h = hy2_i - hy1_i

            if (
                crop_w < MIN_HUMAN_CROP_SIZE
                or crop_h < MIN_HUMAN_CROP_SIZE
            ):
                continue

            human_crop = frame[
                hy1_i:hy2_i,
                hx1_i:hx2_i
            ]

            if human_crop.size == 0:
                continue

            # Preprocess crop for Keras: 180x180 RGB float32 [0,255]
            # Matches predict_viewer.py load_image(): raw [0,255] float32,
            # NO /255 normalization - the model's EfficientNetImageConverter
            # layer does scale/offset normalization internally.
            crop_rgb = cv2.cvtColor(
                human_crop,
                cv2.COLOR_BGR2RGB
            )

            crop_resized = cv2.resize(
                crop_rgb,
                (180, 180),
                interpolation=cv2.INTER_LINEAR
            )

            crops.append(
                crop_resized.astype(np.float32)
            )

            valid_humans.append(
                (
                    hx1_i,
                    hy1_i,
                    hx2_i,
                    hy2_i,
                    h_conf
                )
            )

        # Batch classify all crops at once
        start_classify = time.perf_counter()

        classifications = classify_crops(crops)

        latency_classify = (
            time.perf_counter() - start_classify
        ) * 1000.0

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        phone_count = 0

        for (
            hx1_i,
            hy1_i,
            hx2_i,
            hy2_i,
            h_conf
        ), (cls, prob1) in zip(
            valid_humans,
            classifications
        ):

            if cls == 1:

                phone_count += 1

                color = (0, 0, 255)
                label = f"Human | Phone {prob1:.2f}"

            else:

                color = (0, 255, 0)
                label = f"Human | No Phone {1 - prob1:.2f}"

            cv2.rectangle(
                frame,
                (hx1_i, hy1_i),
                (hx2_i, hy2_i),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    hx1_i,
                    max(
                        hy1_i - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
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
            f"Detect: {latency_detect:.1f} ms",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Classify: {latency_classify:.1f} ms x{len(crops)}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Humans: {len(valid_humans)}  Phones: {phone_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "RF-DETR Human Detection + Keras Phone Classification",
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