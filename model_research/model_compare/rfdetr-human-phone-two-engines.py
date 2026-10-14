import cv2
import time
import numpy as np
import tensorrt as trt

from cuda.bindings import runtime


# ============================================================
# CONFIG
# ============================================================

PHONE_ENGINE_PATH = r"C:\cctv-insight\research_amir\out\rfdetr-base_fp16.trt"
HUMAN_ENGINE_PATH = r"C:\cctv-insight\research_amir\model_research\model_compare\output\rfdetr-medium_fp16.trt"

PHONE_INPUT_SIZE = 560
HUMAN_INPUT_SIZE = 576

CONF_THRESHOLD_HUMAN = 0.5
CONF_THRESHOLD_PHONE = 0.3

HUMAN_CLASS_ID = 1
PHONE_CLASS_ID = 0

MIN_HUMAN_CROP_SIZE = 50
MAX_HUMANS_PER_FRAME = 5

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# CUDA ERROR CHECK
# ============================================================

def check_cuda(result, operation="CUDA operation"):
    """
    Check CUDA Runtime API return value.
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
# ENGINE RESOURCE CONTAINER
# ============================================================

def allocate_engine_resources(engine_path, name):
    print(f"[INFO] Loading engine: {name}")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = trt_runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError(f"Failed to load engine: {name}")

    context = engine.create_execution_context()

    if context is None:
        raise RuntimeError(f"Failed to create context: {name}")

    input_name = "input"
    dets_name = "dets"
    labels_name = "labels"

    input_shape = tuple(engine.get_tensor_shape(input_name))
    dets_shape = tuple(engine.get_tensor_shape(dets_name))
    labels_shape = tuple(engine.get_tensor_shape(labels_name))

    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    dets_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(dets_name)))
    labels_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(labels_name)))

    input_nbytes = int(np.prod(input_shape)) * input_dtype.itemsize
    dets_nbytes = int(np.prod(dets_shape)) * dets_dtype.itemsize
    labels_nbytes = int(np.prod(labels_shape)) * labels_dtype.itemsize

    print(f"[INFO] {name} tensor information")
    print(f"  Input  : {input_shape} {input_dtype}")
    print(f"  Dets   : {dets_shape} {dets_dtype}")
    print(f"  Labels : {labels_shape} {labels_dtype}")

    h_input = np.empty(input_shape, dtype=input_dtype)
    h_dets = np.empty(dets_shape, dtype=dets_dtype)
    h_labels = np.empty(labels_shape, dtype=labels_dtype)

    stream_result = runtime.cudaStreamCreate()
    check_cuda(stream_result, f"cudaStreamCreate {name}")
    stream = stream_result[1]

    d_input = runtime.cudaMalloc(input_nbytes)[1]
    d_dets = runtime.cudaMalloc(dets_nbytes)[1]
    d_labels = runtime.cudaMalloc(labels_nbytes)[1]

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(dets_name, int(d_dets))
    context.set_tensor_address(labels_name, int(d_labels))

    print(f"[INFO] {name} resources allocated\n")

    return {
        "name": name,
        "engine": engine,
        "context": context,
        "stream": stream,
        "input_name": input_name,
        "dets_name": dets_name,
        "labels_name": labels_name,
        "input_shape": input_shape,
        "dets_shape": dets_shape,
        "labels_shape": labels_shape,
        "input_dtype": input_dtype,
        "dets_dtype": dets_dtype,
        "labels_dtype": labels_dtype,
        "input_nbytes": input_nbytes,
        "dets_nbytes": dets_nbytes,
        "labels_nbytes": labels_nbytes,
        "h_input": h_input,
        "h_dets": h_dets,
        "h_labels": h_labels,
        "d_input": d_input,
        "d_dets": d_dets,
        "d_labels": d_labels,
    }


def free_engine_resources(res):
    print(f"[INFO] Freeing resources: {res['name']}")

    try:
        runtime.cudaFree(int(res["d_input"]))
    except Exception as e:
        print(f"[WARNING] cudaFree input {res['name']}: {e}")

    try:
        runtime.cudaFree(int(res["d_dets"]))
    except Exception as e:
        print(f"[WARNING] cudaFree dets {res['name']}: {e}")

    try:
        runtime.cudaFree(int(res["d_labels"]))
    except Exception as e:
        print(f"[WARNING] cudaFree labels {res['name']}: {e}")

    try:
        runtime.cudaStreamDestroy(res["stream"])
    except Exception as e:
        print(f"[WARNING] cudaStreamDestroy {res['name']}: {e}")


# ============================================================
# LOAD BOTH ENGINES
# ============================================================

phone_res = allocate_engine_resources(PHONE_ENGINE_PATH, "phone")
human_res = allocate_engine_resources(HUMAN_ENGINE_PATH, "human")


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(frame, input_size, input_dtype):

    original_h, original_w = frame.shape[:2]

    image = cv2.resize(
        frame,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image, dtype=input_dtype)

    return image, original_w, original_h


# ============================================================
# INFERENCE
# ============================================================

def inference_with_engine(res, frame, input_size):

    image, original_w, original_h = preprocess(
        frame,
        input_size,
        res["input_dtype"]
    )

    np.copyto(res["h_input"], image)

    check_cuda(
        runtime.cudaMemcpyAsync(
            int(res["d_input"]),
            res["h_input"].ctypes.data,
            res["input_nbytes"],
            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
            res["stream"]
        ),
        f"cudaMemcpyAsync H2D {res['name']}"
    )

    start = time.perf_counter()

    success = res["context"].execute_async_v3(
        stream_handle=int(res["stream"])
    )

    if not success:
        raise RuntimeError(
            f"TensorRT execute_async_v3() failed: {res['name']}"
        )

    check_cuda(
        runtime.cudaMemcpyAsync(
            res["h_dets"].ctypes.data,
            int(res["d_dets"]),
            res["dets_nbytes"],
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            res["stream"]
        ),
        f"cudaMemcpyAsync D2H dets {res['name']}"
    )

    check_cuda(
        runtime.cudaMemcpyAsync(
            res["h_labels"].ctypes.data,
            int(res["d_labels"]),
            res["labels_nbytes"],
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            res["stream"]
        ),
        f"cudaMemcpyAsync D2H labels {res['name']}"
    )

    check_cuda(
        runtime.cudaStreamSynchronize(res["stream"]),
        f"cudaStreamSynchronize {res['name']}"
    )

    latency = (time.perf_counter() - start) * 1000.0

    return (
        res["h_dets"].copy(),
        res["h_labels"].copy(),
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
    target_class_id=None,
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

        scores = labels[i]
        class_id = int(np.argmax(scores))

        if (
            target_class_id is not None
            and class_id != target_class_id
        ):
            continue

        confidence = float(scores[class_id])

        if confidence < conf_threshold:
            continue

        box = dets[i]
        cx, cy, w, h = box

        x1 = (float(cx) - float(w) / 2) * original_w
        y1 = (float(cy) - float(h) / 2) * original_h
        x2 = (float(cx) + float(w) / 2) * original_w
        y2 = (float(cy) + float(h) / 2) * original_h

        x1 = max(0, min(original_w - 1, x1))
        y1 = max(0, min(original_h - 1, y1))
        x2 = max(0, min(original_w - 1, x2))
        y2 = max(0, min(original_h - 1, y2))

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

# cap = cv2.VideoCapture("C:/cctv-dev-temp/use_phone_5.MOV")
cap = cv2.VideoCapture("C:/cctv-dev-temp/use_phone_3.MOV")

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

print("[INFO] Starting two-engine inference (Human model + Phone model)...")
print("[INFO] Press Q or ESC to exit.")
print()


try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Failed to read camera frame.")
            break

        # ----------------------------------------------------
        # Stage 1: Human model on full frame (576x576)
        # ----------------------------------------------------

        (
            dets,
            labels,
            latency_human,
            width,
            height
        ) = inference_with_engine(
            human_res,
            frame,
            HUMAN_INPUT_SIZE
        )

        humans = postprocess(
            dets,
            labels,
            width,
            height,
            target_class_id=HUMAN_CLASS_ID,
            conf_threshold=CONF_THRESHOLD_HUMAN
        )

        humans.sort(key=lambda x: x[5], reverse=True)
        humans = humans[:MAX_HUMANS_PER_FRAME]

        all_phone_detections = []
        total_latency_phone = 0.0
        crops_processed = 0

        # ----------------------------------------------------
        # Stage 2: Phone model on each human crop (768x768)
        # ----------------------------------------------------

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

            (
                crop_dets,
                crop_labels,
                crop_latency,
                _,
                _
            ) = inference_with_engine(
                phone_res,
                human_crop,
                PHONE_INPUT_SIZE
            )

            total_latency_phone += crop_latency
            crops_processed += 1

            phones = postprocess(
                crop_dets,
                crop_labels,
                PHONE_INPUT_SIZE,
                PHONE_INPUT_SIZE,
                target_class_id=PHONE_CLASS_ID,
                conf_threshold=CONF_THRESHOLD_PHONE
            )

            for (
                px1,
                py1,
                px2,
                py2,
                p_class_id,
                p_conf
            ) in phones:

                px1_orig = hx1 + (px1 / PHONE_INPUT_SIZE) * crop_w
                py1_orig = hy1 + (py1 / PHONE_INPUT_SIZE) * crop_h
                px2_orig = hx1 + (px2 / PHONE_INPUT_SIZE) * crop_w
                py2_orig = hy1 + (py2 / PHONE_INPUT_SIZE) * crop_h

                all_phone_detections.append(
                    (
                        px1_orig,
                        py1_orig,
                        px2_orig,
                        py2_orig,
                        p_class_id,
                        p_conf
                    )
                )

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        # Humans: green boxes
        for (x1, y1, x2, y2, _, confidence) in humans:
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)

            cv2.rectangle(
                frame,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Human {confidence:.2f}",
                (x1_i, max(y1_i - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.rectangle(
                frame,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 255),
                1
            )

        # Phones: red boxes
        for (x1, y1, x2, y2, _, confidence) in all_phone_detections:
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)

            cv2.rectangle(
                frame,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Phone {confidence:.2f}",
                (x1_i, max(y1_i - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()
        delta_time = current_time - prev_time
        prev_time = current_time

        if delta_time > 0:
            current_fps = 1.0 / delta_time
            if fps > 0:
                fps = 0.9 * fps + 0.1 * current_fps
            else:
                fps = current_fps

        # ----------------------------------------------------
        # Info overlay
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
            f"Human model: {latency_human:.1f} ms",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if crops_processed > 0:
            avg_phone = total_latency_phone / crops_processed
            cv2.putText(
                frame,
                f"Phone model: {avg_phone:.1f} ms x{crops_processed}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"Humans: {len(humans)}  Phones: {len(all_phone_detections)}",
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
            "RF-DETR Two-Engine: Human(576) -> Phone(768)",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
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

    free_engine_resources(phone_res)
    free_engine_resources(human_res)

    print("[INFO] Done.")
