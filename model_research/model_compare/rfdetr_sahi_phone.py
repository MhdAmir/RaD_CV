import time
import cv2
import numpy as np
import tensorrt as trt

from cuda.bindings import runtime

from sahi.annotation import Category
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image_as_pil


# ============================================================
# CONFIG
# ============================================================

ENGINE_PATH = r"C:\cctv-insight\research_amir\output\rfdetr-medium_fp16.trt"

INPUT_SIZE = 768
CONF_THRESHOLD = 0.3

SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# The 768 phone model has 3 output classes (shape 1,300,3).
# If you know which index is the actual phone class, set it here.
# None means accept all classes above CONF_THRESHOLD as phone.
PHONE_CLASS_INDEX = None

# Map every model class to COCO-like phone id 77 for display.
# If PHONE_CLASS_INDEX is None, all classes are treated as phone id 77.
CATEGORY_REMAPPING = {"0": 77, "1": 77, "2": 77}
CATEGORY_MAPPING = {"77": "phone"}

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
# CUSTOM SAHI DETECTION MODEL FOR TENSORRT RF-DETR
# ============================================================

class TensorRTRFDETRDetectionModel(DetectionModel):
    """
    SAHI detection model wrapper for a TensorRT RF-DETR engine.

    Expects the engine to expose tensors:
        input  : (1, 3, H, W)
        dets   : (1, 300, 4)   [cx, cy, w, h] normalized
        labels : (1, 300, num_classes)
    """

    def __init__(self, *args, **kwargs):
        self._original_predictions = None
        self._original_shape = None
        self._original_hw = None
        super().__init__(*args, **kwargs)

    def set_model(self, model, **kwargs):
        """Accept an already-loaded engine/context tuple."""
        if isinstance(model, tuple):
            self.model, self.context = model
        else:
            self.model = model

    def load_model(self):
        """Load TensorRT engine and allocate CUDA resources."""
        print(f"[INFO] Loading TensorRT engine for SAHI: {self.model_path}")

        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.model = trt_runtime.deserialize_cuda_engine(engine_data)

        if self.model is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.model.create_execution_context()

        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        input_name = "input"
        dets_name = "dets"
        labels_name = "labels"

        self.input_shape = tuple(self.model.get_tensor_shape(input_name))
        self.dets_shape = tuple(self.model.get_tensor_shape(dets_name))
        self.labels_shape = tuple(self.model.get_tensor_shape(labels_name))

        self.input_dtype = np.dtype(trt.nptype(self.model.get_tensor_dtype(input_name)))
        self.dets_dtype = np.dtype(trt.nptype(self.model.get_tensor_dtype(dets_name)))
        self.labels_dtype = np.dtype(trt.nptype(self.model.get_tensor_dtype(labels_name)))

        self.input_nbytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self.dets_nbytes = int(np.prod(self.dets_shape)) * self.dets_dtype.itemsize
        self.labels_nbytes = int(np.prod(self.labels_shape)) * self.labels_dtype.itemsize

        self.h_input = np.empty(self.input_shape, dtype=self.input_dtype)
        self.h_dets = np.empty(self.dets_shape, dtype=self.dets_dtype)
        self.h_labels = np.empty(self.labels_shape, dtype=self.labels_dtype)

        stream_result = runtime.cudaStreamCreate()
        check_cuda(stream_result, "cudaStreamCreate")
        self.stream = stream_result[1]

        self.d_input = runtime.cudaMalloc(self.input_nbytes)[1]
        self.d_dets = runtime.cudaMalloc(self.dets_nbytes)[1]
        self.d_labels = runtime.cudaMalloc(self.labels_nbytes)[1]

        self.context.set_tensor_address(input_name, int(self.d_input))
        self.context.set_tensor_address(dets_name, int(self.d_dets))
        self.context.set_tensor_address(labels_name, int(self.d_labels))

        print("[INFO] TensorRT engine loaded and buffers allocated.")

    def unload_model(self):
        """Free CUDA resources."""
        try:
            runtime.cudaFree(int(self.d_input))
        except Exception as e:
            print(f"[WARNING] cudaFree input: {e}")
        try:
            runtime.cudaFree(int(self.d_dets))
        except Exception as e:
            print(f"[WARNING] cudaFree dets: {e}")
        try:
            runtime.cudaFree(int(self.d_labels))
        except Exception as e:
            print(f"[WARNING] cudaFree labels: {e}")
        try:
            runtime.cudaStreamDestroy(self.stream)
        except Exception as e:
            print(f"[WARNING] cudaStreamDestroy: {e}")

        self.model = None
        self.context = None

    def _preprocess(self, image):
        """Preprocess a BGR numpy image for the engine."""
        original_h, original_w = image.shape[:2]

        resized = cv2.resize(
            image,
            (self.input_shape[3], self.input_shape[2]),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = np.expand_dims(rgb, axis=0)
        rgb = np.ascontiguousarray(rgb, dtype=self.input_dtype)

        return rgb, original_w, original_h

    def perform_inference(self, image):
        """Run TensorRT inference on a single image/slice."""
        image_batch, original_w, original_h = self._preprocess(image)

        np.copyto(self.h_input, image_batch)

        check_cuda(
            runtime.cudaMemcpyAsync(
                int(self.d_input),
                self.h_input.ctypes.data,
                self.input_nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync H2D",
        )

        success = self.context.execute_async_v3(
            stream_handle=int(self.stream)
        )

        if not success:
            raise RuntimeError("TensorRT execute_async_v3() failed")

        check_cuda(
            runtime.cudaMemcpyAsync(
                self.h_dets.ctypes.data,
                int(self.d_dets),
                self.dets_nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync D2H dets",
        )

        check_cuda(
            runtime.cudaMemcpyAsync(
                self.h_labels.ctypes.data,
                int(self.d_labels),
                self.labels_nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync D2H labels",
        )

        check_cuda(
            runtime.cudaStreamSynchronize(self.stream),
            "cudaStreamSynchronize",
        )

        self._original_predictions = (
            self.h_dets.copy(),
            self.h_labels.copy(),
            original_w,
            original_h,
        )

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list,
        full_shape_list,
    ):
        """Convert TensorRT raw outputs to SAHI ObjectPrediction list."""

        dets, labels, original_w, original_h = self._original_predictions

        dets = dets[0]
        labels = labels[0]

        shift_x, shift_y = shift_amount_list[0]
        full_shape = full_shape_list[0] if full_shape_list else None

        if full_shape is not None:
            full_h, full_w = full_shape
        else:
            full_h, full_w = int(original_h), int(original_w)

        predictions = []

        num_detections = min(dets.shape[0], labels.shape[0])

        for i in range(num_detections):
            scores = labels[i]
            class_id = int(np.argmax(scores))

            # Filter by specific phone class index if configured
            if (
                PHONE_CLASS_INDEX is not None
                and class_id != PHONE_CLASS_INDEX
            ):
                continue

            confidence = float(scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            cx, cy, w, h = dets[i]

            x1 = (float(cx) - float(w) / 2) * original_w + shift_x
            y1 = (float(cy) - float(h) / 2) * original_h + shift_y
            x2 = (float(cx) + float(w) / 2) * original_w + shift_x
            y2 = (float(cy) + float(h) / 2) * original_h + shift_y

            x1 = max(0.0, min(full_w - 1, x1))
            y1 = max(0.0, min(full_h - 1, y1))
            x2 = max(0.0, min(full_w - 1, x2))
            y2 = max(0.0, min(full_h - 1, y2))

            category_name = self.category_mapping.get(
                str(class_id),
                f"class_{class_id}",
            )

            predictions.append(
                ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    score=confidence,
                    category_id=class_id,
                    category_name=category_name,
                    shift_amount=[shift_x, shift_y],
                    full_shape=[full_h, full_w],
                )
            )

        self._object_prediction_list_per_image = [predictions]


if __name__ == "__main__":

    # ============================================================
    # INITIALIZE SAHI MODEL
    # ============================================================

    print("[INFO] Initializing SAHI TensorRT detection model...")

    phone_model = TensorRTRFDETRDetectionModel(
        model_path=ENGINE_PATH,
        confidence_threshold=CONF_THRESHOLD,
        category_mapping=CATEGORY_MAPPING,
        category_remapping=CATEGORY_REMAPPING,
        device="cuda:0",
    )


    # ============================================================
    # CAMERA
    # ============================================================

    print("[INFO] Opening camera...")

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("C:/cctv-dev-temp/use_phone_3.MOV")


    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")


    # ============================================================
    # FPS
    # ============================================================

    fps = 0.0
    prev_time = time.perf_counter()


    # ============================================================
    # MAIN LOOP
    # ============================================================

    print("[INFO] Starting SAHI sliced inference for phone detection...")
    print("[INFO] Press Q or ESC to exit.")
    print()


    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Failed to read camera frame.")
                break

            start_total = time.perf_counter()

            # SAHI sliced prediction
            result = get_sliced_prediction(
                image=frame,
                detection_model=phone_model,
                slice_height=SLICE_HEIGHT,
                slice_width=SLICE_WIDTH,
                overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
                overlap_width_ratio=OVERLAP_WIDTH_RATIO,
                perform_standard_pred=False,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.5,
                verbose=0,
            )

            total_latency = (time.perf_counter() - start_total) * 1000.0

            # Draw predictions
            for obj in result.object_prediction_list:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox.to_xyxy()]
                score = obj.score.value
                label = f"{obj.category.name} {score:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # FPS
            current_time = time.perf_counter()
            delta_time = current_time - prev_time
            prev_time = current_time

            if delta_time > 0:
                current_fps = 1.0 / delta_time
                if fps > 0:
                    fps = 0.9 * fps + 0.1 * current_fps
                else:
                    fps = current_fps

            # Info overlay
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"SAHI total: {total_latency:.1f} ms",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Phones: {len(result.object_prediction_list)}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "RF-DETR SAHI Phone Detection (TensorRT)",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break


    finally:

        cap.release()
        cv2.destroyAllWindows()

        print()
        print("[INFO] Cleaning up...")

        phone_model.unload_model()

        print("[INFO] Done.")
