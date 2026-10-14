import argparse
import time

import cv2

FACE_CASCADE = "haarcascade_frontalface_default.xml"
MOUTH_CASCADE = "haarcascade_smile.xml"

COLOR_MASK = (0, 200, 0)
COLOR_NO_MASK = (0, 0, 220)
COLOR_MOUTH = (0, 140, 255)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time mask detection using Haar cascades: "
        "no mouth inside face -> mask, mouth visible -> no mask."
    )
    parser.add_argument("--device", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--scale-face", type=float, default=1.1, help="Face scaleFactor")
    parser.add_argument(
        "--neighbors-face", type=int, default=5, help="Face minNeighbors"
    )
    parser.add_argument(
        "--neighbors-mouth", type=int, default=15, help="Mouth minNeighbors (higher = stricter)"
    )
    parser.add_argument(
        "--mouth-ratio", type=float, default=0.5,
        help="Bottom fraction of face ROI searched for mouth (0.5 = lower half)",
    )
    parser.add_argument(
        "--show-mouth", action="store_true",
        help="Also draw the detected mouth/lips box",
    )
    parser.add_argument(
        "--no-flip", action="store_true", help="Disable horizontal mirror flip"
    )
    return parser.parse_args()


def open_camera(device, width, height):
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {device}. Try --device 1, 2, ... or check camera privacy settings."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "cv2.CascadeClassifier not found (OpenCV 5.x removed it). "
            "Run with Python that has OpenCV 4.x, e.g. "
            r"C:\Users\istd.magang17\AppData\Local\Python\pythoncore-3.14-64\python.exe"
        )

    args = parse_args()

    cascade_dir = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(cascade_dir + FACE_CASCADE)
    mouth_cascade = cv2.CascadeClassifier(cascade_dir + MOUTH_CASCADE)
    if face_cascade.empty() or mouth_cascade.empty():
        raise RuntimeError("Failed to load Haar cascades from " + cascade_dir)

    cap = open_camera(args.device, args.width, args.height)

    window = "Mask Detection (Haar) - press q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame, exiting.")
                break

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_face,
                minNeighbors=args.neighbors_face,
                minSize=(60, 60),
            )

            n_mask = 0
            n_no_mask = 0

            for (x, y, w, h) in faces:
                mouth_zone_top = int(y + h * (1.0 - args.mouth_ratio))
                mouth_roi = gray[mouth_zone_top : y + h, x : x + w]

                mouths = (
                    mouth_cascade.detectMultiScale(
                        mouth_roi,
                        scaleFactor=1.1,
                        minNeighbors=args.neighbors_mouth,
                        minSize=(int(w * 0.15), int(w * 0.10)),
                    )
                    if mouth_roi.size
                    else []
                )

                if len(mouths) > 0:
                    label = "No Mask"
                    color = COLOR_NO_MASK
                    n_no_mask += 1
                else:
                    label = "Mask"
                    color = COLOR_MASK
                    n_mask += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label_size, base = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                top = max(y - label_size[1] - 8, 0)
                cv2.rectangle(
                    frame,
                    (x, top),
                    (x + label_size[0] + 6, top + label_size[1] + base + 6),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x + 3, top + label_size[1] + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if args.show_mouth and len(mouths) > 0:
                    for (mx, my, mw, mh) in mouths:
                        cv2.rectangle(
                            frame,
                            (x + mx, mouth_zone_top + my),
                            (x + mx + mw, mouth_zone_top + my + mh),
                            COLOR_MOUTH,
                            2,
                        )

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            info = f"FPS: {fps:4.1f} | Mask: {n_mask} | No Mask: {n_no_mask}"
            cv2.putText(
                frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (30, 200, 30), 2,
            )
            cv2.putText(
                frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1,
            )

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Done.")


if __name__ == "__main__":
    main()