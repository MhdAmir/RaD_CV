import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

model = YOLO("yolo26n-depth.pt")

click_point = None


def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera 0")

win_overlay = "YOLO Depth - click to measure"
cv2.namedWindow(win_overlay)
cv2.setMouseCallback(win_overlay, on_mouse)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, verbose=False)[0]

    depth = result.depth.data.cpu().numpy()  # (H, W) float32, meters

    # Colorize with near = warm
    colored = colorize_depth(depth, cmap="spectral")

    # Fixed range 0-20 m so the same color means the same distance across frames
    metric = colorize_depth(depth, vmin=0.0, vmax=20.0, cmap="inferno", mode="metric")

    # Blended overlay (uses cmap="jet", mode="disparity")
    overlay = result.plot()

    # Resize depth and visualizations to frame size so click coords match
    h, w = frame.shape[:2]
    depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.resize(colored, (w, h))
    metric = cv2.resize(metric, (w, h))

    if click_point is not None:
        x, y = click_point
        dist_m = float(depth_full[y, x])
        cv2.drawMarker(overlay, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(overlay, (x, y), 6, (0, 255, 255), 2)
        label = f"{dist_m:.2f} m"
        cv2.putText(overlay, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    cv2.imshow(win_overlay, overlay)
    cv2.imshow("YOLO Depth - spectral", colored)
    cv2.imshow("YOLO Depth - metric 0-20m", metric)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()