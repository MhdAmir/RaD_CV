import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import dinov3_common as common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load DINOv3-Small backbone
# --------------------------
backbone = common.load_backbone().to(DEVICE)

# --------------------------
# Classification Head
# --------------------------
classifier, classes = common.load_classifier(
    Path(__file__).resolve().parent / "classifier.pth", DEVICE
)

if classes is None:
    print("[warn] classifier.pth has no class names; using hardcoded order")
    classes = ["employee", "visitor", "security"]

assert classifier.in_features == backbone.embed_dim, (
    f"classifier expects {classifier.in_features}-dim features "
    f"but {common.HUB_ENTRY} outputs {backbone.embed_dim}-dim"
)

classifier = classifier.to(DEVICE).eval()

# --------------------------
# Preprocess (official DINOv3 eval transform: bicubic resize + ImageNet norm)
# --------------------------
transform = common.make_eval_transform()

# --------------------------
# Camera
# --------------------------
cap = cv2.VideoCapture(0)

prev = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(rgb)

    x = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        # DINOv3-Small CLS-token feature
        feat = backbone(x)

        logits = classifier(feat)
        probs = F.softmax(logits, dim=1)

        conf, pred = probs[0].max(0)
        pred, conf = pred.item(), conf.item()

    fps = 1 / (time.time() - prev)
    prev = time.time()

    cv2.putText(
        frame,
        f"{classes[pred]} {conf*100:.0f}%  FPS:{fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("DINOv3 Classification", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()