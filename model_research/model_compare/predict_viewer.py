"""Prediction viewer: shows an image and its predicted class side by side.

Usage (from the classification folder):

    venv\Scripts\python.exe predict_viewer.py                 # browse images/
    venv\Scripts\python.exe predict_viewer.py --dir images    # explicit folder
    venv\Scripts\python.exe predict_viewer.py --model fine_tuning.keras

Controls:
    Right / Space = next image      Left = previous image
    0 / 1         = save corrected label to label.csv
    R             = re-run prediction on current image
    Esc           = quit
"""
import argparse
import csv
import os
import sys
import tkinter as tk
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import patch_tf_check  # noqa: F401  (must come before keras_hub imports)

import keras
import numpy as np
from PIL import Image, ImageTk

DEFAULT_MODEL = "feature_extraction_with_augmentation.keras"
IMAGE_SIZE = (180, 180)
LABEL_CSV = BASE_DIR / "label.csv"
ALLOWED = (".jpg", ".jpeg", ".png", ".bmp")


def load_labels():
    labels = {}
    if LABEL_CSV.exists():
        with LABEL_CSV.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row.get("image_name", "").strip()
                try:
                    labels[name] = int(float(row.get("class_id", "")))
                except (TypeError, ValueError):
                    continue
    return labels


def save_labels(labels):
    with LABEL_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_name", "class_id"])
        for name, cls in labels.items():
            writer.writerow([name, cls])


def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
            img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        return np.asarray(img, dtype="float32")


class PredictViewer:
    def __init__(self, root, files, model, labels):
        self.root = root
        self.files = files
        self.model = model
        self.labels = labels
        self.idx = 0
        self.photo = None
        self.setup_ui()
        self.bind_keys()
        self.show_image()

    def setup_ui(self):
        self.root.title("Prediction Viewer")
        self.root.geometry("1000x800")
        self.root.configure(bg="#1e1e1e")

        self.info_label = tk.Label(
            self.root, font=("Consolas", 12), fg="#dddddd", bg="#1e1e1e",
            anchor="w", justify="left")
        self.info_label.pack(fill="x", padx=8, pady=(6, 2))

        self.pred_label = tk.Label(
            self.root, font=("Consolas", 16, "bold"), fg="#66cc66",
            bg="#1e1e1e", anchor="w")
        self.pred_label.pack(fill="x", padx=8, pady=(0, 2))

        self.canvas = tk.Canvas(self.root, bg="#141414", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self.canvas.bind("<Configure>",
                         lambda e: self.show_image(keep_idx=True))

        self.help_label = tk.Label(
            self.root,
            text="Right/Space = next  |  Left = back  |  "
                 "0/1 = save corrected label  |  Esc = quit",
            font=("Consolas", 11), fg="#888888", bg="#1e1e1e")
        self.help_label.pack(side="left", padx=8, pady=(2, 8))

        self.msg_label = tk.Label(
            self.root, text="", font=("Consolas", 11), fg="#66cc66",
            bg="#1e1e1e")
        self.msg_label.pack(side="right", padx=8, pady=(2, 8))

    def bind_keys(self):
        self.root.bind("<Right>", lambda e: self.move(1))
        self.root.bind("<space>", lambda e: self.move(1))
        self.root.bind("<Left>", lambda e: self.move(-1))
        self.root.bind("0", lambda e: self.correct(0))
        self.root.bind("1", lambda e: self.correct(1))
        self.root.bind("r", lambda e: self.show_image())
        self.root.bind("R", lambda e: self.show_image())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def predict_current(self, name):
        batch = np.expand_dims(load_image(self.dir / name), axis=0)
        prob1 = float(self.model.predict(batch, verbose=0).ravel()[0])
        cls = 1 if prob1 >= 0.5 else 0
        conf = prob1 if cls == 1 else 1 - prob1
        return cls, conf, prob1

    def show_image(self, keep_idx=False):
        if not self.files:
            self.info_label.config(text="No images found")
            return
        self.idx = max(0, min(self.idx, len(self.files) - 1))
        path = self.files[self.idx]
        name = path.name

        try:
            cls, conf, prob1 = self.predict_current(name)
        except Exception as exc:
            self.pred_label.config(text=f"prediction failed: {exc}",
                                   fg="#ff5555")
            return

        pred_txt = f"MODEL: class {cls}   confidence {conf:.2%}   P(class1)={prob1:.3f}"
        truth = self.labels.get(name)
        if truth is not None:
            agree = "OK" if truth == cls else "MISMATCH"
            pred_txt += f"   |   LABEL.CSV: {truth}  [{agree}]"

        color = "#66cc66" if truth is None or truth == cls else "#ffaa55"
        self.pred_label.config(text=pred_txt, fg=color)

        info = (f"{self.idx + 1}/{len(self.files)}  |  {name}  |  "
                f"Right/Left to browse")
        self.info_label.config(text=info)

        with Image.open(path) as img:
            cw = max(self.canvas.winfo_width(), 200)
            ch = max(self.canvas.winfo_height(), 200)
            img = img.copy()
            img.thumbnail((cw, ch))
            self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        x = self.canvas.winfo_width() // 2
        y = self.canvas.winfo_height() // 2
        self.canvas.create_image(x, y, image=self.photo, anchor="center")

    def move(self, step):
        if not self.files:
            return
        self.idx = max(0, min(self.idx + step, len(self.files) - 1))
        self.show_image()

    def correct(self, cls):
        if not self.files:
            return
        name = self.files[self.idx].name
        self.labels[name] = cls
        save_labels(self.labels)
        self.msg_label.config(text=f"saved: {name} = {cls}", fg="#66cc66")
        self.show_image(keep_idx=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="images",
                        help="folder of images to browse (default: images)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="saved .keras model to use")
    args = parser.parse_args()

    directory = BASE_DIR / args.dir
    files = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in ALLOWED)
    if not files:
        raise SystemExit(f"no images found in {directory}")

    model_path = BASE_DIR / args.model
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    print(f"Loading model {args.model} ...")
    model = keras.models.load_model(str(model_path))

    labels = load_labels()
    print(f"{len(files)} images, {len(labels)} existing labels")

    root = tk.Tk()
    viewer = PredictViewer(root, files, model, labels)
    viewer.dir = directory
    root.mainloop()


if __name__ == "__main__":
    main()