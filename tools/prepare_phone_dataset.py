import pandas as pd
import shutil
from pathlib import Path

dataset_root = Path(
    r"C:\Users\istd.magang17\Downloads\reorganized_phone_dataset_yolo\reorganized_dataset"
)

csv_path = Path(
    r"C:\Users\istd.magang17\Downloads\label.csv"
)

df = pd.read_csv(csv_path)

label_map = {
    0: "not_use_handphone",
    1: "use_handphone"
}

label_dict = {
    row["image_name"]: label_map[int(row["class_id"])]
    for _, row in df.iterrows()
}

for split in ["train", "val", "test"]:
    split_dir = dataset_root / split

    for img_path in split_dir.glob("*.*"):
        if img_path.name not in label_dict:
            continue

        class_name = label_dict[img_path.name]

        target_dir = split_dir / class_name
        target_dir.mkdir(exist_ok=True)

        shutil.move(
            str(img_path),
            str(target_dir / img_path.name)
        )

print("Done.")