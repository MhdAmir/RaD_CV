import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import dinov3_common as common

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSIFIER_PATH = Path(__file__).resolve().parent / "classifier.pth"


def evaluate(backbone, head, loader) -> float:
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            pred = head(backbone(x)).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="DINOv3-Small linear probe")
    ap.add_argument("--data_dir", required=True,
                    help="ImageFolder root containing train/ and val/ subdirs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(
            f"Expected {data_dir}\\train and {data_dir}\\val subdirectories,\n"
            f"each containing one folder per class (e.g. employee/, visitor/, security/)."
        )

    backbone = common.load_backbone().to(DEVICE)
    backbone.requires_grad_(False)

    train_ds = ImageFolder(train_dir, transform=common.make_train_transform())
    val_ds = ImageFolder(val_dir, transform=common.make_eval_transform())
    classes = train_ds.classes  # alphabetical, saved with the head
    print(f"Classes ({len(classes)}): {classes}")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    head = nn.Linear(backbone.embed_dim, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        running = n = 0
        for x, y in train_dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                feat = backbone(x)
            opt.zero_grad()
            loss = loss_fn(head(feat), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()

        val_acc = evaluate(backbone, head, val_dl)
        print(f"epoch {epoch:3d}  train_loss {running / n:.4f}  val_acc {val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            common.save_classifier(CLASSIFIER_PATH, head, classes)
            print(f"  saved -> {CLASSIFIER_PATH}")

    print(f"Done. Best val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()