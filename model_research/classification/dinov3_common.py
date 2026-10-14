import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torchvision.transforms as T

HUB_REPO = "facebookresearch/dinov3"
HUB_ENTRY = "dinov3_vits16"  # Small variant: ~22M params, embed_dim=384

CHECKPOINT_FILENAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / CHECKPOINT_FILENAME

META_DOWNLOAD_PAGE = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BICUBIC = T.InterpolationMode.BICUBIC


def _download_weights(url: str) -> Path:
    name = Path(urlparse(url).path).name or CHECKPOINT_FILENAME
    dest = WEIGHTS_DIR / name
    if dest.exists():
        return dest
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 dinov3-local"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)
    return dest


def _missing_weights_message() -> str:
    return (
        "DINOv3 weights not found.\n"
        "\n"
        "DINOv3 checkpoints are license-gated by Meta; the public URL returns\n"
        "HTTP 403 until access is granted. To obtain the weights:\n"
        f"  1. Request access: {META_DOWNLOAD_PAGE}\n"
        "  2. Meta emails signed URLs (download with wget/curl, not a browser)\n"
        "  3. Save the ViT-S/16 LVD-1689M checkpoint as:\n"
        f"       {DEFAULT_WEIGHTS_PATH}\n"
        "     or set the DINOV3_WEIGHTS environment variable to the local file\n"
        "     path (or to the signed URL, which is downloaded automatically)."
    )


def resolve_weights() -> Path:
    env = os.environ.get("DINOV3_WEIGHTS", "").strip()
    if env:
        if env.lower().startswith(("http://", "https://")):
            return _download_weights(env)
        path = Path(env).expanduser()
        if not path.exists():
            raise SystemExit(f"DINOV3_WEIGHTS points to a missing file: {env}")
        return path.resolve()
    if DEFAULT_WEIGHTS_PATH.exists():
        return DEFAULT_WEIGHTS_PATH
    raise SystemExit(_missing_weights_message())


def load_backbone(weights: Path | str | None = None) -> torch.nn.Module:
    weights = resolve_weights() if weights is None else Path(weights)
    backbone = torch.hub.load(HUB_REPO, HUB_ENTRY, pretrained=True, weights=str(weights))
    return backbone.eval()


def make_eval_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), interpolation=BICUBIC, antialias=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_train_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Resize((256, 256), interpolation=BICUBIC, antialias=True),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def save_classifier(path: Path, head: nn.Linear, classes: list[str]) -> None:
    torch.save({"classes": classes, "state_dict": head.state_dict()}, path)


def load_classifier(path: Path, device: str) -> tuple[nn.Linear, list[str] | None]:
    if not path.exists():
        raise SystemExit(
            f"Classifier head not found: {path}\n"
            "Train it first, e.g.:  python train_dino.py --data_dir <dataset>"
        )
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        classes = obj.get("classes")
        state = obj["state_dict"]
    else:
        classes = None
        state = obj
    num_classes, in_features = state["weight"].shape
    head = nn.Linear(in_features, num_classes)
    head.load_state_dict(state)
    return head, classes