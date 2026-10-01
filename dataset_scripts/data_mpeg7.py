"""MPEG-7 dataset utilities

Extracted from the original preprocessing script; only the dataset scanning, mask loading,
silhouette normalization, Zernike extraction and deep feature extraction functions used by
the AMC study are retained. Set the environment variable ZERNIKE_BASE to the project folder
if it differs from the repository root."""

from __future__ import annotations
import os
import copy
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights

BASE_DIR = Path(os.environ.get("ZERNIKE_BASE", Path(__file__).resolve().parents[1]))

DATA_ROOT_CANDIDATES = [
    BASE_DIR / "data" / "mpeg7",
    BASE_DIR / "data" / "MPEG7",
    BASE_DIR / "data" / "MPEG-7",
]

CACHE_DIR = BASE_DIR / "results_mpeg7_unified_preprocessing" / "feature_cache"

Z_ORDER = 32

Z_SIZE = 128

DEEP_SIZE = 224

IMG_EXTS = {
    ".gif", ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff", ".webp"
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_data_root():
    for p in DATA_ROOT_CANDIDATES:
        if p.exists():
            return p

    raise RuntimeError(
        "MPEG-7 folder not found.\nExpected one of:\n"
        + "\n".join(str(p) for p in DATA_ROOT_CANDIDATES)
    )

def parse_mpeg7_label(path: Path):
    """
    Standard MPEG-7 filenames are typically like:
      apple-1.gif
      apple-20.gif
      bat-3.gif
    """
    stem = path.stem.strip()

    m = re.match(
        r"^(.*?)-(\d+)$",
        stem
    )

    if m:
        return m.group(1).lower()

    return None

def scan_mpeg7():
    root = resolve_data_root()

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
    )

    parsed = [
        (
            p,
            parse_mpeg7_label(p)
        )
        for p in files
    ]

    # Ignore auxiliary / non-shape files.
    parsed = [
        (p, lab)
        for p, lab in parsed
        if lab is not None
    ]

    paths = [
        p
        for p, _ in parsed
    ]

    labels = np.asarray(
        [
            lab
            for _, lab in parsed
        ],
        dtype=object
    )

    print(f"MPEG-7 root: {root}")
    print(f"Parsed shape images: {len(paths)}")

    if len(paths) != 1400:
        raise RuntimeError(
            f"Expected 1400 MPEG-7 shape images after filtering, "
            f"but found {len(paths)}."
        )

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    if (
        len(unique) != 70
        or counts.min() != 20
        or counts.max() != 20
    ):
        print("\nClass count summary:")
        for c, n in zip(unique, counts):
            print(f"  {c}: {n}")

        raise RuntimeError(
            "Expected 70 classes × 20 images."
        )

    print(
        f"Classes: {len(unique)}"
    )

    print(
        f"Images per class: "
        f"{counts.min()}..{counts.max()}"
    )

    return paths, labels

def load_foreground_mask(path: Path):
    img = Image.open(path).convert("L")

    arr = (
        np.asarray(
            img,
            dtype=np.float32
        )
        / 255.0
    )

    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1],
    ])

    border_median = float(
        np.median(border)
    )

    # Infer polarity from border background.
    if border_median >= 0.5:
        mask = arr < 0.5
    else:
        mask = arr >= 0.5

    frac = float(
        mask.mean()
    )

    # Fallback for unusual polarity / threshold cases.
    if frac < 0.005 or frac > 0.95:
        dark = arr < 0.5
        bright = arr >= 0.5

        dark_frac = float(
            dark.mean()
        )

        bright_frac = float(
            bright.mean()
        )

        mask = (
            dark
            if dark_frac < bright_frac
            else bright
        )

    if not np.any(mask):
        raise RuntimeError(
            f"No foreground detected in {path.name}"
        )

    return mask.astype(
        np.uint8
    )

def normalize_silhouette(
    mask,
    out_size
):
    ys, xs = np.where(
        mask > 0
    )

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    crop = mask[
        y0:y1 + 1,
        x0:x1 + 1
    ]

    h, w = crop.shape
    side = max(h, w)

    margin = max(
        2,
        int(
            round(
                side * 0.10
            )
        )
    )

    canvas_side = (
        side
        + 2 * margin
    )

    canvas = np.zeros(
        (
            canvas_side,
            canvas_side
        ),
        dtype=np.uint8
    )

    yoff = (
        canvas_side - h
    ) // 2

    xoff = (
        canvas_side - w
    ) // 2

    canvas[
        yoff:yoff + h,
        xoff:xoff + w
    ] = crop

    pil = Image.fromarray(
        canvas * 255
    )

    pil = pil.resize(
        (
            out_size,
            out_size
        ),
        Image.Resampling.NEAREST
    )

    return (
        np.asarray(
            pil,
            dtype=np.float32
        )
        / 255.0
    )

def valid_nm_pairs(max_order):
    pairs = []

    for n in range(
        max_order + 1
    ):
        for m in range(
            0,
            n + 1
        ):
            if (
                (n - m) % 2
                == 0
            ):
                pairs.append(
                    (n, m)
                )

    return pairs

def radial_polynomial(
    n,
    m,
    r
):
    result = np.zeros_like(
        r,
        dtype=np.float64
    )

    upper = (
        n - m
    ) // 2

    for s in range(
        upper + 1
    ):
        coeff = (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial(
                    (n + m) // 2 - s
                )
                * math.factorial(
                    (n - m) // 2 - s
                )
            )
        )

        result += (
            coeff
            * np.power(
                r,
                n - 2 * s
            )
        )

    return result

class ZernikeExtractor:
    def __init__(
        self,
        size=128,
        max_order=32
    ):
        self.pairs = valid_nm_pairs(
            max_order
        )

        axis = np.linspace(
            -1.0,
            1.0,
            size,
            dtype=np.float64
        )

        xx, yy = np.meshgrid(
            axis,
            axis
        )

        rr = np.sqrt(
            xx * xx
            + yy * yy
        )

        theta = np.arctan2(
            yy,
            xx
        )

        disk = (
            rr <= 1.0
        )

        self.disk = disk
        self.rr = rr[disk]
        self.theta = theta[disk]

        self.basis = []

        for n, m in self.pairs:
            R = radial_polynomial(
                n,
                m,
                self.rr
            )

            basis = (
                R
                * np.exp(
                    -1j
                    * m
                    * self.theta
                )
            )

            self.basis.append(
                (
                    n,
                    basis
                )
            )

    def __call__(
        self,
        silhouette
    ):
        values = silhouette[
            self.disk
        ].astype(
            np.float64
        )

        feats = []

        for n, basis in self.basis:
            z = (
                (n + 1)
                / np.pi
                * np.mean(
                    values
                    * basis
                )
            )

            feats.append(
                abs(z)
            )

        feat = np.asarray(
            feats,
            dtype=np.float32
        )

        norm = float(
            np.linalg.norm(
                feat
            )
        )

        if norm > 0:
            feat /= norm

        return feat

def build_resnet18(device):
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    return model

def deep_tensor_from_silhouette(
    silhouette
):
    arr = np.stack(
        [
            silhouette,
            silhouette,
            silhouette
        ],
        axis=0
    ).astype(
        np.float32
    )

    x = torch.from_numpy(
        arr
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(
        3, 1, 1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(
        3, 1, 1
    )

    return (
        x - mean
    ) / std

@torch.no_grad()
def extract_all_features(
    paths,
    device
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    deep_cache = (
        CACHE_DIR
        / "resnet18_mpeg7_unified.npy"
    )

    z_cache = (
        CACHE_DIR
        / "zernike_n32_mpeg7_unified.npy"
    )

    if (
        deep_cache.exists()
        and z_cache.exists()
    ):
        deep = np.load(
            deep_cache
        ).astype(
            np.float32
        )

        z = np.load(
            z_cache
        ).astype(
            np.float32
        )

        if (
            deep.shape[0] == 1400
            and z.shape[0] == 1400
        ):
            print(
                "\nUsing cached unified MPEG-7 features."
            )

            return deep, z

    print(
        "\nExtracting unified MPEG-7 features..."
    )

    z_extractor = ZernikeExtractor(
        size=Z_SIZE,
        max_order=Z_ORDER
    )

    print(
        f"Zernike dimension: "
        f"{len(z_extractor.pairs)}"
    )

    deep_model = build_resnet18(
        device
    )

    z_features = []
    deep_tensors = []

    for i, path in enumerate(paths):
        mask = load_foreground_mask(
            path
        )

        sil_z = normalize_silhouette(
            mask,
            Z_SIZE
        )

        sil_deep = normalize_silhouette(
            mask,
            DEEP_SIZE
        )

        z_features.append(
            z_extractor(
                sil_z
            )
        )

        deep_tensors.append(
            deep_tensor_from_silhouette(
                sil_deep
            )
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == len(paths)
        ):
            print(
                f"  Preprocessed "
                f"{i + 1}/{len(paths)}"
            )

    deep_features = []

    for start in range(
        0,
        len(paths),
        64
    ):
        batch = torch.stack(
            deep_tensors[
                start:start + 64
            ],
            dim=0
        ).to(
            device
        )

        feat = deep_model(
            batch
        )

        feat = F.normalize(
            feat,
            p=2,
            dim=1
        )

        deep_features.append(
            feat.cpu().numpy()
        )

    deep = np.concatenate(
        deep_features,
        axis=0
    ).astype(
        np.float32
    )

    z = np.stack(
        z_features,
        axis=0
    ).astype(
        np.float32
    )

    np.save(
        deep_cache,
        deep
    )

    np.save(
        z_cache,
        z
    )

    print(
        "\nSaved feature cache:"
    )

    print(
        " ",
        deep_cache
    )

    print(
        " ",
        z_cache
    )

    return deep, z

def normalize_rows(x):
    x = x.astype(
        np.float64
    )

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    norms[
        norms == 0
    ] = 1.0

    return (
        x / norms
    ).astype(
        np.float32
    )
