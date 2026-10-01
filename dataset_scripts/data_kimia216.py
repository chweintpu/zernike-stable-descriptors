"""Kimia-216 dataset utilities

Extracted from the original experiment script; only the dataset scanning, mask loading,
splitting and feature-extraction functions used by the AMC study are retained.
Set the environment variable ZERNIKE_BASE to the project folder if it differs from the
repository root."""

from __future__ import annotations
import os
import copy
import csv
import json
import math
import random
import re
from collections import Counter
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
    BASE_DIR / "data" / "Kimia216",
    BASE_DIR / "data" / "kimia216",
]

CACHE_DIR = BASE_DIR / "results_kimia216_cross_dataset" / "feature_cache"

TRAIN_PER_CLASS = 8

VAL_PER_CLASS = 2

Z_ORDER = 32

Z_SIZE = 128

DEEP_SIZE = 224

IMG_EXTS = {
    ".png", ".gif", ".bmp", ".jpg", ".jpeg",
    ".tif", ".tiff", ".webp"
}

KIMIA216_CLASSES = [
    "bird",
    "bone",
    "brick",
    "camel",
    "car",
    "children",
    "classic",
    "elephant",
    "face",
    "fork",
    "fountain",
    "glass",
    "hammer",
    "heart",
    "key",
    "misk",
    "ray",
    "turtle",
]

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
        "Kimia-216 folder not found.\n"
        "Expected one of:\n"
        + "\n".join(str(p) for p in DATA_ROOT_CANDIDATES)
    )

def parse_prefix_label(path: Path):
    """
    Robust Kimia-216 filename parser.

    Supports names such as:
      bird02.jpg
      Glas01.jpg
      elephant_07.bmp
      fountain12.gif

    Some distributed Kimia-216 copies use shortened prefixes,
    e.g. "Glas" instead of "glass".
    """
    stem = path.stem.lower()

    # Known filename-prefix aliases found in distributed copies
    alias_map = {
        "glas": "glass",
        "glass": "glass",
    }

    # First try aliases / shortened names
    for prefix, canonical in alias_map.items():
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            if (
                rest == ""
                or re.fullmatch(r"[\s_\-]*\d+", rest)
            ):
                return canonical

    # Then try canonical class names
    for cls in KIMIA216_CLASSES:
        if stem.startswith(cls):
            rest = stem[len(cls):]

            if (
                rest == ""
                or re.fullmatch(r"[\s_\-]*\d+", rest)
            ):
                return cls

    return None

def scan_kimia216():
    root = resolve_data_root()

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTS
    )

    print(f"Kimia-216 root: {root}")
    print(f"Image files found: {len(files)}")

    if len(files) != 216:
        raise RuntimeError(
            f"Expected exactly 216 image files, found {len(files)}."
        )

    # --------------------------------------------------------
    # Strategy A: 18 class subfolders × 12 images
    # --------------------------------------------------------
    parent_labels = [
        p.parent.name.lower()
        for p in files
    ]

    parent_counts = Counter(parent_labels)

    if (
        len(parent_counts) == 18
        and sorted(parent_counts.values()) == [12] * 18
    ):
        labels = np.asarray(
            parent_labels,
            dtype=object
        )

        print("Label inference: class subfolders")

    else:
        # ----------------------------------------------------
        # Strategy B: class prefix in filename
        # ----------------------------------------------------
        inferred = [
            parse_prefix_label(p)
            for p in files
        ]

        if all(x is not None for x in inferred):
            counts = Counter(inferred)

            if (
                len(counts) == 18
                and sorted(counts.values()) == [12] * 18
            ):
                labels = np.asarray(
                    inferred,
                    dtype=object
                )

                print("Label inference: filename prefixes")

            else:
                print("\nFilename-prefix counts:")
                print(dict(counts))

                raise RuntimeError(
                    "Could not confirm 18 classes × 12 images."
                )

        else:
            bad = [
                p.name
                for p, lab in zip(files, inferred)
                if lab is None
            ]

            print("\nCould not infer labels for:")
            for x in bad[:30]:
                print(" ", x)

            raise RuntimeError(
                "Please arrange Kimia-216 as either:\n"
                "1) 18 class subfolders × 12 images, or\n"
                "2) filenames beginning with the class name, "
                "e.g. bird-1.gif, elephant-2.gif."
            )

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    if (
        len(unique) != 18
        or counts.min() != 12
        or counts.max() != 12
    ):
        raise RuntimeError(
            "Kimia-216 class distribution is not 18 × 12."
        )

    print("\nClass counts:")
    for c, n in zip(unique, counts):
        print(f"  {c}: {n}")

    return files, labels

def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    for cls in sorted(set(labels.tolist())):
        ids = np.where(
            labels == cls
        )[0].astype(np.int64)

        if len(ids) != 12:
            raise RuntimeError(
                f"{cls}: expected 12 images, found {len(ids)}"
            )

        ids = ids.copy()
        rng.shuffle(ids)

        train_idx.extend(
            ids[:TRAIN_PER_CLASS]
        )

        val_idx.extend(
            ids[
                TRAIN_PER_CLASS:
                TRAIN_PER_CLASS + VAL_PER_CLASS
            ]
        )

        test_idx.extend(
            ids[
                TRAIN_PER_CLASS + VAL_PER_CLASS:
            ]
        )

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )

def load_foreground_mask(path: Path):
    img = Image.open(path).convert("L")

    arr = (
        np.asarray(
            img,
            dtype=np.float32
        ) / 255.0
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

    if border_median >= 0.5:
        mask = arr < 0.5
    else:
        mask = arr >= 0.5

    frac = float(
        mask.mean()
    )

    if frac < 0.005 or frac > 0.95:
        dark = arr < 0.5
        bright = arr >= 0.5

        mask = (
            dark
            if dark.mean() < bright.mean()
            else bright
        )

    if not np.any(mask):
        raise RuntimeError(
            f"No foreground detected in {path.name}"
        )

    return mask.astype(
        np.uint8
    )

def normalize_silhouette(mask, out_size):
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

        norm = np.linalg.norm(
            feat
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
        / "resnet18_kimia216.npy"
    )

    z_cache = (
        CACHE_DIR
        / "zernike_n32_kimia216.npy"
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
            deep.shape[0] == 216
            and z.shape[0] == 216
        ):
            print(
                "\nUsing cached Kimia-216 features."
            )

            return deep, z

    print(
        "\nExtracting Kimia-216 features..."
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

    deep_tensors = []
    z_features = []

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
            (i + 1) % 25 == 0
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
        32
    ):
        batch = torch.stack(
            deep_tensors[
                start:start + 32
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
