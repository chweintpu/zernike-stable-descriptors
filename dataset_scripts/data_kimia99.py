"""Kimia-99 dataset utilities

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
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights

BASE_DIR = Path(os.environ.get("ZERNIKE_BASE", Path(__file__).resolve().parents[1]))

DATA_ROOT = BASE_DIR / "data" / "Kimia99"

PNG_DIR = DATA_ROOT / "png"

MAT_DIR = DATA_ROOT / "mat"

CACHE_DIR = BASE_DIR / "results_kimia99_cross_dataset_v2" / "feature_cache"

TRAIN_PER_CLASS = 7

VAL_PER_CLASS = 2

Z_ORDER = 32

Z_SIZE = 128

DEEP_SIZE = 224

KIMIA99_CLASSES = {
    "fish": [
        "bonefishes",
        "bonefishesocc1",
        "dogfishsharks",
        "fish14",
        "fish23",
        "fish28",
        "fish30",
        "herrings",
        "mullets",
        "swordfishes",
        "whalesharks",
    ],

    "rabbits": [
        "bunny04",
        "desertcottontail",
        "easterncottontail",
        "marshrabbit",
        "mountaincottontail",
        "mountaincottontailocc1",
        "mountaincottontailocc2",
        "mountaincottontailrot",
        "pygmyrabbit",
        "swamprabbit",
        "swamprabbitocc2",
    ],

    "dudes": [
        "dude0",
        "dude1",
        "dude10",
        "dude11",
        "dude12",
        "dude2",
        "dude4",
        "dude5",
        "dude6",
        "dude7",
        "dude8",
    ],

    "hands": [
        "hand",
        "hand2",
        "hand2occ1",
        "hand2occ2",
        "hand2occ3",
        "hand3",
        "hand90",
        "handbent1",
        "handbent2",
        "handdeform",
        "handdeform2",
    ],

    "tools": [
        "tool04",
        "tool04bent1",
        "tool07",
        "tool08",
        "tool09",
        "tool12",
        "tool17",
        "tool22",
        "tool27",
        "tool38",
        "tool44",
    ],

    "aircraft_f": [
        "f15",
        "f16",
        "f16occ1",
        "fgen1ap",
        "fgen1bp",
        "fgen1ep",
        "fgen1fp",
        "fgen2dp",
        "fgen2fp",
        "fgen3bp",
        "fgen5cp",
    ],

    "aircraft_other": [
        "harrier",
        "harrierocc1",
        "harrierocc2",
        "harrierocc3",
        "mgen1bp",
        "mgen2ap",
        "mgen2fp",
        "phantom",
        "phantomocc1",
        "skyhawk",
        "skyhawkocc1",
    ],

    "quadrupeds": [
        "calf1",
        "calf2",
        "cat1",
        "cat2",
        "cow1",
        "cow2",
        "dog1",
        "dog2",
        "dog3",
        "donkey1",
        "fox1",
    ],

    "kk_group": [
        "kk0728",
        "kk0729",
        "kk0731",
        "kk0732",
        "kk0735",
        "kk0736",
        "kk0737",
        "kk0738",
        "kk0739",
        "kk0740",
        "kk0741",
    ],
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_filename_to_class():
    mapping = {}

    for class_name, stems in KIMIA99_CLASSES.items():
        if len(stems) != 11:
            raise RuntimeError(
                f"Class {class_name} has {len(stems)} entries; expected 11."
            )

        for stem in stems:
            key = stem.lower()

            if key in mapping:
                raise RuntimeError(
                    f"Duplicate Kimia stem in class table: {stem}"
                )

            mapping[key] = class_name

    if len(mapping) != 99:
        raise RuntimeError(
            f"Class table contains {len(mapping)} unique shapes; expected 99."
        )

    return mapping

def scan_kimia99():
    if not PNG_DIR.exists():
        raise RuntimeError(
            f"PNG folder not found:\n{PNG_DIR}"
        )

    png_files = sorted(
        p for p in PNG_DIR.glob("*.png")
        if p.is_file()
    )

    print(f"PNG directory: {PNG_DIR}")
    print(f"PNG files found: {len(png_files)}")

    if len(png_files) != 99:
        raise RuntimeError(
            f"Expected exactly 99 PNG images, found {len(png_files)}."
        )

    mapping = build_filename_to_class()

    found_stems = {
        p.stem.lower()
        for p in png_files
    }

    expected_stems = set(
        mapping.keys()
    )

    missing = sorted(
        expected_stems - found_stems
    )

    unexpected = sorted(
        found_stems - expected_stems
    )

    if missing or unexpected:
        print("\nMissing expected stems:")
        print(missing)

        print("\nUnexpected PNG stems:")
        print(unexpected)

        raise RuntimeError(
            "PNG filenames do not match the expected Kimia-99 file list."
        )

    paths = []
    labels = []

    # Keep deterministic class-major order.
    for class_name in KIMIA99_CLASSES:
        for stem in KIMIA99_CLASSES[class_name]:
            p = PNG_DIR / f"{stem}.png"

            if not p.exists():
                raise RuntimeError(
                    f"Missing PNG file: {p}"
                )

            paths.append(p)
            labels.append(class_name)

    labels = np.asarray(
        labels,
        dtype=object
    )

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print("\nClass counts:")
    for c, n in zip(unique, counts):
        print(f"  {c}: {n}")

    print(f"\nMAT folder exists: {MAT_DIR.exists()}")
    if MAT_DIR.exists():
        mat_count = len(
            list(
                MAT_DIR.glob("*.mat")
            )
        )

        print(
            f"MAT files found: {mat_count} "
            "(not used in this experiment)"
        )

    return paths, labels

def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)

    by_class = {}

    for class_name in KIMIA99_CLASSES:
        by_class[class_name] = np.where(
            labels == class_name
        )[0]

    train_idx = []
    val_idx = []
    test_idx = []

    for class_name in KIMIA99_CLASSES:
        ids = np.asarray(
            by_class[class_name],
            dtype=np.int64
        ).copy()

        if len(ids) != 11:
            raise RuntimeError(
                f"{class_name}: expected 11 images, found {len(ids)}"
            )

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

    # Border is treated as background.
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
        self.size = size
        self.max_order = max_order

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

        disk = rr <= 1.0

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
            np.linalg.norm(feat)
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
        3,
        1,
        1
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
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
        / "resnet18_kimia99_v2.npy"
    )

    z_cache = (
        CACHE_DIR
        / "zernike_n32_kimia99_v2.npy"
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
            deep.shape[0] == 99
            and z.shape[0] == 99
        ):
            print(
                "\nUsing cached Kimia-99 v2 features."
            )

            return (
                deep,
                z
            )

    print(
        "\nExtracting Kimia-99 features..."
    )

    z_extractor = ZernikeExtractor(
        size=Z_SIZE,
        max_order=Z_ORDER
    )

    print(
        f"Zernike feature dimension: "
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
            (i + 1) % 10 == 0
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

    return (
        deep,
        z
    )
