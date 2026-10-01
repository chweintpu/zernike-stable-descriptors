"""ETHZ Shape Classes dataset utilities

Extracted from the original experiment script; only the dataset scanning, mask loading,
splitting and feature-extraction functions used by the AMC study are retained.
Set the environment variable ZERNIKE_BASE to the project folder if it differs from the
repository root."""

from __future__ import annotations
import os
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import (
    ResNet18_Weights,
    ViT_B_16_Weights,
    resnet18,
    vit_b_16,
)

BASE_DIR = Path(os.environ.get("ZERNIKE_BASE", Path(__file__).resolve().parents[1]))

DATASET_ROOT = BASE_DIR / "ethz_instance_dataset"

OUTPUT_ROOT = BASE_DIR / "ethz_external_results_harmonized"

NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEEP_BATCH_SIZE = 32

FORCE_RECOMPUTE_DEEP = False

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def imread_windows_safe(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    path = Path(path)

    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    if data.size == 0:
        raise RuntimeError(
            f"Empty/unreadable image file: {path}"
        )

    image = cv2.imdecode(
        data,
        flags,
    )

    if image is None:
        raise RuntimeError(
            f"Could not decode image: {path}"
        )

    return image

REQUIRED_COLUMNS = {
    "sample_id",
    "class",
    "source_group",
    "image_path",
    "mask_path",
}

def load_metadata() -> pd.DataFrame:
    path = DATASET_ROOT / "metadata.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found: {path}"
        )

    df = pd.read_csv(
        path,
        dtype={
            "sample_id": str,
            "class": str,
            "source_group": str,
        },
    )

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise RuntimeError(
            f"metadata.csv is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    df["image_path_obj"] = [
        Path(str(x))
        for x in df["image_path"]
    ]

    df["mask_path_obj"] = [
        Path(str(x))
        for x in df["mask_path"]
    ]

    missing_images = [
        p for p in df["image_path_obj"]
        if not p.exists()
    ]

    missing_masks = [
        p for p in df["mask_path_obj"]
        if not p.exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image crops are missing. "
            f"Example: {missing_images[0]}"
        )

    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} masks are missing. "
            f"Example: {missing_masks[0]}"
        )

    classes = sorted(
        df["class"].unique().tolist()
    )

    class_to_label = {
        c: i
        for i, c in enumerate(classes)
    }

    df["label"] = (
        df["class"]
        .map(class_to_label)
        .astype(int)
    )

    df = df.sort_values(
        [
            "class",
            "source_group",
            "sample_id",
        ]
    ).reset_index(drop=True)

    return df

def validate_metadata(df: pd.DataFrame) -> None:
    print("\nDataset validation")
    print("------------------")
    print(
        f"Usable object instances: {len(df)}"
    )
    print(
        f"Semantic classes: {df['class'].nunique()}"
    )
    print(
        f"Unique source images: "
        f"{df['source_group'].nunique()}"
    )

    class_counts = (
        df.groupby("class")
        .size()
        .sort_index()
    )

    group_counts = (
        df.groupby("class")["source_group"]
        .nunique()
        .sort_index()
    )

    print("\nInstances by class:")
    print(class_counts.to_string())

    print("\nSource-image groups by class:")
    print(group_counts.to_string())

    if len(df) != 286:
        print(
            "\nWARNING: expected 286 usable instances "
            "from the prepared ETHZ release."
        )

    if df["class"].nunique() != 5:
        raise RuntimeError(
            "Expected 5 ETHZ classes."
        )

@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

def _partition_group_list(
    group_names: List[str],
    rng: np.random.Generator,
):
    names = list(group_names)
    rng.shuffle(names)

    n = len(names)

    # Approx. 50/25/25 at group level.
    n_train = int(round(0.50 * n))
    n_val = int(round(0.25 * n))

    # Ensure at least one group in every partition.
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_groups = names[:n_train]
    val_groups = names[
        n_train:n_train + n_val
    ]
    test_groups = names[
        n_train + n_val:
    ]

    return (
        train_groups,
        val_groups,
        test_groups,
    )

def make_group_safe_split(
    df: pd.DataFrame,
    seed: int,
) -> SplitIndices:
    """
    Class-stratified split at source-image group level.

    All object instances from one original source image stay together.
    """
    rng = np.random.default_rng(seed)

    train_groups = set()
    val_groups = set()
    test_groups = set()

    for class_name in sorted(
        df["class"].unique()
    ):
        class_groups = sorted(
            df.loc[
                df["class"] == class_name,
                "source_group",
            ].unique().tolist()
        )

        (
            cls_train,
            cls_val,
            cls_test,
        ) = _partition_group_list(
            class_groups,
            rng,
        )

        train_groups.update(cls_train)
        val_groups.update(cls_val)
        test_groups.update(cls_test)

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise AssertionError(
            "Source-group leakage detected."
        )

    train_idx = df.index[
        df["source_group"].isin(
            train_groups
        )
    ].to_numpy(dtype=int)

    val_idx = df.index[
        df["source_group"].isin(
            val_groups
        )
    ].to_numpy(dtype=int)

    test_idx = df.index[
        df["source_group"].isin(
            test_groups
        )
    ].to_numpy(dtype=int)

    # Check every class appears in every partition.
    for split_name, idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        present = set(
            df.loc[idx, "class"]
        )

        expected = set(
            df["class"].unique()
        )

        if present != expected:
            raise RuntimeError(
                f"{split_name} split is missing classes: "
                f"{sorted(expected - present)}"
            )

    return SplitIndices(
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )

class RGBInstanceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.loc[
            index,
            "image_path_obj",
        ]

        bgr = imread_windows_safe(
            path,
            cv2.IMREAD_COLOR,
        )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(rgb)
        x = self.transform(image)

        return x, index

def make_backbone(name: str):
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(
            weights=weights
        )
        deep_dim = model.fc.in_features
        model.fc = nn.Identity()
        transform = weights.transforms()

    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(
            weights=weights
        )
        deep_dim = model.heads.head.in_features
        model.heads = nn.Identity()
        transform = weights.transforms()

    else:
        raise ValueError(
            f"Unknown backbone: {name}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    model.eval()
    model.to(DEVICE)

    return (
        model,
        transform,
        deep_dim,
    )

@torch.no_grad()
def extract_deep_features(
    df: pd.DataFrame,
    backbone_name: str,
):
    cache_dir = (
        OUTPUT_ROOT
        / "feature_cache"
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        cache_dir
        / f"{backbone_name}_deep.npy"
    )

    meta_path = (
        cache_dir
        / f"{backbone_name}_deep_meta.json"
    )

    if (
        cache_path.exists()
        and meta_path.exists()
        and not FORCE_RECOMPUTE_DEEP
    ):
        features = np.load(cache_path)
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        if features.shape[0] == len(df):
            print(
                f"Loaded cached Deep features: "
                f"{backbone_name} {features.shape}"
            )
            return (
                features.astype(np.float32),
                int(meta["deep_dim"]),
            )

    print(
        f"\nExtracting Deep features: "
        f"{backbone_name}"
    )

    (
        model,
        transform,
        deep_dim,
    ) = make_backbone(
        backbone_name
    )

    dataset = RGBInstanceDataset(
        df,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=DEEP_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    features = np.zeros(
        (len(df), deep_dim),
        dtype=np.float32,
    )

    for x, idx in loader:
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        f = model(x)

        if f.ndim > 2:
            f = torch.flatten(
                f,
                1,
            )

        f = F.normalize(
            f,
            p=2,
            dim=1,
        )

        features[
            idx.numpy()
        ] = f.cpu().numpy().astype(
            np.float32
        )

    np.save(
        cache_path,
        features,
    )

    meta_path.write_text(
        json.dumps(
            {
                "backbone": backbone_name,
                "deep_dim": deep_dim,
                "num_samples": len(df),
                "normalized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        features,
        deep_dim,
    )

def read_binary_mask_for_zernike(
    mask_path: Path,
):
    mask = imread_windows_safe(
        mask_path,
        cv2.IMREAD_GRAYSCALE,
    )

    return (
        mask > 0
    ).astype(
        np.uint8
    )
