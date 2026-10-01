"""MPEG-7 deep features for ResNet-18, ResNet-50, EfficientNet-B0, ViT-B/16 and Swin-T

Extracted from the original feature-extraction script; only the functions that compute and
cache the frozen features used by the AMC study are retained. The MPEG-7 preprocessing functions
(mask loading, silhouette normalization, Zernike extractor) are imported from data_mpeg7.py."""

from __future__ import annotations
import os
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import data_mpeg7 as base

BASE_DIR = Path(os.environ.get("ZERNIKE_BASE", Path(__file__).resolve().parents[1]))

OUT_DIR = BASE_DIR / "results_mpeg7_multibackbone_zernike"

CACHE_DIR = OUT_DIR / "feature_cache"

DEEP_SIZE = 224

Z_SIZE = 128

Z_ORDER = 32

EXTRACT_BATCH_SIZE = {
    "resnet18": 64,
    "resnet50": 48,
    "efficientnet_b0": 48,
    "vit_b_16": 16,
    "swin_t": 24,
}

BACKBONES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "vit_b_16",
    "swin_t",
]

def build_backbone(name: str, device: torch.device):
    """
    Returns a pretrained feature extractor with the classification head removed.
    The returned output is the penultimate/pre-classification feature vector.
    """
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()

    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()

    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()

    elif name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()

    elif name == "swin_t":
        model = swin_t(weights=Swin_T_Weights.DEFAULT)
        model.head = nn.Identity()

    else:
        raise ValueError(f"Unknown backbone: {name}")

    model.eval()
    model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model

def extract_zernike_features(paths):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / "zernike_n32_mpeg7_unified.npy"

    # Reuse finalized cache if it already exists in the original experiment.
    old_cache = (
        BASE_DIR
        / "results_mpeg7_unified_preprocessing"
        / "feature_cache"
        / "zernike_n32_mpeg7_unified.npy"
    )

    if cache_path.exists():
        z = np.load(cache_path).astype(np.float32)
        if z.shape[0] == len(paths):
            print(f"Using Zernike cache: {cache_path}")
            return z

    if old_cache.exists():
        z = np.load(old_cache).astype(np.float32)
        if z.shape[0] == len(paths):
            np.save(cache_path, z)
            print(f"Reused finalized Zernike cache: {old_cache}")
            return z

    print("Extracting Zernike features...")

    extractor = base.ZernikeExtractor(
        size=Z_SIZE,
        max_order=Z_ORDER,
    )

    features = []

    for i, path in enumerate(paths):
        mask = base.load_foreground_mask(path)
        silhouette = base.normalize_silhouette(mask, Z_SIZE)
        features.append(extractor(silhouette))

        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  Zernike {i + 1}/{len(paths)}")

    z = np.asarray(features, dtype=np.float32)
    z = base.normalize_rows(z)
    np.save(cache_path, z)

    print(f"Saved Zernike cache: {cache_path}")
    print(f"Zernike shape: {z.shape}")

    return z

@torch.no_grad()
def extract_deep_features(paths, backbone_name, device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / f"{backbone_name}_mpeg7_unified.npy"

    # Reuse the finalized ResNet-18 cache when possible.
    old_resnet18_cache = (
        BASE_DIR
        / "results_mpeg7_unified_preprocessing"
        / "feature_cache"
        / "resnet18_mpeg7_unified.npy"
    )

    if cache_path.exists():
        deep = np.load(cache_path).astype(np.float32)
        if deep.shape[0] == len(paths):
            print(f"Using Deep cache: {cache_path}")
            return deep

    if backbone_name == "resnet18" and old_resnet18_cache.exists():
        deep = np.load(old_resnet18_cache).astype(np.float32)
        if deep.shape[0] == len(paths):
            deep = base.normalize_rows(deep)
            np.save(cache_path, deep)
            print(f"Reused finalized ResNet-18 cache: {old_resnet18_cache}")
            return deep

    print(f"\nExtracting {backbone_name} features...")
    model = build_backbone(backbone_name, device)
    batch_size = EXTRACT_BATCH_SIZE[backbone_name]

    all_features = []

    for start in range(0, len(paths), batch_size):
        stop = min(start + batch_size, len(paths))

        tensors = []

        for path in paths[start:stop]:
            mask = base.load_foreground_mask(path)
            silhouette = base.normalize_silhouette(mask, DEEP_SIZE)
            tensors.append(base.deep_tensor_from_silhouette(silhouette))

        batch = torch.stack(tensors, dim=0).to(device)

        output = model(batch)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim > 2:
            output = torch.flatten(output, start_dim=1)

        all_features.append(
            output.detach().cpu().numpy().astype(np.float32)
        )

        print(
            f"  {backbone_name}: {stop}/{len(paths)}",
            end="\r" if stop < len(paths) else "\n",
        )

    deep = np.concatenate(all_features, axis=0)
    deep = base.normalize_rows(deep)
    np.save(cache_path, deep)

    print(f"Saved Deep cache: {cache_path}")
    print(f"{backbone_name} feature shape: {deep.shape}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return deep
