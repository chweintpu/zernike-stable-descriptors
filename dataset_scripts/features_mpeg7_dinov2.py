"""MPEG-7 DINOv2 ViT-B/14 features

Extracted from the original feature-extraction script; only the functions that compute and
cache the frozen features used by the AMC study are retained. The MPEG-7 preprocessing functions
(mask loading, silhouette normalization, Zernike extractor) are imported from data_mpeg7.py."""

from __future__ import annotations
import os
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(os.environ.get("ZERNIKE_BASE", Path(__file__).resolve().parents[1]))

OUT_DIR = HERE / "results_mpeg7_dinov2_zernike_foundation"

CACHE_DIR = OUT_DIR / "feature_cache"

DINO_MODEL_NAME = "dinov2_vitb14"

DINO_DIM_EXPECTED = 768

DINO_INPUT_SIZE = 224

DINO_BATCH_SIZE = 32

def load_base_module():
    import data_mpeg7
    return data_mpeg7


base = load_base_module()

def configure_secure_ssl():
    """
    Use certifi's CA bundle for HTTPS verification.

    This keeps certificate verification ENABLED. It does not disable SSL checks.
    It is mainly useful for Windows/Anaconda environments whose Python SSL
    configuration cannot locate the CA bundle automatically.
    """
    import os
    import ssl

    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError(
            "The 'certifi' package is required for secure HTTPS setup.\n"
            "Install it in the same Anaconda environment used by Spyder:\n"
            "  conda install -c conda-forge certifi ca-certificates openssl"
        ) from exc

    ca_file = certifi.where()
    os.environ["SSL_CERT_FILE"] = ca_file
    os.environ["REQUESTS_CA_BUNDLE"] = ca_file

    # urllib/torch.hub uses Python's default HTTPS context.
    ssl._create_default_https_context = (
        lambda: ssl.create_default_context(cafile=ca_file)
    )

    print(f"Secure CA bundle:\n  {ca_file}")

def build_dinov2(device):
    """
    Loading order:
      1. If a local 'dinov2' repository exists next to this script, use it.
      2. Otherwise use official Meta GitHub via torch.hub with secure SSL.
    """
    local_repo = HERE / "dinov2"

    if local_repo.exists() and (local_repo / "hubconf.py").exists():
        print("\nLoading DINOv2 from local repository:")
        print(f"  {local_repo}")
        model = torch.hub.load(
            str(local_repo),
            DINO_MODEL_NAME,
            source="local"
        )
    else:
        configure_secure_ssl()

        print("\nLoading official DINOv2 ViT-B/14 from Meta via torch.hub...")
        print("The first run requires internet access to download the repo/model.")

        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                DINO_MODEL_NAME,
                trust_repo=True
            )
        except Exception as exc:
            raise RuntimeError(
                "\nDINOv2 download failed.\n\n"
                "Recommended fix in Anaconda Prompt:\n"
                "  conda install -c conda-forge ca-certificates certifi openssl\n\n"
                "Then restart Spyder and run this script again.\n\n"
                "If your network still blocks GitHub certificates, download/clone "
                "the official facebookresearch/dinov2 repository into:\n"
                f"  {local_repo}\n"
                "The script will then use the local repository automatically "
                "without contacting GitHub for the repo code.\n"
            ) from exc

    model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model

def tensor_from_silhouette(silhouette):
    """Use the same 3-channel + ImageNet normalization as finalized MPEG-7."""
    arr = np.stack(
        [silhouette, silhouette, silhouette],
        axis=0
    ).astype(np.float32)

    x = torch.from_numpy(arr)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(3, 1, 1)

    return (x - mean) / std

def unpack_dino_output(output):
    """
    Official hub DINOv2 backbone normally returns an [N, 768] tensor.
    This helper also tolerates dict/tuple outputs.
    """
    if torch.is_tensor(output):
        feat = output
    elif isinstance(output, dict):
        for key in (
            "x_norm_clstoken",
            "x_prenorm",
            "cls_token",
            "features",
        ):
            if key in output and torch.is_tensor(output[key]):
                feat = output[key]
                break
        else:
            raise RuntimeError(
                f"Unsupported DINOv2 dict keys: {list(output.keys())}"
            )
    elif isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            raise RuntimeError("DINOv2 returned no tensor output.")
        feat = tensors[0]
    else:
        raise RuntimeError(
            f"Unsupported DINOv2 output type: {type(output)}"
        )

    if feat.ndim == 3:
        # If token sequence is returned, use CLS token.
        feat = feat[:, 0, :]

    if feat.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D DINO feature matrix, got shape {tuple(feat.shape)}"
        )

    return feat

@torch.no_grad()
def extract_features(paths, device):
    deep_cache = CACHE_DIR / "dinov2_vitb14_mpeg7_224.npy"
    z_cache = CACHE_DIR / "zernike_n32_mpeg7.npy"

    if deep_cache.exists() and z_cache.exists():
        print("\nUsing cached DINOv2 and Zernike features.")
        deep = np.load(deep_cache).astype(np.float32)
        z = np.load(z_cache).astype(np.float32)
        return deep, z

    model = build_dinov2(device)
    z_extractor = base.ZernikeExtractor(
        size=base.Z_SIZE,
        max_order=base.Z_ORDER
    )

    deep_tensors = []
    z_features = []

    print("\nPreprocessing MPEG-7 silhouettes...")
    for i, path in enumerate(paths):
        mask = base.load_foreground_mask(path)

        sil_z = base.normalize_silhouette(
            mask,
            base.Z_SIZE
        )
        sil_deep = base.normalize_silhouette(
            mask,
            DINO_INPUT_SIZE
        )

        z_features.append(
            z_extractor(sil_z)
        )
        deep_tensors.append(
            tensor_from_silhouette(sil_deep)
        )

        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            print(f"  preprocessed {i + 1}/{len(paths)}")

    z = np.asarray(
        z_features,
        dtype=np.float32
    )

    deep_features = []
    print("\nExtracting frozen DINOv2 features...")
    for start in range(0, len(paths), DINO_BATCH_SIZE):
        batch = torch.stack(
            deep_tensors[start:start + DINO_BATCH_SIZE]
        ).to(device)

        output = model(batch)
        feat = unpack_dino_output(output)

        deep_features.append(
            feat.detach().cpu().numpy().astype(np.float32)
        )

        print(
            f"  extracted {min(start + DINO_BATCH_SIZE, len(paths))}"
            f"/{len(paths)}"
        )

    deep = np.concatenate(
        deep_features,
        axis=0
    )

    if deep.shape[1] != DINO_DIM_EXPECTED:
        raise RuntimeError(
            f"Expected DINOv2 ViT-B/14 dimension {DINO_DIM_EXPECTED}, "
            f"got {deep.shape[1]}."
        )

    np.save(deep_cache, deep)
    np.save(z_cache, z)

    print("\nSaved feature cache:")
    print(f"  {deep_cache}")
    print(f"  {z_cache}")

    return deep, z
