"""Recompute the cached MPEG-7 features (Spyder: F5).
Writes the feature caches of the five ImageNet backbones, the baseline Zernike descriptor and DINOv2
to the feature_cache folders read by phase5_data.py. Requires torch, torchvision and the MPEG-7 data
in data/mpeg7 under the repository root (or under the folder given by ZERNIKE_BASE)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import data_mpeg7
import features_mpeg7_backbones as fb
import features_mpeg7_dinov2 as fd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels = data_mpeg7.scan_mpeg7()
    print(f"MPEG-7: {len(paths)} shapes, device {device}")
    fb.extract_zernike_features(paths)
    for name in fb.BACKBONES:
        print("backbone", name)
        fb.extract_deep_features(paths, name, device)
    print("DINOv2")
    fd.extract_features(paths, device)
    print("done")

if __name__ == "__main__":
    main()
