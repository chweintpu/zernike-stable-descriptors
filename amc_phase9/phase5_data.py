"""Dataset loaders for phase 5. Kimia-216, Kimia-99 and ETHZ reuse the scan / mask / split
functions of the dataset modules in dataset_scripts/, so the file order,
labels and splits follow the original dataset scripts. Needs the dataset-loading environment (torch, cv2, pandas)."""
import importlib.util, io, contextlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import zernike_amc as za

SEEDS = list(range(3001, 3011))    # independent split set for the AMC study
K_GRID = [2.0, 1.75, 2.25, 1.5, 2.5, 3.0]          # preference order for ties: a-priori k = 2 first


REPO = Path(__file__).resolve().parents[1]

def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod

def _mpeg7_split(labels, seed, train=12, val=4):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[str(lab)].append(i)
    tr, va, te = [], [], []
    for lab in sorted(by):
        ids = np.asarray(by[lab], np.int64).copy(); rng.shuffle(ids)
        tr += list(ids[:train]); va += list(ids[train:train + val]); te += list(ids[train + val:])
    return np.array(tr), np.array(va), np.array(te)

def load_dataset(name, base):
    """Returns dict: items (paths or masks), labels (str array), splits [(seed,tr,va,te)],
    deep {backbone: path}, leg_cache (baseline Zernike cache path)."""
    base = Path(base)
    if name == "MPEG-7":
        fs = za.list_mpeg7(base / "data" / "mpeg7")
        labels = np.array([za.label_of(f) for f in fs])
        mb = base / "results_mpeg7_multibackbone_zernike" / "feature_cache"
        deep = {"ResNet-18": mb / "resnet18_mpeg7_unified.npy", "ResNet-50": mb / "resnet50_mpeg7_unified.npy",
                "EfficientNet-B0": mb / "efficientnet_b0_mpeg7_unified.npy", "ViT-B/16": mb / "vit_b_16_mpeg7_unified.npy",
                "Swin-T": mb / "swin_t_mpeg7_unified.npy",
                "DINOv2": base / "results_mpeg7_dinov2_zernike_foundation/feature_cache/dinov2_vitb14_mpeg7_224.npy"}
        return dict(items=[str(f) for f in fs], labels=labels,
                    splits=[(s, *_mpeg7_split(labels, s)) for s in SEEDS], deep=deep,
                    leg_cache=base / "results_mpeg7_unified_preprocessing/feature_cache/zernike_n32_mpeg7_unified.npy")
    if name == "Kimia-216":
        mod = _import(REPO / "dataset_scripts" / "data_kimia216.py", "k216")
        with contextlib.redirect_stdout(io.StringIO()):
            files, labels = mod.scan_kimia216()
        labels = np.asarray(labels)
        cache = base / "results_kimia216_cross_dataset" / "feature_cache"
        return dict(items=[mod.load_foreground_mask(Path(p)) for p in files], labels=labels.astype(str),
                    splits=[(s, *mod.stratified_split(labels, s)) for s in SEEDS],
                    deep={"ResNet-18": cache / "resnet18_kimia216.npy"}, leg_cache=cache / "zernike_n32_kimia216.npy")
    if name == "Kimia-99":
        mod = _import(REPO / "dataset_scripts" / "data_kimia99.py", "k99")
        with contextlib.redirect_stdout(io.StringIO()):
            files, labels = mod.scan_kimia99()
        labels = np.asarray(labels)
        cache = base / "results_kimia99_cross_dataset_v2" / "feature_cache"
        return dict(items=[mod.load_foreground_mask(Path(p)) for p in files], labels=labels.astype(str),
                    splits=[(s, *mod.stratified_split(labels, s)) for s in SEEDS],
                    deep={"ResNet-18": cache / "resnet18_kimia99_v2.npy"}, leg_cache=cache / "zernike_n32_kimia99_v2.npy")
    if name == "ETHZ":
        mod = _import(REPO / "dataset_scripts" / "data_ethz.py", "ethz")
        with contextlib.redirect_stdout(io.StringIO()):
            df = mod.load_metadata()
        df = df.reset_index(drop=True)
        splits = []
        for s in SEEDS:
            with contextlib.redirect_stdout(io.StringIO()):
                sp = mod.make_group_safe_split(df, s)
            splits.append((s, np.asarray(sp.train), np.asarray(sp.val), np.asarray(sp.test)))
        cache = mod.OUTPUT_ROOT / "feature_cache"
        return dict(items=[mod.read_binary_mask_for_zernike(Path(p)) for p in df["mask_path_obj"]],
                    labels=np.array([str(x) for x in df["class"]]), splits=splits,
                    deep={"ResNet-18": cache / "resnet18_deep.npy", "ViT-B/16": cache / "vit_b_16_deep.npy"},
                    leg_cache=cache / "zernike_MPEG7_exact_N32_G128.npy")
    raise ValueError(name)
