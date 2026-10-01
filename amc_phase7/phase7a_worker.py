"""Worker for phase7a_moments.py."""
import os
from pathlib import Path
import numpy as np
import zernike_amc as za

K_LIST = [2.0, 2.5, 3.0, 3.5, 4.0]
G_OF = {k: int(round(128 * k)) for k in K_LIST}
NAMES = ["CEN_MAX"] + [f"{f}_k{k:g}" for f in ("CEN_AREA", "CEN_RG") for k in K_LIST]
ROT_NAMES = ["LEG_128", "CEN_MAX", "CEN_AREA_k2", "CEN_RG_k2"]

def angle_of(i):                       # identical to phase 4
    return float(np.random.default_rng(12345 + i).uniform(0.0, 360.0))

def _one(cv, name):
    if name == "LEG_128":
        return za.grid_moments(cv, "inscribed", 128, "nearest", "endpoint")
    if name == "CEN_MAX":
        return za.exact_grid_moments(cv, "CEN_MAX", 256)[0]
    fam, k = name.rsplit("_k", 1); k = float(k)
    return za.exact_grid_moments(cv, fam, G_OF[k], k)[0]

def work(task):
    i, item, outdir, with_rot = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    mask = za.load_foreground_mask(Path(item)) if isinstance(item, str) else np.asarray(item)
    cv = za.legacy_canvas(mask)
    out = {n: _one(cv, n) for n in NAMES}
    cx, cy, R = za.general_mapping(cv, "CEN_AREA", 2.0)
    ii, jj = np.nonzero(cv)
    out["outside_area_k2"] = np.array([np.mean(np.hypot(jj + .5 - cx, ii + .5 - cy) > R)])
    if with_rot:
        out["LEG_128"] = _one(cv, "LEG_128")
        cvr = za.legacy_canvas(za.rotate_mask(mask, angle_of(i)))
        for n in ROT_NAMES:
            out[f"ROT_{n}"] = _one(cvr, n)
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, final)
    return i
