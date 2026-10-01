"""Worker for phase5a_moments.py."""
import os
from pathlib import Path
import numpy as np
import zernike_amc as za
from phase5_data import K_GRID

NAMES = ["LEG_128", "CEN_MAX"] + [f"CEN_RG_k{k:g}" for k in K_GRID]

def work(task):
    i, item, outdir = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    mask = za.load_foreground_mask(Path(item)) if isinstance(item, str) else np.asarray(item)
    cv = za.legacy_canvas(mask)
    ii, jj = np.nonzero(cv)
    out = {"LEG_128": za.grid_moments(cv, "inscribed", 128, "nearest", "endpoint")}
    out["CEN_MAX"], _ = za.exact_grid_moments(cv, "CEN_MAX", 256)
    outside = []
    for k in K_GRID:
        Z, (cx, cy, R) = za.exact_grid_moments(cv, "CEN_RG", 256, k)
        out[f"CEN_RG_k{k:g}"] = Z
        outside.append(float(np.mean(np.hypot(jj + .5 - cx, ii + .5 - cy) > R)))
    out["outside"] = np.array(outside)
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, final)
    return i
