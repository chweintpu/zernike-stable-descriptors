"""Worker for phase6a_moments.py: CEN_RG moments for an extended k grid.
The grid size grows with k (G = 128 k) so that the shape itself is always sampled
with the same density as at k = 2, G = 256 (phase 5)."""
import os
from pathlib import Path
import numpy as np
import zernike_amc as za

K_EXT = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
G_OF = {k: int(round(128 * k)) for k in K_EXT}

def work(task):
    i, item, outdir = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    mask = za.load_foreground_mask(Path(item)) if isinstance(item, str) else np.asarray(item)
    cv = za.legacy_canvas(mask)
    out = {}
    for k in K_EXT:
        out[f"k{k:g}"], _ = za.exact_grid_moments(cv, "CEN_RG", G_OF[k], k)
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **out); os.replace(tmp, final)
    return i

K_FB = [3.0, 4.0, 6.0, 8.0]

def fb_work(task):
    """Predicted |Z_n^m| from the Fourier-Bessel transform of the shape (pixel centres,
    coordinates in radius-of-gyration units, same centroid as CEN_RG)."""
    from scipy.special import jv
    i, item = task
    mask = za.load_foreground_mask(Path(item)) if isinstance(item, str) else np.asarray(item)
    cv = za.legacy_canvas(mask)
    cx, cy, R = za.general_mapping(cv, "CEN_RG", 2.0); rg = R / 2.0
    ii, jj = np.nonzero(cv)
    x, y = (jj + .5 - cx) / rg, (ii + .5 - cy) / rg
    s, th = np.hypot(x, y), np.arctan2(y, x); dA = 1.0 / rg ** 2
    out = {}
    for k in K_FB:
        v = np.zeros(len(za.PAIRS))
        for p, (n, m) in enumerate(za.PAIRS):
            v[p] = abs(np.sum(jv(m, (n + 1) / k * s) * np.exp(-1j * m * th)) * dA) * (n + 1) / (np.pi * k * k)
        out[f"k{k:g}"] = v
    return i, out
