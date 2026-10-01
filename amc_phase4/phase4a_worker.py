"""Worker for phase4a_moments.py (importable -> multiprocessing works from Spyder)."""
import os
from pathlib import Path
import numpy as np
import zernike_amc as za

K_RG = 2.0
ORIG = [("LEG_128", None, 128), ("BOX_INS_EX_256", "BOX_INS", 256),
        ("BOX_CIR_EX_128", "BOX_CIR", 128), ("BOX_CIR_EX_256", "BOX_CIR", 256), ("BOX_CIR_EX_512", "BOX_CIR", 512),
        ("CEN_MAX_EX_256", "CEN_MAX", 256), ("CEN_RG_EX_256", "CEN_RG", 256)]
ROT = [("LEG_128", None, 128), ("BOX_INS_EX_256", "BOX_INS", 256), ("BOX_CIR_EX_256", "BOX_CIR", 256),
       ("CEN_MAX_EX_256", "CEN_MAX", 256), ("CEN_RG_EX_256", "CEN_RG", 256)]
KEYS = [f"{n}|orig" for n, _, _ in ORIG] + [f"{n}|rot" for n, _, _ in ROT]

def angle_of(i):
    return float(np.random.default_rng(12345 + i).uniform(0.0, 360.0))

def _compute(mask, specs, tag, out, extra):
    cv = za.legacy_canvas(mask)
    for name, kind, G in specs:
        if kind is None:
            out[f"{name}|{tag}"] = za.grid_moments(cv, "inscribed", G, "nearest", "endpoint")
        else:
            Z, (cx, cy, R) = za.exact_grid_moments(cv, kind, G, K_RG)
            out[f"{name}|{tag}"] = Z
            if kind == "CEN_RG" and tag == "orig":        # foreground fraction outside the disk
                ii, jj = np.nonzero(cv)
                extra[0] = float(np.mean(np.hypot(jj + .5 - cx, ii + .5 - cy) > R))

def work(task):
    i, path, outdir = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    mask = za.load_foreground_mask(Path(path))
    out, extra = {}, np.zeros(2)
    _compute(mask, ORIG, "orig", out, extra)
    extra[1] = angle_of(i)
    _compute(za.rotate_mask(mask, extra[1]), ROT, "rot", out, extra)
    out["extra"] = extra
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **{k.replace("|", "__"): v for k, v in out.items()})
    os.replace(tmp, final)
    return i
