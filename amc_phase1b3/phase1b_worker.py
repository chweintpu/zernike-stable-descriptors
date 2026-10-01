"""Worker for phase1b.py (importable module -> multiprocessing works from Spyder)."""
import os
from pathlib import Path
import numpy as np
import zernike_amc as za

# area resampling with finer sub-sampling, to test the G=512 saturation
EXTRA = [("INS_AR_256_ss8", "inscribed", 256, 8), ("INS_AR_512_ss8", "inscribed", 512, 8),
         ("CIR_AR_256_ss8", "circumscribed", 256, 8), ("CIR_AR_512_ss8", "circumscribed", 512, 8)]
EXTRA_NAMES = [e[0] for e in EXTRA]

def work(task):
    i, path, outdir = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    cv = za.legacy_canvas(za.load_foreground_mask(Path(path)))
    out = {name: za.grid_moments(cv, kind, G, "area", "center", ss=ss) for name, kind, G, ss in EXTRA}
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, final)
    return i
