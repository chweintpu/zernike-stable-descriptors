"""Worker for phase9_stability.py: controlled boundary perturbations of one silhouette."""
import numpy as np
from scipy import ndimage
import zernike_amc as za

RADII = [1, 2, 3, 4, 5]              # erosion / dilation radius in native pixels
BLOBS = [1, 2, 3]                    # radius (native pixels) of a remote blob, Proposition 4.2(b)
NORMS = [("CEN-RG", "CEN_RG", 2.0), ("CEN-AREA", "CEN_AREA", 2.0), ("CEN-MAX", "CEN_MAX", 2.0)]
G = 256

def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]; return x * x + y * y <= r * r

def stats(mask):
    """centroid (pixel-centre coordinates), area, r_g (with the 1/6 unit-square term), R_max about the centroid"""
    ii, jj = np.nonzero(mask); A = ii.size
    cy, cx = ii.mean() + .5, jj.mean() + .5
    rg = np.sqrt(np.mean((jj + .5 - cx) ** 2 + (ii + .5 - cy) ** 2) + 1 / 6)
    dx = np.maximum(np.abs(jj - cx), np.abs(jj + 1 - cx)); dy = np.maximum(np.abs(ii - cy), np.abs(ii + 1 - cy))
    return cx, cy, A, rg, float(np.sqrt(dx * dx + dy * dy).max())

def descriptors(mask):
    cv = za.legacy_canvas(mask); out = {}
    for name, kind, k in NORMS:
        Z, _ = za.exact_grid_moments(cv, kind, G, k)
        d = np.abs(Z); nrm = np.linalg.norm(d)
        out[name] = d / nrm if (np.isfinite(nrm) and nrm > 0) else np.full_like(d, np.nan)   # degenerate case -> NaN
    return out

def work(task):
    i, item, seed = task
    m0 = za.load_foreground_mask(item) if isinstance(item, str) else np.asarray(item)
    pad = 2 * max(m0.shape) + 10
    M = np.pad(m0.astype(bool), pad)
    cx, cy, A, rg, Rmax = stats(M); d0 = descriptors(M)
    rows = []
    def record(kind, param, M1):
        if M1.sum() < 0.1 * A:
            rows.append((i, kind, param) + (np.nan,) * (7 + len(NORMS))); return
        cx1, cy1, A1, rg1, Rmax1 = stats(M1)
        delta = np.logical_xor(M, M1).sum()
        ii, jj = np.nonzero(M | M1)                               # rho_0: disk about c containing both sets
        rho0 = float(np.sqrt(np.maximum((jj + 1 - cx) ** 2, (jj - cx) ** 2) + np.maximum((ii + 1 - cy) ** 2, (ii - cy) ** 2)).max())
        bound = 6 * rho0 ** 2 * delta / (A * rg)                 # Proposition 4.2(a), valid for delta <= A/2
        d1 = descriptors(M1)
        rows.append((i, kind, param, delta / A, float(delta <= A / 2), abs(rg1 - rg) / rg, abs(np.sqrt(A1) - np.sqrt(A)) / np.sqrt(A),
                     abs(Rmax1 - Rmax) / Rmax, np.hypot(cx1 - cx, cy1 - cy) / rg, abs(rg1 - rg) / bound,
                     *[float(np.linalg.norm(d1[n] - d0[n])) for n, _, _ in NORMS]))
    for r in RADII:
        record("erosion", r, ndimage.binary_erosion(M, structure=disk(r)))
        record("dilation", r, ndimage.binary_dilation(M, structure=disk(r)))
    ang = np.random.default_rng(seed + i).uniform(0, 2 * np.pi)
    for b in BLOBS:                                               # blob at distance 2 R_max from the centroid
        M1 = M.copy(); yc, xc = int(round(cy + 2 * Rmax * np.sin(ang))), int(round(cx + 2 * Rmax * np.cos(ang)))
        y, x = np.ogrid[:M.shape[0], :M.shape[1]]
        M1 |= (x - xc) ** 2 + (y - yc) ** 2 <= b * b
        record("remote blob", b, M1)
    return rows
