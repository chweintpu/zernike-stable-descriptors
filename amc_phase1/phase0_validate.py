"""Phase 0: sanity checks before the full run (about 1 minute).
  1. Chebyshev engine vs high-precision evaluation (needs mpmath; skipped if absent)
  2. Legacy configuration reproduces the baseline Zernike cache
  3. Reference quadrature convergence on a few shapes
Usage:  python phase0_validate.py            (run from inside the amc_phase1 folder)
"""
import argparse, time
from pathlib import Path
import numpy as np
import zernike_amc as za

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent),
                help="Zernike project folder (contains data/mpeg7)")
args = ap.parse_args()
BASE = Path(args.base)
ok_all = True

# 1 ---------------------------------------------------------------
try:
    import mpmath as mp
    mp.mp.dps = 50
    rng = np.random.default_rng(0)
    r = np.sqrt(rng.random(200)); th = rng.random(200) * 2 * np.pi
    x, y, w = r * np.cos(th), r * np.sin(th), rng.random(200)
    Z = za.moments(x, y, w)
    def zmp(n, m):
        s = mp.mpf(0)
        for xi, yi, wi in zip(x, y, w):
            rr = mp.sqrt(mp.mpf(xi) ** 2 + mp.mpf(yi) ** 2); t = mp.atan2(yi, xi)
            R = sum((-1) ** l * mp.factorial(n - l) / (mp.factorial(l) * mp.factorial((n + m) // 2 - l)
                    * mp.factorial((n - m) // 2 - l)) * rr ** (n - 2 * l) for l in range((n - m) // 2 + 1))
            s += wi * R * mp.e ** (-1j * m * t)
        return complex((n + 1) / mp.pi * s)
    err = max(abs(Z[za.PAIRS.index(p)] - zmp(*p)) for p in [(32, 0), (32, 2), (31, 1), (30, 14), (32, 32)])
    ok = err < 1e-6; ok_all &= ok
    print(f"[1] engine vs 50-digit reference: max abs err = {err:.2e}  {'PASS' if ok else 'FAIL'}")
except ImportError:
    print("[1] mpmath not installed -> skipped (pip install mpmath)")

# 2 ---------------------------------------------------------------
fs = za.list_mpeg7(BASE / "data" / "mpeg7")
print(f"    MPEG-7 files found: {len(fs)} (expected 1400)")
cache_p = BASE / "results_mpeg7_unified_preprocessing" / "feature_cache" / "zernike_n32_mpeg7_unified.npy"
cache = np.load(cache_p)
worst = 0.0
for i in range(0, 1400, 35):
    cv = za.legacy_canvas(za.load_foreground_mask(fs[i]))
    d = za.descriptor(za.grid_moments(cv, "inscribed", 128, "nearest", "endpoint"))
    worst = max(worst, float(np.abs(d - cache[i]).max()))
ok = worst < 1e-5; ok_all &= ok
print(f"[2] legacy config vs baseline cache (40 shapes): max abs diff = {worst:.2e}  {'PASS' if ok else 'FAIL'}")
if not ok:
    print("    -> check file ordering / data folder; features would be misaligned with Deep caches")

# 3 ---------------------------------------------------------------
def blockerr(d, r):
    return max(np.linalg.norm(d[za.P_N == n] - r[za.P_N == n]) / np.linalg.norm(r[za.P_N == n]) for n in range(33))
byname = {f.name: f for f in fs}
worst_c = worst_i = 0.0
t0 = time.time()
for nm in ["lmfish-19.gif", "stef-06.gif", "apple-1.gif"]:
    cv = za.legacy_canvas(za.load_foreground_mask(byname[nm]))
    a = za.descriptor(za.reference_moments_split(cv, "circumscribed", 4, 128))
    b = za.descriptor(za.reference_moments_split(cv, "circumscribed", 8, 128))
    worst_c = max(worst_c, blockerr(a, b))
    a = za.descriptor(za.reference_moments_split(cv, "inscribed", 4, 128))
    b = za.descriptor(za.reference_moments_split(cv, "inscribed", 4, 256))
    worst_i = max(worst_i, blockerr(a, b))
ok = worst_c < 1e-5 and worst_i < 1e-3; ok_all &= ok
print(f"[3] reference convergence: circumscribed Q4 vs Q8 = {worst_c:.1e}; "
      f"inscribed Qb128 vs Qb256 = {worst_i:.1e}  {'PASS' if ok else 'CHECK'}  ({time.time()-t0:.0f}s)")
print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED - please send me the output")
