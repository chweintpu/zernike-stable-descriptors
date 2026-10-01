"""Phase 5a: Zernike moments (legacy, CEN_MAX, CEN_RG for several k) on MPEG-7, Kimia-216,
Kimia-99 and ETHZ. Spyder: F5.  Terminal: python phase5a_moments.py [--workers 12]
Writes <base>/results_amc_phase5/moments_<dataset>.npz
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import zernike_amc as za
from phase5_data import load_dataset, K_GRID
from phase5a_worker import NAMES, work

WORKERS = None
DATASETS = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]

def desc(C):
    d = np.abs(C); return d / np.linalg.norm(d, axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase5"; out.mkdir(parents=True, exist_ok=True)
    report = []
    for ds in a.datasets.split(","):
        t0 = time.time()
        try:
            D = load_dataset(ds, base)
        except Exception:
            msg = f"[{ds}] LOADING FAILED -- please send me this message:\n{traceback.format_exc()}"
            print(msg); report.append(msg); continue
        n = len(D["labels"]); per = out / f"per_image_{ds}"; per.mkdir(exist_ok=True)
        tasks = [(i, D["items"][i], str(per)) for i in range(n) if not (per / f"{i:04d}.npz").exists()]
        print(f"[{ds}] {n} shapes, {len(tasks)} to compute, workers={a.workers}", flush=True)
        if tasks:
            with Pool(a.workers) as pool:
                for k, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                    if k % 100 == 0 or k == len(tasks):
                        print(f"   {k}/{len(tasks)}  {(time.time()-t0)/60:.1f} min", flush=True)
        Z = {k: np.stack([np.load(per / f"{i:04d}.npz")[k] for i in range(n)]) for k in NAMES}
        outside = np.stack([np.load(per / f"{i:04d}.npz")["outside"] for i in range(n)])
        np.savez(out / f"moments_{ds}.npz", labels=np.array([str(x) for x in D["labels"]]), pairs=np.array(za.PAIRS), outside=outside,
                 k_grid=np.array(K_GRID), **Z)
        # consistency with the baseline Zernike cache
        line = f"[{ds}] n={n}"
        if Path(D["leg_cache"]).exists():
            cache = np.load(D["leg_cache"])
            diff = float(np.abs(desc(Z["LEG_128"]) - cache).max()) if cache.shape == Z["LEG_128"].shape else float("nan")
            line += f"  legacy recompute vs baseline cache: max diff {diff:.2e} ({'OK' if diff < 1e-4 else 'CHECK'})"
        line += "  | foreground outside disk (mean/max) " + ", ".join(
            f"k={k:g}: {outside[:, j].mean():.4f}/{outside[:, j].max():.3f}" for j, k in enumerate(K_GRID))
        print(line); report.append(line)
    (out / "phase5a_report.txt").write_text("\n".join(report) + "\n")
    print(f"outputs in {out}")

if __name__ == "__main__":
    main()
