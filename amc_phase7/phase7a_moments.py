"""Phase 7a: moments for the classical centroid + area normalization (CEN_AREA) together with
CEN_RG and CEN_MAX on the four datasets; rotated copies on MPEG-7.
Spyder: F5.  Writes <base>/results_amc_phase7/moments_<dataset>.npz"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import zernike_amc as za
from phase5_data import load_dataset
from phase7a_worker import NAMES, ROT_NAMES, work

WORKERS = None
DATASETS = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase7"; out.mkdir(parents=True, exist_ok=True)
    for ds in a.datasets.split(","):
        t0 = time.time()
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        n = len(D["labels"]); per = out / f"per_image_{ds}"; per.mkdir(exist_ok=True)
        rot = ds == "MPEG-7"
        tasks = [(i, D["items"][i], str(per), rot) for i in range(n) if not (per / f"{i:04d}.npz").exists()]
        print(f"[{ds}] {n} shapes, {len(tasks)} to compute", flush=True)
        if tasks:
            with Pool(a.workers) as pool:
                for c, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                    if c % 100 == 0 or c == len(tasks):
                        print(f"   {c}/{len(tasks)}  {(time.time()-t0)/60:.1f} min", flush=True)
        keys = NAMES + ["outside_area_k2"] + (["LEG_128"] + [f"ROT_{r}" for r in ROT_NAMES] if rot else [])
        Z = {k: np.stack([np.load(per / f"{i:04d}.npz")[k] for i in range(n)]) for k in keys}
        np.savez(out / f"moments_{ds}.npz", labels=np.array([str(x) for x in D["labels"]]),
                 pairs=np.array(za.PAIRS), **Z)
        o = Z["outside_area_k2"][:, 0]
        print(f"[{ds}] saved; CEN_AREA k=2 foreground outside disk: mean {o.mean():.4f}, max {o.max():.3f} "
              f"({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
