"""Phase 1 (compute): complex Zernike moments of all 1,400 MPEG-7 shapes for
2 native-resolution references + 15 discrete configurations. Parallel, resumable.
Usage:  python phase1_run.py --workers 6
Output: <base>/results_amc_phase1/moments_all.npz
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")          # one BLAS thread per worker process
import argparse, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import zernike_amc as za

CONFIGS = [(f"LEG_{G}", "inscribed", G, "nearest", "endpoint") for G in (128, 256, 512)]
for kind, K in (("inscribed", "INS"), ("circumscribed", "CIR")):
    for rs, R in (("nearest", "NN"), ("area", "AR")):
        for G in (128, 256, 512):
            CONFIGS.append((f"{K}_{R}_{G}", kind, G, rs, "center"))
NAMES = ["REF_INS", "REF_CIR"] + [c[0] for c in CONFIGS]

def work(task):
    i, path, outdir, Q, Qb = task
    final = Path(outdir) / f"{i:04d}.npz"
    if final.exists():
        return i
    cv = za.legacy_canvas(za.load_foreground_mask(Path(path)))
    out = {"REF_INS": za.reference_moments_split(cv, "inscribed", Q, Qb),
           "REF_CIR": za.reference_moments_split(cv, "circumscribed", Q, Qb)}
    for name, kind, G, rs, smp in CONFIGS:
        out[name] = za.grid_moments(cv, kind, G, rs, smp, ss=(2 if G == 512 else 4))
    S = cv.shape[0]
    _, Rc = za.mapping(cv, "circumscribed")
    ii, jj = np.nonzero(cv)
    rr = np.hypot((jj + .5 - S / 2) / (S / 2), (ii + .5 - S / 2) / (S / 2))
    # outside-disk foreground fraction (inscribed), canvas side, circ/inscribed radius ratio
    out["extra"] = np.array([(rr > 1).mean(), S, Rc / (S / 2)])
    tmp = Path(outdir) / f"{i:04d}_tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, final)                   # atomic: an interrupted file is never "done"
    return i

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=1400, help="for quick tests only")
    ap.add_argument("--Q", type=int, default=4)
    ap.add_argument("--Qb", type=int, default=128)
    a = ap.parse_args()
    base = Path(a.base)
    fs = za.list_mpeg7(base / "data" / "mpeg7")
    assert len(fs) == 1400, f"expected 1400 MPEG-7 images, found {len(fs)}"
    out = base / "results_amc_phase1"; per = out / "per_image"
    per.mkdir(parents=True, exist_ok=True)
    n = min(a.limit, 1400)
    tasks = [(i, str(fs[i]), str(per), a.Q, a.Qb) for i in range(n) if not (per / f"{i:04d}.npz").exists()]
    print(f"{n - len(tasks)} already done, {len(tasks)} to compute, workers = {a.workers}")
    t0 = time.time()
    if tasks:
        with Pool(a.workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                if k % 25 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)}  elapsed {el/60:.1f} min  ETA {el/k*(len(tasks)-k)/60:.1f} min", flush=True)
    D = {k: np.zeros((n, len(za.PAIRS)), np.complex128) for k in NAMES}
    extra = np.zeros((n, 3))
    for i in range(n):
        z = np.load(per / f"{i:04d}.npz")
        for k in NAMES:
            D[k][i] = z[k]
        extra[i] = z["extra"]
    np.savez(out / "moments_all.npz", names=np.array(NAMES),
             files=np.array([f.name for f in fs[:n]]), labels=np.array([za.label_of(f) for f in fs[:n]]),
             pairs=np.array(za.PAIRS), extra=extra, **D)
    print(f"saved {out / 'moments_all.npz'}  (total {(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
