"""Phase 7e: is the weakness of the classical area scale caused by clipping?
(1) CEN_AREA moments for k = 5, 6, 8, 10, 12 and clipped-mass statistics for CEN_AREA and CEN_RG;
(2) evaluation of CEN_AREA with k selected from {2, ..., 12} against CEN_RG (k from {2, ..., 4});
(3) run time and accuracy of batched Kintner vs batched Chebyshev.
Needs results_amc_phase7/moments_<dataset>.npz from phase 7a.  Spyder: F5."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from phase5_data import load_dataset
from phase7e_worker import K_ALL, K_EXTRA, work_moments, run_eval
try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None

WORKERS = None
DATASETS = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]

def paired(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d); m = d.mean(); s = d.std(ddof=1)
    h = (2.262 if n == 10 else 1.96) * s / np.sqrt(n)
    p = ttest_rel(a, b).pvalue if (ttest_rel is not None and s > 0) else np.nan
    return m, m - h, m + h, int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum()), p

def timing(out):
    import zernike_methods as zm, mpmath as mp
    from phase7c_numerics import mp_reference
    os.environ["OMP_NUM_THREADS"] = "1"
    rng = np.random.default_rng(0); rr = np.sqrt(rng.random(300)) * 0.999; tt = rng.random(300) * 2 * np.pi
    xs, ys, ws = rr * np.cos(tt), rr * np.sin(tt), rng.random(300)
    rows = ["N,G,points,method,seconds,max_rel_moment_error"]
    for N in (32, 48, 64):
        refR = mp_reference(N, rr); Zref = np.zeros(len(zm.pairs(N)), complex)
        for p, (n, m) in enumerate(zm.pairs(N)):
            Zref[p] = (n + 1) / np.pi * np.sum(refR[(n, m)] * ws * np.exp(-1j * m * tt))
        sc = np.abs(Zref).max()
        acc = {"batched Kintner": np.abs(zm.moments_kintner_batched(N, xs, ys, ws) - Zref).max() / sc,
               "batched Chebyshev": np.abs(zm.moments_chebyshev_batched(N, xs, ys, ws) - Zref).max() / sc}
        zm.cheb_coeffs(N)
        for G in (128, 256, 512):
            c = -1 + (2 * np.arange(G) + 1) / G; X, Y = np.meshgrid(c, c); k = X * X + Y * Y <= 1
            x, y = X[k], Y[k]; w = np.ones_like(x)
            for name, f in (("batched Kintner", zm.moments_kintner_batched), ("batched Chebyshev", zm.moments_chebyshev_batched)):
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); f(N, x, y, w); ts.append(time.perf_counter() - t0)
                rows.append(f"{N},{G},{x.size},{name},{min(ts):.4f},{acc[name]:.2e}")
    (out / "batched_runtime.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase7"; t0 = time.time()
    clip_rows = ["dataset,family," + ",".join(f"k{k:g}_mean" for k in K_ALL) + "," + ",".join(f"k{k:g}_max" for k in K_ALL)
                 + ",rg_over_ra_median,rg_over_ra_max"]
    tasks = []
    for ds in a.datasets.split(","):
        f7 = out / f"moments_{ds}.npz"
        if not f7.exists():
            print(f"[{ds}] run phase7a first"); continue
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        n = len(D["labels"]); per = out / f"per_image_7e_{ds}"; per.mkdir(exist_ok=True)
        todo = [(i, D["items"][i], str(per)) for i in range(n) if not (per / f"{i:04d}.npz").exists()]
        print(f"[{ds}] {len(todo)} shapes to compute", flush=True)
        if todo:
            with Pool(a.workers) as pool:
                for c, _ in enumerate(pool.imap_unordered(work_moments, todo, chunksize=2), 1):
                    if c % 200 == 0 or c == len(todo):
                        print(f"   {c}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)
        E = [np.load(per / f"{i:04d}.npz") for i in range(n)]
        clip = np.stack([e["clip"] for e in E]); ratio = np.array([e["rg_over_ra"][0] for e in E])
        for j, fam in enumerate(("CEN_AREA", "CEN_RG")):
            blk = clip[:, j * len(K_ALL):(j + 1) * len(K_ALL)]
            clip_rows.append(f"{ds},{fam}," + ",".join(f"{v:.4f}" for v in blk.mean(0)) + "," +
                             ",".join(f"{v:.4f}" for v in blk.max(0)) + f",{np.median(ratio):.3f},{ratio.max():.3f}")
        M = np.load(f7, allow_pickle=True)
        mags = {k: np.abs(M[k]) for k in M.files if k.startswith(("CEN_AREA_k", "CEN_RG_k"))}
        for k in K_EXTRA:
            mags[f"CEN_AREA_k{k:g}"] = np.abs(np.stack([e[f"CEN_AREA_k{k:g}"] for e in E]))
        mags["LEG"] = np.load(D["leg_cache"]).astype(np.float64)
        labels = np.array([str(x) for x in D["labels"]])
        for bb, p in D["deep"].items():
            if Path(p).exists():
                tasks.append((ds, bb, str(p), mags, labels, D["splits"], M["pairs"][:, 0]))
    (out / "clipping_7e.csv").write_text("\n".join(clip_rows) + "\n"); print("\n".join(clip_rows))
    recs = []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r in pool.imap_unordered(run_eval, tasks):
            recs += r; print(f"  {r[0][0]} / {r[0][1]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    R = {}
    for r in recs:
        R.setdefault(r[:4], {})[r[4]] = r
    combos = sorted({k[:2] for k in R}, key=lambda c: (DATASETS.index(c[0]), c[1]))
    get = lambda ds, bb, p, kind, i=7: np.array([R[(ds, bb, p, kind)][s][i] for s in sorted(R[(ds, bb, p, kind)])])
    rows = ["dataset,backbone,comparison,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_t,sel_CEN_AREA"]
    curve = ["dataset,backbone,kind," + ",".join(f"AREA_k{k:g}" for k in K_ALL) + ",RG_k2,RG_k2.5,RG_k3,RG_k3.5,RG_k4"]
    for ds, bb in combos:
        sel = "/".join(R[(ds, bb, "CEN_AREA_kval", "trunc")][s][9] for s in sorted(R[(ds, bb, "CEN_AREA_kval", "trunc")]))
        for name, A_, B_ in [("CEN_RG kval trunc - CEN_AREA kval(2..12) trunc", ("CEN_RG_kval", "trunc"), ("CEN_AREA_kval", "trunc")),
                             ("CEN_RG kval full - CEN_AREA kval(2..12) full", ("CEN_RG_kval", "full"), ("CEN_AREA_kval", "full")),
                             ("CEN_AREA kval(2..12) trunc - BOX-INS full", ("CEN_AREA_kval", "trunc"), ("LEG", "full"))]:
            s = paired(get(ds, bb, *A_), get(ds, bb, *B_))
            rows.append(f"{ds},{bb},{name},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[3]},{s[4]},{s[5]},{s[6]:.3g},{sel}")
        for kind in ("full", "trunc"):
            curve.append(f"{ds},{bb},{kind}," + ",".join(f"{get(ds,bb,f'CEN_AREA_k{k:g}',kind).mean():.4f}" for k in K_ALL) + "," +
                         ",".join(f"{get(ds,bb,f'CEN_RG_k{k:g}',kind).mean():.4f}" for k in (2, 2.5, 3, 3.5, 4)))
    (out / "phase7e_comparisons.csv").write_text("\n".join(rows) + "\n")
    (out / "phase7e_k_curves.csv").write_text("\n".join(curve) + "\n")
    print("\n".join(rows))
    if not a.skip_timing:
        timing(out)
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
