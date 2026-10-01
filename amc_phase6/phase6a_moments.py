"""Phase 6a: CEN_RG moments for k in {2,...,8} on four datasets, and verification of the
small-support asymptotics   Z_n^m = (n+1)/pi * a_{n,m} * mu_m * rho^(m+2) + O(rho^(m+4)),
rho = 1/k, a_{n,m} = lowest-power coefficient of R_n^m (for m = 1 the centroid makes mu_1 = 0,
so the leading power is rho^5 with the next coefficient).
Spyder: F5.  Writes <base>/results_amc_phase6/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, math, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zernike_amc as za
from phase5_data import load_dataset
from phase6a_worker import K_EXT, K_FB, work, fb_work

WORKERS = None
DATASETS = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]
K_FIT = [4.0, 5.0, 6.0, 8.0]             # asymptotic regime used for the slope fit
M_SHOW = [0, 1, 2, 3, 4, 6, 8]

def coef(n, m, s):
    return (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + m) // 2 - s)
                                               * math.factorial((n - m) // 2 - s))

def predicted_profile(m, ns):
    """|Z_n^m| for fixed m, up to a common factor, as rho -> 0."""
    if m == 1:     # r^1 term vanishes (centroid); leading term is the r^3 coefficient
        return np.array([(n + 1) * abs(coef(n, 1, (n - 1) // 2 - 1)) for n in ns], float)
    return np.array([(n + 1) * math.comb((n + m) // 2, m) for n in ns], float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase6"; out.mkdir(parents=True, exist_ok=True)
    pairs = np.array(za.PAIRS); Pn, Pm = pairs[:, 0], pairs[:, 1]
    slope_rows, colin = [], {}
    for ds in a.datasets.split(","):
        t0 = time.time()
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        n = len(D["labels"]); per = out / f"per_image_{ds}"; per.mkdir(exist_ok=True)
        tasks = [(i, D["items"][i], str(per)) for i in range(n) if not (per / f"{i:04d}.npz").exists()]
        print(f"[{ds}] {n} shapes, {len(tasks)} to compute", flush=True)
        if tasks:
            with Pool(a.workers) as pool:
                for c, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                    if c % 100 == 0 or c == len(tasks):
                        print(f"   {c}/{len(tasks)}  {(time.time()-t0)/60:.1f} min", flush=True)
        Z = {k: np.stack([np.load(per / f"{i:04d}.npz")[f"k{k:g}"] for i in range(n)]) for k in K_EXT}
        np.savez(out / f"moments_{ds}.npz", labels=np.array([str(x) for x in D["labels"]]), pairs=pairs,
                 k_ext=np.array(K_EXT), **{f"k{k:g}": v for k, v in Z.items()})
        # ---- decay exponents: slope of log|Z_n^m| against log(rho), rho = 1/k
        x = np.log(1.0 / np.array(K_FIT)); xc = x - x.mean()
        Y = np.log(np.stack([np.abs(Z[k]) for k in K_FIT]) + 1e-300)          # K x shapes x pairs
        sl = np.tensordot(xc, Y - Y.mean(0), axes=(0, 0)) / (xc ** 2).sum()    # shapes x pairs
        for p, (nn, mm) in enumerate(za.PAIRS):
            if (nn, mm) == (1, 1):
                continue
            pred = 5 if mm == 1 else mm + 2
            q = np.percentile(sl[:, p], [25, 50, 75])
            slope_rows.append((ds, nn, mm, pred, *q))
        # ---- collinearity of the n-profile with the predicted leading-order profile
        for mm in M_SHOW:
            ns = [nn for nn in range(mm, 33, 2) if not (mm == 1 and nn < 3)]
            sel = [za.PAIRS.index((nn, mm)) for nn in ns]
            qv = predicted_profile(mm, ns); qv = qv / np.linalg.norm(qv)
            vals = []
            for k in K_EXT:
                V = np.abs(Z[k][:, sel]); V = V / np.linalg.norm(V, axis=1, keepdims=True)
                vals.append(np.median(1 - V @ qv))
            colin[(ds, mm)] = vals
        print(f"[{ds}] done ({(time.time()-t0)/60:.1f} min)")
    # ---- Mehler-Heine / Fourier-Bessel check (MPEG-7 subset):
    #      Z_n^m(k) ~ (n+1)/(pi k^2) * (-1)^((n-m)/2) * B_m((n+1)/k),
    #      B_m(nu) = int J_m(nu s) e^{-i m theta} f(s) ds   (s in radius-of-gyration units)
    if "MPEG-7" in a.datasets.split(","):
        D = load_dataset("MPEG-7", base); idx = list(range(0, len(D["labels"]), 7))
        tasks = [(i, D["items"][i]) for i in idx]
        print(f"[FB check] {len(tasks)} MPEG-7 shapes", flush=True)
        with Pool(a.workers) as pool:
            FB = dict(pool.imap_unordered(fb_work, tasks, chunksize=1))
        Zm = np.load(out / "moments_MPEG-7.npz")
        rows = []
        for k in K_FB:
            dz = np.abs(Zm[f"k{k:g}"][idx]); dz /= np.linalg.norm(dz, axis=1, keepdims=True)
            df = np.stack([FB[i][f"k{k:g}"] for i in idx]); df /= np.linalg.norm(df, axis=1, keepdims=True)
            e = np.linalg.norm(dz - df, axis=1)
            eo = [np.median(np.linalg.norm(dz[:, Pn == nn] - df[:, Pn == nn], axis=1) /
                            np.maximum(np.linalg.norm(dz[:, Pn == nn], axis=1), 1e-300)) for nn in range(33)]
            rows.append((k, np.median(e), np.percentile(e, 90), eo))
        with open(out / "fourier_bessel_check.csv", "w") as f:
            f.write("k,median_descriptor_diff,p90," + ",".join(f"n{nn}" for nn in range(33)) + "\n")
            f.writelines(f"{r[0]:g},{r[1]:.4f},{r[2]:.4f}," + ",".join(f"{x:.4f}" for x in r[3]) + "\n" for r in rows)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for r in rows:
            ax.semilogy(range(33), r[3], marker="o", ms=2.5, label=f"k={r[0]:g} (descriptor diff {r[1]:.3f})")
        ax.set_xlabel("radial order n"); ax.set_ylabel("relative block difference |Z| vs Fourier-Bessel")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_fourier_bessel_check.png", dpi=200); plt.close(fig)
        print(open(out / "fourier_bessel_check.csv").read().split("\n")[0:len(rows)+1])
    with open(out / "decay_slopes.csv", "w") as f:
        f.write("dataset,n,m,predicted_slope,q25,median,q75\n")
        f.writelines(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n" for r in slope_rows)
    dsets = sorted({r[0] for r in slope_rows}, key=DATASETS.index)
    fig, axs = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.6), squeeze=False, sharey=True)
    for ax, ds in zip(axs[0], dsets):
        rr = [r for r in slope_rows if r[0] == ds]
        sc = ax.scatter([r[2] + (r[1] - r[2]) * 0.012 for r in rr], [r[5] for r in rr], c=[r[1] for r in rr], s=8, cmap="viridis")
        ms = np.arange(0, 33); ax.plot(ms, ms + 2, "r--", lw=1, label="predicted m+2")
        ax.plot([1], [5], "r*", ms=9, label="predicted m=1: 5")
        ax.set_title(f"{ds}: fitted exponent of |Z_n^m| vs rho", fontsize=8); ax.set_xlabel("m (points coloured by n)")
    axs[0][0].set_ylabel("median slope d log|Z| / d log rho"); axs[0][0].legend(fontsize=7)
    fig.colorbar(sc, ax=axs[0][-1], label="n")
    fig.savefig(out / "fig_decay_slopes.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.6), squeeze=False, sharey=True)
    rho = 1.0 / np.array(K_EXT)
    for ax, ds in zip(axs[0], dsets):
        for mm in M_SHOW:
            ax.loglog(rho, np.maximum(colin[(ds, mm)], 1e-12), marker="o", ms=3, label=f"m={mm}")
        ax.loglog(rho, 0.5 * (rho / rho[0]) ** 2 * max(colin[(ds, 0)][0], 1e-6), "k--", lw=1, label="slope 2")
        ax.set_title(f"{ds}: 1 - cos(n-profile, leading-order profile)", fontsize=8); ax.set_xlabel("rho = 1/k")
    axs[0][0].set_ylabel("median over shapes"); axs[0][-1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_profile_collapse.png", dpi=200); plt.close(fig)
    with open(out / "profile_collapse.csv", "w") as f:
        f.write("dataset,m," + ",".join(f"k{k:g}" for k in K_EXT) + "\n")
        f.writelines(f"{ds},{mm}," + ",".join(f"{v:.3e}" for v in vals) + "\n" for (ds, mm), vals in colin.items())
    print(f"outputs in {out}")

if __name__ == "__main__":
    main()
