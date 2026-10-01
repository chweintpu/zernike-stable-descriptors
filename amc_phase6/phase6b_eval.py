"""Phase 6b: retrieval with the extended k grid (validation-selected k, N, omega).
Spyder: F5.  Reads results_amc_phase6/moments_<dataset>.npz. Writes results_amc_phase6/."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase5_data import load_dataset
from phase6b_worker import run
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase6"; t0 = time.time(); tasks = []
    for ds in a.datasets.split(","):
        f = out / f"moments_{ds}.npz"
        if not f.exists():
            print(f"[{ds}] run phase6a first"); continue
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        M = np.load(f, allow_pickle=True)
        assert [str(x) for x in M["labels"]] == [str(x) for x in D["labels"]], f"{ds}: label order mismatch"
        k_ext = [float(k) for k in M["k_ext"]]
        mags = {f"k{k:g}": np.abs(M[f"k{k:g}"]) for k in k_ext}
        mags["LEG"] = np.load(D["leg_cache"]).astype(np.float64)
        for bb, p in D["deep"].items():
            if Path(p).exists():
                tasks.append((ds, bb, str(p), mags, D["labels"], D["splits"], M["pairs"][:, 0], k_ext))
    print(f"{len(tasks)} jobs", flush=True)
    recs, surf = [], []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r, sf in pool.imap_unordered(run, tasks):
            recs += r; surf += sf; print(f"  {r[0][0]} / {r[0][1]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "phase6_records.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,seed,deep_mAP,zernike_mAP,fused_mAP,omega,param\n")
        f.writelines(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.6f}" for x in r[5:9]) + f",{r[9]}\n" for r in recs)
    R = {}
    for r in recs:
        R.setdefault(r[:4], {})[r[4]] = r
    combos = sorted({k[:2] for k in R}, key=lambda c: (DATASETS.index(c[0]), c[1]))
    get = lambda ds, bb, p, kind, i=7: np.array([R[(ds, bb, p, kind)][s][i] for s in sorted(R[(ds, bb, p, kind)])])
    k_ext = sorted({float(p[1:]) for (_, _, p, _) in R if p.startswith("k") and p[1].isdigit()})
    with open(out / "k_curve.csv", "w") as f:
        f.write("dataset,backbone,kind,measure," + ",".join(f"k{k:g}" for k in k_ext) + ",LEG\n")
        for ds, bb in combos:
            for kind in ("full", "trunc"):
                for meas, i in (("fused_mAP", 7), ("zernike_mAP", 6)):
                    f.write(f"{ds},{bb},{kind},{meas}," + ",".join(f"{get(ds,bb,f'k{k:g}',kind,i).mean():.5f}" for k in k_ext)
                            + f",{get(ds,bb,'LEG',kind,i).mean():.5f}\n")
    with open(out / "phase6_comparisons.csv", "w") as f:
        f.write("dataset,backbone,comparison,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_t,selected_k_ext\n")
        for ds, bb in combos:
            sel = "/".join(R[(ds, bb, "kval_ext", "trunc")][s][9] for s in sorted(R[(ds, bb, "kval_ext", "trunc")]))
            for name, A_, B_ in [("kval_ext trunc - kval_le3 trunc", ("kval_ext", "trunc"), ("kval_le3", "trunc")),
                                 ("kval_ext full - kval_le3 full", ("kval_ext", "full"), ("kval_le3", "full")),
                                 ("kval_ext trunc - LEG full (baseline)", ("kval_ext", "trunc"), ("LEG", "full")),
                                 ("kval_ext full - kval_ext trunc", ("kval_ext", "full"), ("kval_ext", "trunc"))]:
                s = paired(get(ds, bb, *A_), get(ds, bb, *B_))
                f.write(f"{ds},{bb},{name},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[3]},{s[4]},{s[5]},{s[6]:.3g},{sel}\n")
    # ---- (k, N) surface and the effective bandwidth (N+1)/k  (Mehler-Heine: order n at scale k
    #      probes the Fourier-Bessel spectrum of the shape at radial frequency (n+1)/k)
    S = {}
    for (ds, bb, seed, p, N, fu, z, v) in surf:
        S.setdefault((ds, bb, float(p[1:]), N), []).append(fu)
    with open(out / "kN_surface.csv", "w") as f:
        f.write("dataset,backbone,k,N,bandwidth_(N+1)/k,fused_mAP_mean\n")
        f.writelines(f"{ds},{bb},{k:g},{N},{(N+1)/k:.3f},{np.mean(v):.5f}\n" for (ds, bb, k, N), v in sorted(S.items()))
    bw_rows = []
    for ds, bb in combos:
        sel = [R[(ds, bb, "kval_ext", "trunc")][s][9] for s in sorted(R[(ds, bb, "kval_ext", "trunc")])]
        bws = [(int(x.split("/")[1]) + 1) / float(x.split("/")[0][1:]) for x in sel]
        bw_rows.append((ds, bb, np.median(bws), np.min(bws), np.max(bws)))
    with open(out / "selected_bandwidth.csv", "w") as f:
        f.write("dataset,backbone,median_(N+1)/k,min,max\n")
        f.writelines(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f}\n" for r in bw_rows)
    fig, axs = plt.subplots(2, 5, figsize=(21, 7.5), squeeze=False)
    for ax, (ds, bb) in zip(axs.ravel(), combos):
        ks = k_ext; Ns = sorted({N for (d, b, k, N) in S if d == ds and b == bb})
        H = np.array([[np.mean(S[(ds, bb, k, N)]) for k in ks] for N in Ns])
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[-.5, len(ks) - .5, Ns[0] - .5, Ns[-1] + .5])
        for bwv in (6, 8, 10):
            ax.plot(range(len(ks)), [bwv * k - 1 for k in ks], "w--", lw=.8)
        ax.set_ylim(Ns[0] - .5, Ns[-1] + .5)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks], fontsize=7)
        ax.set_title(f"{ds} / {bb}", fontsize=8); ax.set_xlabel("k"); ax.set_ylabel("N")
        fig.colorbar(im, ax=ax, fraction=.046)
    for ax in axs.ravel()[len(combos):]:
        ax.axis("off")
    fig.suptitle("test fused mAP over (k, N); white dashed: (N+1)/k = 6, 8, 10", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "fig_kN_surface.png", dpi=180); plt.close(fig)
    print(open(out / "selected_bandwidth.csv").read())
    fig, axs = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    for j, ds in enumerate(DATASETS):
        for bb in [c[1] for c in combos if c[0] == ds]:
            for row, (meas, i) in enumerate((("fused mAP", 7), ("Zernike-only mAP", 6))):
                ax = axs[row, j]
                l = ax.plot(k_ext, [get(ds, bb, f"k{k:g}", "trunc", i).mean() for k in k_ext], marker="o", label=f"{bb} trunc")[0]
                ax.plot(k_ext, [get(ds, bb, f"k{k:g}", "full", i).mean() for k in k_ext], ls="--", marker=".", color=l.get_color())
                ax.axhline(get(ds, bb, "LEG", "full", i).mean(), color=l.get_color(), ls=":", lw=1)
                ax.set_title(f"{ds}: {meas} (dashed = no truncation, dotted = baseline)", fontsize=8); ax.set_xlabel("k")
        axs[0, j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_k_extended.png", dpi=200); plt.close(fig)
    print(open(out / "phase6_comparisons.csv").read()); print(f"done in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
