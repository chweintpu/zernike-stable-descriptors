"""Phase 5b: cross-dataset validation of normalisation + truncation, k sensitivity,
angular (m) redundancy. Spyder: F5.  Terminal: python phase5b_eval.py [--workers 12]
Reads <base>/results_amc_phase5/moments_<dataset>.npz (phase5a) and baseline caches.
"""
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
from phase5_data import load_dataset, K_GRID, SEEDS
from phase5b_worker import run
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
    base = Path(a.base); out = base / "results_amc_phase5"
    t0 = time.time(); tasks = []
    for ds in a.datasets.split(","):
        f = out / f"moments_{ds}.npz"
        if not f.exists():
            print(f"[{ds}] moments file missing - run phase5a first"); continue
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        M = np.load(f, allow_pickle=True)   # our own file; labels may be stored as an object array
        Pn, Pm = M["pairs"][:, 0], M["pairs"][:, 1]
        assert [str(x) for x in M["labels"]] == [str(x) for x in D["labels"]], f"{ds}: label order mismatch"
        mags = {f"CEN_RG_k{k:g}": np.abs(M[f"CEN_RG_k{k:g}"]) for k in K_GRID}
        mags["CEN_MAX"] = np.abs(M["CEN_MAX"])
        leg = np.load(D["leg_cache"]).astype(np.float64) if Path(D["leg_cache"]).exists() else None
        mags["LEG"] = leg if (leg is not None and leg.shape == M["LEG_128"].shape) else np.abs(M["LEG_128"])
        for bb, p in D["deep"].items():
            if Path(p).exists():
                tasks.append((ds, bb, str(p), mags, D["labels"], D["splits"], Pn, Pm, K_GRID))
            else:
                print(f"[{ds}] Deep cache not found for {bb}: {p}")
    print(f"{len(tasks)} (dataset, backbone) jobs, workers={min(a.workers, len(tasks))}", flush=True)
    recs, ckas = [], []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r, c in pool.imap_unordered(run, tasks):
            recs += r; ckas += c
            print(f"  {r[0][0]} / {r[0][1]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "phase5_records.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,seed,deep_mAP,zernike_mAP,fused_mAP,gain,omega,param\n")
        for r in recs:
            f.write(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.6f}" for x in r[5:10]) + f",{r[10]}\n")
    with open(out / "cka_by_m.csv", "w") as f:
        f.write("dataset,backbone,pipeline,part,linear_CKA\n")
        f.writelines(",".join(map(str, r[:4])) + f",{r[4]:.6f}\n" for r in ckas)

    R = {}
    for r in recs:
        R.setdefault(r[:4], {})[r[4]] = r
    combos = sorted({k[:2] for k in R})
    get = lambda ds, bb, p, kind, i=7: np.array([R[(ds, bb, p, kind)][s][i] for s in sorted(R[(ds, bb, p, kind)])])
    # --- baseline summary
    lines = []
    for ds, bb in combos:
        if bb == ("ResNet-18"):
            lines.append(f"[{ds} / {bb}] LEG full: Deep {get(ds,bb,'LEG','full',5).mean():.4f}, Zernike "
                         f"{get(ds,bb,'LEG','full',6).mean():.4f}, Fixed {get(ds,bb,'LEG','full',7).mean():.4f}")
    # --- summary + paired stats
    methods = [("LEG", "full"), ("LEG", "trunc"), ("CEN_MAX", "full"), ("CEN_MAX", "trunc"),
               ("CEN_RG_k2", "full"), ("CEN_RG_k2", "trunc"), ("CEN_RG_kval", "full"), ("CEN_RG_kval", "trunc")]
    with open(out / "phase5_summary.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,deep_mAP,zernike_mAP,fused_mAP,gain,delta_vs_LEG_full,ci_lo,ci_hi,wins,ties,losses,p_t,selected_params\n")
        for ds, bb in combos:
            base_f = get(ds, bb, "LEG", "full")
            for p, kind in methods:
                v = get(ds, bb, p, kind); s = paired(v, base_f)
                prm = "/".join(R[(ds, bb, p, kind)][x][10] for x in sorted(R[(ds, bb, p, kind)]))
                f.write(f"{ds},{bb},{p},{kind},{get(ds,bb,p,kind,5).mean():.5f},{get(ds,bb,p,kind,6).mean():.5f},"
                        f"{v.mean():.5f},{get(ds,bb,p,kind,8).mean():.5f},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},"
                        f"{s[3]},{s[4]},{s[5]},{s[6]:.3g},{prm}\n")
    with open(out / "phase5_key_comparisons.csv", "w") as f:
        f.write("dataset,backbone,comparison,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_t\n")
        for ds, bb in combos:
            for name, A_, B_ in [("CEN_RG_kval trunc - LEG full", ("CEN_RG_kval", "trunc"), ("LEG", "full")),
                                 ("CEN_RG_kval trunc - LEG trunc", ("CEN_RG_kval", "trunc"), ("LEG", "trunc")),
                                 ("CEN_RG_k2 full - LEG full", ("CEN_RG_k2", "full"), ("LEG", "full")),
                                 ("CEN_RG_k2 full - CEN_MAX full", ("CEN_RG_k2", "full"), ("CEN_MAX", "full")),
                                 ("LEG trunc - LEG full", ("LEG", "trunc"), ("LEG", "full")),
                                 ("CEN_RG_kval full - CEN_RG_k2 full", ("CEN_RG_kval", "full"), ("CEN_RG_k2", "full"))]:
                s = paired(get(ds, bb, *A_), get(ds, bb, *B_))
                f.write(f"{ds},{bb},{name},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[3]},{s[4]},{s[5]},{s[6]:.3g}\n")
    # --- k sensitivity table + figure
    with open(out / "k_sensitivity.csv", "w") as f:
        f.write("dataset,backbone,kind," + ",".join(f"k{k:g}" for k in sorted(K_GRID)) + ",LEG\n")
        for ds, bb in combos:
            for kind in ("full", "trunc"):
                f.write(f"{ds},{bb},{kind}," + ",".join(f"{get(ds,bb,f'CEN_RG_k{k:g}',kind).mean():.5f}" for k in sorted(K_GRID))
                        + f",{get(ds,bb,'LEG',kind).mean():.5f}\n")
    dsets = sorted({c[0] for c in combos}, key=lambda d: DATASETS.index(d))
    fig, axs = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.6), squeeze=False)
    for ax, ds in zip(axs[0], dsets):
        for bb in [c[1] for c in combos if c[0] == ds]:
            ks = sorted(K_GRID)
            l = ax.plot(ks, [get(ds, bb, f"CEN_RG_k{k:g}", "trunc").mean() for k in ks], marker="o", label=f"{bb} (trunc)")[0]
            ax.plot(ks, [get(ds, bb, f"CEN_RG_k{k:g}", "full").mean() for k in ks], ls="--", marker=".", color=l.get_color())
            ax.axhline(get(ds, bb, "LEG", "full").mean(), color=l.get_color(), ls=":", lw=1)
        ax.set_title(f"{ds}: fused mAP vs k  (dotted = baseline descriptor)", fontsize=8); ax.set_xlabel("k (R = k * r_g)")
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_k_sensitivity.png", dpi=200); plt.close(fig)
    # --- cross-dataset gains figure
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(combos)), 4))
    x = np.arange(len(combos)); wdt = 0.8 / len(methods)
    for j, (p, kind) in enumerate(methods):
        ax.bar(x + (j - len(methods) / 2 + .5) * wdt, [(get(ds, bb, p, kind) - get(ds, bb, "LEG", "full")).mean() for ds, bb in combos],
               wdt, label=f"{p} {kind}")
    ax.axhline(0, color="gray", lw=.6); ax.set_xticks(x); ax.set_xticklabels([f"{ds}\n{bb}" for ds, bb in combos], fontsize=7)
    ax.set_ylabel("fused mAP minus baseline fixed fusion"); ax.legend(fontsize=6, ncol=4)
    fig.tight_layout(); fig.savefig(out / "fig_cross_dataset_gains.png", dpi=200); plt.close(fig)
    # --- CKA by m
    ck = {}
    for r in ckas:
        ck.setdefault(r[:3], {})[r[3]] = r[4]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, p in zip(axs, ("LEG", "CEN_RG_k2")):
        for ds, bb in combos:
            d = ck[(ds, bb, p)]; ms = list(range(33))
            ax.plot(ms, [d[f"m={m}"] for m in ms], marker="o", ms=2, lw=1, label=f"{ds}/{bb}")
        ax.set_title(f"{p}: CKA(Deep, all components with angular index m)", fontsize=9); ax.set_xlabel("m")
    axs[0].set_ylabel("linear CKA"); axs[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig_cka_by_m.png", dpi=200); plt.close(fig)
    (out / "phase5b_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
