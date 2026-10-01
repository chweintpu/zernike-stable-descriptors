"""Phase 4b: validation-selected truncation / order weighting, m-decomposition,
normalisation comparison and rotation robustness, with paired statistics.
Spyder: F5.   Terminal: python phase4b_select.py [--workers 6]
Reads  <base>/results_amc_phase4/moments_phase4.npz  (from phase4a_moments.py)
Writes <base>/results_amc_phase4/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase4b_worker import run_backbone, stratified_split, ap_mean, l2n, SEEDS, M_GROUPS
try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:
    ttest_rel = wilcoxon = None

WORKERS = None
PIPELINES = ["LEG_128", "BOX_INS_EX_256", "BOX_CIR_EX_256", "CEN_MAX_EX_256", "CEN_RG_EX_256"]
MAIN = "CEN_MAX_EX_256"
BACKBONES = {
    "ResNet-18": "results_mpeg7_multibackbone_zernike/feature_cache/resnet18_mpeg7_unified.npy",
    "ResNet-50": "results_mpeg7_multibackbone_zernike/feature_cache/resnet50_mpeg7_unified.npy",
    "EfficientNet-B0": "results_mpeg7_multibackbone_zernike/feature_cache/efficientnet_b0_mpeg7_unified.npy",
    "ViT-B/16": "results_mpeg7_multibackbone_zernike/feature_cache/vit_b_16_mpeg7_unified.npy",
    "Swin-T": "results_mpeg7_multibackbone_zernike/feature_cache/swin_t_mpeg7_unified.npy",
    "DINOv2": "results_mpeg7_dinov2_zernike_foundation/feature_cache/dinov2_vitb14_mpeg7_224.npy",
}

def paired(a, b):
    """stats of a - b over matched seeds"""
    d = np.asarray(a) - np.asarray(b); n = len(d); m = d.mean(); s = d.std(ddof=1)
    h = 2.262 * s / np.sqrt(n) if n == 10 else 1.96 * s / np.sqrt(n)
    pt = pw = np.nan
    if ttest_rel is not None and s > 0:
        pt = ttest_rel(a, b).pvalue
        try: pw = wilcoxon(a, b).pvalue
        except ValueError: pass
    return m, m - h, m + h, int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum()), pt, pw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase4"
    M = np.load(out / "moments_phase4.npz")
    labels = [str(s) for s in M["labels"]]; pairs = M["pairs"]; Pn, Pm = pairs[:, 0], pairs[:, 1]
    mags = {p: np.abs(M[f"{p}__orig"]) for p in PIPELINES}
    t0 = time.time()

    # ------------------------------------------------ rotation robustness (Zernike only, no Deep)
    lab = np.array(labels); rot_rows = []; rot = {}
    for p in PIPELINES:
        Z0, Zr = l2n(np.abs(M[f"{p}__orig"])), l2n(np.abs(M[f"{p}__rot"]))
        u, r = [], []
        for seed in SEEDS:
            tr, va, te = stratified_split(lab, seed); gal = np.concatenate([tr, va])
            u.append(ap_mean(Z0[te] @ Z0[gal].T, lab[te], lab[gal]))
            r.append(ap_mean(Zr[te] @ Z0[gal].T, lab[te], lab[gal]))    # rotated queries, upright gallery
        rot[p] = (np.array(u), np.array(r))
    with open(out / "rotation_retrieval.csv", "w") as f:
        f.write("normalisation,mAP_upright_queries,mAP_rotated_queries,drop,drop_lo,drop_hi,p_t\n")
        for p in PIPELINES:
            s = paired(rot[p][0], rot[p][1])
            f.write(f"{p},{rot[p][0].mean():.5f},{rot[p][1].mean():.5f},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[6]:.3g}\n")
    print(open(out / "rotation_retrieval.csv").read())

    # ------------------------------------------------ fusion experiments (parallel over backbones)
    tasks = [(bb, str(base / rel), mags, labels, Pn, Pm) for bb, rel in BACKBONES.items() if (base / rel).exists()]
    recs, ckas = [], []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r, c in pool.imap_unordered(run_backbone, tasks):
            recs += r; ckas += c
            print(f"  {r[0][0]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "phase4_records.csv", "w") as f:
        f.write("backbone,pipeline,method,seed,deep_mAP,zernike_mAP,fused_mAP,gain,omega,param\n")
        for r in recs:
            f.write(",".join(map(str, r[:4])) + "," + ",".join(f"{x:.6f}" for x in r[4:9]) + f",{r[9]}\n")
    with open(out / "cka_m_split.csv", "w") as f:
        f.write("backbone,pipeline,part,order,linear_CKA\n")
        f.writelines(",".join(map(str, r[:4])) + f",{r[4]:.6f}\n" for r in ckas)

    R = {}
    for r in recs:
        R.setdefault((r[0], r[1], r[2]), {})[r[3]] = r
    get = lambda bb, p, m, i=6: np.array([R[(bb, p, m)][s][i] for s in SEEDS])
    bbs = [t[0] for t in tasks]
    # ------------------------------------------------ paired comparisons
    rows = []
    def add(name, bb, A_, B_, extra=""):
        rows.append((name, bb, extra) + paired(A_, B_))
    for bb in bbs:
        for p in PIPELINES:
            for m in ["trunc", "parseval", "power", "parseval_trunc"]:
                add(f"{m} - full", bb, get(bb, p, m), get(bb, p, "full"), p)
            add("m0 unique (full - no_m0)", bb, get(bb, p, "full"), get(bb, p, "no_m0"), p)
            for g in range(len(M_GROUPS)):
                add(f"mgrp{g} unique (full - out)", bb, get(bb, p, "full"), get(bb, p, f"mgrp_out_{g}"), p)
        for p in PIPELINES[1:]:
            add("full: pipeline - LEG_128", bb, get(bb, p, "full"), get(bb, "LEG_128", "full"), p)
            add("best(parseval_trunc): pipeline - LEG_128 full", bb, get(bb, p, "parseval_trunc"), get(bb, "LEG_128", "full"), p)
        add("full: CEN_MAX - BOX_CIR", bb, get(bb, "CEN_MAX_EX_256", "full"), get(bb, "BOX_CIR_EX_256", "full"))
        add("full: CEN_RG - CEN_MAX", bb, get(bb, "CEN_RG_EX_256", "full"), get(bb, "CEN_MAX_EX_256", "full"))
    with open(out / "paired_stats.csv", "w") as f:
        f.write("comparison,backbone,pipeline,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_ttest,p_wilcoxon\n")
        for r in rows:
            f.write(",".join(map(str, r[:3])) + "," + ",".join(f"{x:.5f}" for x in r[3:6]) +
                    f",{r[6]},{r[7]},{r[8]},{r[9]:.3g},{r[10]:.3g}\n")
    # ------------------------------------------------ method summary + selected parameters
    with open(out / "method_summary.csv", "w") as f:
        f.write("backbone,pipeline,method,deep_mAP,zernike_mAP,fused_mAP,gain_mean,omega_values,param_values\n")
        for (bb, p, m), d in sorted(R.items()):
            rr = [d[s] for s in SEEDS]
            f.write(f"{bb},{p},{m},{np.mean([x[4] for x in rr]):.5f},{np.mean([x[5] for x in rr]):.5f},"
                    f"{np.mean([x[6] for x in rr]):.5f},{np.mean([x[7] for x in rr]):.5f},"
                    f"{'/'.join(str(x[8]) for x in rr)},{'/'.join(x[9] for x in rr)}\n")
    # ------------------------------------------------ figures
    meths = ["full", "trunc", "parseval", "power", "parseval_trunc"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for ax, bb in zip(axs.ravel(), bbs):
        x = np.arange(len(PIPELINES)); wdt = 0.16
        for k, m in enumerate(meths):
            g = np.array([get(bb, p, m, 7).mean() for p in PIPELINES])
            ax.bar(x + (k - 2) * wdt, g, wdt, label=m)
        lo = min(get(bb, p, m, 7).mean() for p in PIPELINES for m in meths)
        hi = max(get(bb, p, m, 7).mean() for p in PIPELINES for m in meths)
        ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.set_xticks(x); ax.set_xticklabels([p.replace("_EX_256", "") for p in PIPELINES], fontsize=7)
        ax.set_title(bb)
    axs[0, 0].set_ylabel("fused mAP gain over Deep"); axs.ravel()[len(bbs) - 1].legend(fontsize=7)
    for ax in axs.ravel()[len(bbs):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "fig_method_gains.png", dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for bb in bbs:
        only = [get(bb, MAIN, f"mgrp_only_{g}", 7).mean() for g in range(len(M_GROUPS))]
        uniq = [(get(bb, MAIN, "full", 7) - get(bb, MAIN, f"mgrp_out_{g}", 7)).mean() for g in range(len(M_GROUPS))]
        axs[0].plot(only, marker="o", label=bb); axs[1].plot(uniq, marker="o", label=bb)
    for ax, t in zip(axs, ["gain with this m-group only", "unique contribution (full - leave-group-out)"]):
        ax.set_xticks(range(len(M_GROUPS))); ax.set_xticklabels([f"m={lo}" if lo == hi else f"m={lo}-{hi}" for lo, hi in M_GROUPS])
        ax.axhline(0, color="gray", lw=.6); ax.set_title(f"{MAIN}: {t}", fontsize=9)
    axs[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_m_groups.png", dpi=200); plt.close(fig)

    ck = {}
    for r in ckas:
        ck.setdefault((r[0], r[1], r[2]), {})[r[3]] = r[4]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, part in zip(axs, ["m0", "mpos"]):
        for bb in bbs:
            d = ck[(bb, MAIN, part)]; ks = sorted(d)
            ax.plot(ks, [d[k] for k in ks], marker="o", ms=2.5, lw=1.1, label=bb)
        ax.set_title(f"{MAIN}: CKA(Deep, order-n, {'m = 0' if part == 'm0' else 'm > 0'} part)", fontsize=9)
        ax.set_xlabel("radial order n")
    axs[0].set_ylabel("linear CKA"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_cka_m_split.png", dpi=200); plt.close(fig)

    key = ("ResNet-18", "LEG_128", "full")
    if key in R:
        print(f"Baseline (LEG_128, full descriptor): Deep {get(*key, 4).mean():.4f}, "
              f"Zernike {get(*key, 5).mean():.4f}, Fixed {get(*key, 6).mean():.4f}")
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
