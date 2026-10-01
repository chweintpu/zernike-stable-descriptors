"""Phase 3: order-resolved Deep-Zernike complementarity spectrum.
Spyder: press F5.   Terminal: python phase3_spectrum.py [--workers 6]
Reads  <base>/results_amc_phase1/moments_all.npz (+ results_amc_phase1b/moments_extra.npz if present)
       cached Deep features of the six backbones
Writes <base>/results_amc_phase3/
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
from phase3_worker import run_backbone, BANDS, SEEDS

WORKERS = None
# LEG_128 = baseline descriptor; REF_* = native-resolution truth; CIR_AR_256 = proposed main pipeline
PIPELINES = ["LEG_128", "REF_INS", "CIR_AR_256", "REF_CIR"]
BACKBONES = {
    "ResNet-18": "results_mpeg7_multibackbone_zernike/feature_cache/resnet18_mpeg7_unified.npy",
    "ResNet-50": "results_mpeg7_multibackbone_zernike/feature_cache/resnet50_mpeg7_unified.npy",
    "EfficientNet-B0": "results_mpeg7_multibackbone_zernike/feature_cache/efficientnet_b0_mpeg7_unified.npy",
    "ViT-B/16": "results_mpeg7_multibackbone_zernike/feature_cache/vit_b_16_mpeg7_unified.npy",
    "Swin-T": "results_mpeg7_multibackbone_zernike/feature_cache/swin_t_mpeg7_unified.npy",
    "DINOv2": "results_mpeg7_dinov2_zernike_foundation/feature_cache/dinov2_vitb14_mpeg7_224.npy",
}

def ci(x):
    x = np.asarray(x); m = x.mean(); s = x.std(ddof=1) if len(x) > 1 else 0.0
    h = 2.262 * s / np.sqrt(len(x)) if len(x) == 10 else 1.96 * s / np.sqrt(max(1, len(x)))
    return m, s, m - h, m + h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase3"; out.mkdir(parents=True, exist_ok=True)
    M = np.load(base / "results_amc_phase1" / "moments_all.npz")
    labels = [str(s) for s in M["labels"]]; Pn = M["pairs"][:, 0]
    src = {k: M for k in M.files}
    ex = base / "results_amc_phase1b" / "moments_extra.npz"
    if ex.exists():
        E = np.load(ex); src.update({k: E for k in E.files})
    mags = {p: np.abs(src[p][p]) for p in PIPELINES}
    tasks = [(bb, str(base / rel), mags, labels, Pn) for bb, rel in BACKBONES.items() if (base / rel).exists()]
    print(f"backbones: {[t[0] for t in tasks]}, pipelines: {PIPELINES}, workers: {min(a.workers, len(tasks))}")
    t0 = time.time(); recs, ckas = [], []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r, c in pool.imap_unordered(run_backbone, tasks):
            recs += r; ckas += c
            print(f"  {r[0][0]} done  ({(time.time()-t0)/60:.1f} min)", flush=True)

    with open(out / "retrieval_records.csv", "w") as f:
        f.write("backbone,pipeline,vtype,vid,seed,deep_mAP,zernike_mAP,fused_mAP,gain,omega,rescue\n")
        for r in recs:
            f.write(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.6f}" for x in r[5:]) + "\n")
    with open(out / "cka_spectrum.csv", "w") as f:
        f.write("backbone,pipeline,type,order,linear_CKA\n")
        for r in ckas:
            f.write(",".join(map(str, r[:4])) + f",{r[4]:.6f}\n")

    # ---- summary with 95% CI over seeds
    agg = {}
    for r in recs:
        agg.setdefault(r[:4], []).append(r)
    with open(out / "retrieval_summary.csv", "w") as f:
        f.write("backbone,pipeline,vtype,vid,deep_mAP,zernike_mAP,fused_mAP,gain_mean,gain_sd,gain_lo,gain_hi,rescue_mean,omega_mode\n")
        for k, rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
            rs = np.array([x[5:] for x in rs], float)
            g = ci(rs[:, 3]); om = np.round(rs[:, 4], 2)
            vals, cnt = np.unique(om, return_counts=True)
            f.write(",".join(map(str, k)) + "," + ",".join(f"{x:.5f}" for x in
                    [rs[:, 0].mean(), rs[:, 1].mean(), rs[:, 2].mean(), *g, rs[:, 5].mean()]) + f",{vals[np.argmax(cnt)]}\n")

    # ---- baseline summary (ResNet-18, LEG_128, full descriptor)
    key = ("ResNet-18", "LEG_128", "full", -1)
    if key in agg:
        rs = np.array([x[5:] for x in agg[key]], float)
        print(f"\nBaseline (LEG_128, full descriptor): "
              f"Deep {rs[:,0].mean():.4f}, Zernike {rs[:,1].mean():.4f}, Fixed {rs[:,2].mean():.4f}")

    # ---- figures
    bbs = [t[0] for t in tasks]; orders = np.arange(Pn.max() + 1)
    def mean_gain(bb, p, vt):
        ids = sorted({k[3] for k in agg if k[:3] == (bb, p, vt)})
        return np.array(ids), np.array([[x[8] for x in agg[(bb, p, vt, i)]] for i in ids])
    ck = {}
    for r in ckas:
        ck.setdefault(r[:3], {})[r[3]] = r[4]
    fig, axs = plt.subplots(1, len(PIPELINES), figsize=(4 * len(PIPELINES), 3.6), sharey=True)
    for ax, p in zip(np.atleast_1d(axs), PIPELINES):
        for bb in bbs:
            d = ck[(bb, p, "order")]; ax.plot(orders, [d[n] for n in orders], lw=1.2, marker="o", ms=2, label=bb)
        ax.set_title(p); ax.set_xlabel("radial order n")
    np.atleast_1d(axs)[0].set_ylabel("CKA(Deep, order-n block)"); np.atleast_1d(axs)[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_cka_by_order.png", dpi=200); plt.close(fig)

    for vt, fname, ylab, xl in [("cum", "fig_cumulative_gain.png", "fused mAP gain over Deep (orders <= n)", "max order n"),
                                ("band", "fig_band_gain.png", "fused gain, single band only", "band"),
                                ("leaveout", "fig_band_unique.png", "unique contribution (full - leave-band-out)", "band")]:
        nc = 3; nr = int(np.ceil(len(bbs) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(13, 3.6 * nr), squeeze=False)
        for ax, bb in zip(axs.ravel(), bbs):
            for p in PIPELINES:
                x, G = mean_gain(bb, p, vt)
                if vt == "leaveout":
                    full = np.array([r[8] for r in agg[(bb, p, "full", -1)]])
                    G = full[None, :] - G
                m, s = G.mean(1), G.std(1, ddof=1) * 2.262 / np.sqrt(G.shape[1])
                ax.plot(x, m, marker="o", ms=2.5, lw=1.2, label=p); ax.fill_between(x, m - s, m + s, alpha=.2)
            ax.axhline(0, color="gray", lw=.6); ax.set_title(bb); ax.set_xlabel(xl)
            if vt != "cum":
                ax.set_xticks(range(len(BANDS))); ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS], fontsize=7)
        axs[0, 0].set_ylabel(ylab); axs.ravel()[len(bbs) - 1].legend(fontsize=7)
        for ax in axs.ravel()[len(bbs):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(out / fname, dpi=200); plt.close(fig)
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
