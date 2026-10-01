"""Regenerate all figures of the manuscript (Figures 3-6) and the Supplementary Material (Figures S1-S6)
from the saved results, with the labels used in the manuscript.
For every figure the plotted numbers are also written to figure_data/<name>.csv.
Place this folder in the Zernike folder (next to results_amc_phase1, ...) and press F5 in Spyder.
Outputs: <Zernike>/figures_final/ (PNG at 300 dpi and PDF) and <Zernike>/figures_final/figure_data/."""
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "figures_final"; DATA = OUT / "figure_data"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7.5, "savefig.dpi": 300})
KS = [2, 2.5, 3, 3.5, 4, 5, 6, 8]
COMBOS = [("MPEG-7", "ResNet-18"), ("MPEG-7", "ResNet-50"), ("MPEG-7", "EfficientNet-B0"), ("MPEG-7", "ViT-B/16"),
          ("MPEG-7", "Swin-T"), ("MPEG-7", "DINOv2"), ("Kimia-216", "ResNet-18"), ("Kimia-99", "ResNet-18"),
          ("ETHZ", "ResNet-18"), ("ETHZ", "ViT-B/16")]

def read_csv(p):
    import csv
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}")

def write(name, header, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.csv", "w") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(str(x) for x in r) + "\n" for r in rows)

def need(*paths):
    miss = [str(p) for p in paths if not Path(p).exists()]
    if miss:
        print("  skipped, missing:", *miss)
    return not miss

def desc(C):
    d = np.abs(C); return d / np.linalg.norm(d, axis=1, keepdims=True)

def eta_by_order(d, r, Pn):
    ORD = np.arange(Pn.max() + 1)
    e = np.array([np.sqrt(np.mean(np.sum((d[:, Pn == n] - r[:, Pn == n]) ** 2, 1))) for n in ORD])
    s = np.array([np.sqrt(np.mean(np.sum(r[:, Pn == n] ** 2, 1))) for n in ORD])
    return ORD, e / s

# ---------------------------------------------------------------- Figure 1
def figure1():
    print("Figure S8: Fourier-Bessel check")
    p = BASE / "results_amc_phase6" / "fourier_bessel_check.csv"
    if not need(p): return
    rows = read_csv(p); fig, ax = plt.subplots(figsize=(6.2, 3.6)); out = []
    for r in rows:
        n = [i for i in range(33) if i != 1]; v = [float(r[f"n{i}"]) for i in n]
        ax.semilogy(n, v, marker="o", ms=2.5, lw=1.1, label=f"k = {float(r['k']):g} (median distance {float(r['median_descriptor_diff']):.3f})")
        out += [(r["k"], i, x) for i, x in zip(n, v)]
    ax.set_xlabel("radial order n"); ax.set_ylabel("relative deviation")
    ax.legend(); save(fig, "FigureS8_fourier_bessel_check")
    write("FigureS8_fourier_bessel_check", ["k", "n", "median_relative_block_deviation"], out)

# ---------------------------------------------------------------- Figure 2
def figure2():
    print("Figure 3: (k, N) retrieval surface")
    p = BASE / "results_amc_phase6" / "kN_surface.csv"
    if not need(p): return
    S = {}
    for r in read_csv(p):
        S[(r["dataset"], r["backbone"], float(r["k"]), int(r["N"]))] = float(r["fused_mAP_mean"])
    fig, axs = plt.subplots(2, 5, figsize=(17, 6.4))
    for ax, (ds, bb) in zip(axs.ravel(), COMBOS):
        Ns = sorted({N for (d, b, k, N) in S if d == ds and b == bb})
        H = np.array([[S[(ds, bb, k, N)] for k in KS] for N in Ns])
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[-.5, len(KS) - .5, Ns[0] - .5, Ns[-1] + .5])
        for bw in (6, 8, 10):
            ax.plot(range(len(KS)), [bw * k - 1 for k in KS], "w--", lw=.8)
        ax.set_ylim(Ns[0] - .5, Ns[-1] + .5); ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"{k:g}" for k in KS])
        ax.set_title(f"{ds} / {bb}"); ax.set_xlabel("k"); ax.set_ylabel("N"); fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); save(fig, "Figure3_kN_surface_retrieval")
    write("Figure3_kN_surface_retrieval", ["dataset", "backbone", "k", "N", "bandwidth", "test_fused_mAP"],
          [(d, b, f"{k:g}", N, f"{(N+1)/k:.3f}", f"{v:.5f}") for (d, b, k, N), v in sorted(S.items())])

# ---------------------------------------------------------------- Figure 3
def figure3():
    print("Figure 4: classification validation surface")
    p = BASE / "results_amc_phase8" / "surface_val" / "surface_val_mean.csv"
    if not need(p): return
    R = read_csv(p); S = {(float(r["k"]), int(r["N"])): float(r["mean_val_acc"]) for r in R}
    Ns = sorted({N for _, N in S}); H = np.array([[S[(k, N)] for k in KS] for N in Ns])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[-.5, len(KS) - .5, Ns[0] - 1, Ns[-1] + 1])
    for bw in (6, 8, 10):
        ax.plot(range(len(KS)), [bw * k - 1 for k in KS], "w--", lw=.8)
    ax.set_ylim(Ns[0] - 1, Ns[-1] + 1); ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"{k:g}" for k in KS])
    ax.set_xlabel("k"); ax.set_ylabel("N"); fig.colorbar(im, ax=ax, label="validation accuracy")
    save(fig, "Figure4_classification_surface_val")
    write("Figure4_classification_surface_val", ["k", "N", "bandwidth", "mean_validation_accuracy"],
          [(f"{k:g}", N, f"{(N+1)/k:.3f}", f"{v:.5f}") for (k, N), v in sorted(S.items())])

# ---------------------------------------------------------------- Figure 4
def figure4():
    print("Figure 5: cross-task gains")
    p6 = BASE / "results_amc_phase6" / "phase6_records.csv"; p8 = BASE / "results_amc_phase8" / "full" / "classification_comparisons.csv"
    if not need(p6, p8): return
    rec = read_csv(p6); cmp_ = read_csv(p8)
    # colour = dataset, marker = backbone: no text labels, so nothing can overlap
    DCOL = {"MPEG-7": "#1b9e77", "Kimia-216": "#d95f02", "Kimia-99": "#7570b3", "ETHZ": "#e7298a"}
    BMK = {"ResNet-18": "o", "ResNet-50": "s", "EfficientNet-B0": "^", "ViT-B/16": "D", "Swin-T": "v", "DINOv2": "P"}
    BAB = {"ResNet-18": "R18", "ResNet-50": "R50", "EfficientNet-B0": "EffB0", "ViT-B/16": "ViT", "Swin-T": "Swin", "DINOv2": "DINOv2"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); out = []
    for ds, bb in COMBOS:
        a = {r["seed"]: float(r["fused_mAP"]) for r in rec if r["dataset"] == ds and r["backbone"] == bb and r["pipeline"] == "kval_ext" and r["kind"] == "trunc"}
        b = {r["seed"]: float(r["fused_mAP"]) for r in rec if r["dataset"] == ds and r["backbone"] == bb and r["pipeline"] == "LEG" and r["kind"] == "full"}
        dr = float(np.mean([a[s] - b[s] for s in a]))
        c = [r for r in cmp_ if r["dataset"] == ds and r["backbone"] == bb and r["metric"] == "accuracy" and r["comparison"] == "Deep+CEN-RG - Deep+BOX-INS"][0]
        y, lo, hi = float(c["delta"]), float(c["ci_lo"]), float(c["ci_hi"])
        ax.errorbar(dr, y, yerr=[[y - lo], [hi - y]], fmt=BMK[bb], ms=6.5, capsize=2.5, color=DCOL[ds],
                    mec="k", mew=0.5, elinewidth=0.9, alpha=0.95, zorder=3)
        out.append((ds, bb, f"{dr:.4f}", f"{y:.4f}", f"{lo:.4f}", f"{hi:.4f}"))
    ax.axhline(0, color="gray", lw=.7, zorder=1)
    ax.set_xlabel("retrieval gain in fused mAP (proposed − baseline)")
    ax.set_ylabel("classification gain in accuracy\n(Deep + CEN-RG − Deep + BOX-INS)")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], ls="", marker="o", ms=7, mfc=c, mec="k", mew=0.5, label=d) for d, c in DCOL.items()]
    h2 = [Line2D([], [], ls="", marker=m, ms=6.5, mfc="white", mec="k", label=BAB[b]) for b, m in BMK.items()]
    l1 = ax.legend(handles=h1, title="Dataset", loc="upper left", fontsize=7.5, title_fontsize=8, frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=h2, title="Backbone", loc="upper left", bbox_to_anchor=(0.24, 1.0), fontsize=7.5, title_fontsize=8, frameon=False, ncol=2)
    ylo, yhi = ax.get_ylim(); ax.set_ylim(ylo, yhi + 0.45 * (yhi - ylo))
    save(fig, "Figure5_cross_task_gain")
    write("Figure5_cross_task_gain", ["dataset", "backbone", "retrieval_gain", "classification_gain", "ci_lo", "ci_hi"], out)

# ---------------------------------------------------------------- Figures S1, S2
def figures_s1_s2():
    print("Figures S1, S2: controlled perturbations")
    p = BASE / "results_amc_phase9" / "stability_per_shape_MPEG-7.csv"
    if not need(p): return
    R = [r for r in read_csv(p) if r["delta_rel"] not in ("nan", "")]
    f = lambda r, c: float(r[c])
    ed = [r for r in R if r["perturbation"] in ("erosion", "dilation")]; bl = [r for r in R if r["perturbation"] == "remote blob"]
    names = [("d_rg_rel", "radius of gyration", "C0"), ("d_sqrtA_rel", "square root of the area", "C2"), ("d_Rmax_rel", "maximal radius", "C3")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for c, lab, col in names:
        axs[0].scatter([f(r, "delta_rel") for r in ed], [max(f(r, c), 1e-6) for r in ed], s=2, alpha=.25, color=col, label=lab)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].legend(markerscale=5)
    axs[0].set_xlabel("relative symmetric difference"); axs[0].set_ylabel("relative change of the scale functional")
    axs[0].set_title("(a) erosion and dilation by 1–5 pixels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        axs[1].boxplot([[max(f(r, c), 1e-7) for r in bl] for c, _, _ in names], showfliers=False)
    axs[1].set_xticks([1, 2, 3]); axs[1].set_xticklabels([lab for _, lab, _ in names])
    axs[1].set_yscale("log"); axs[1].set_ylabel("relative change of the scale functional")
    axs[1].set_title("(b) remote disk of radius 1–3 pixels at distance 2R_max")
    fig.tight_layout(); save(fig, "FigureS1_scale_stability")
    norms = [("desc_change_CEN-RG", "CEN-RG", "C0"), ("desc_change_CEN-AREA", "CEN-AREA", "C2"), ("desc_change_CEN-MAX", "CEN-MAX", "C3")]
    groups = [(kd, p) for kd in ("erosion", "dilation", "remote blob") for p in sorted({int(r["param"]) for r in R if r["perturbation"] == kd})]
    fig, ax = plt.subplots(figsize=(9, 3.8)); out = []
    for gi, (kd, p) in enumerate(groups):
        sel = [r for r in R if r["perturbation"] == kd and int(r["param"]) == p]
        for j, (c, lab, col) in enumerate(norms):
            v = [f(r, c) for r in sel if r[c] not in ("nan", "")]
            ax.boxplot(v, positions=[gi + (j - 1) * 0.27], widths=0.22, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=.6))
            out.append((kd, p, lab, len(v), f"{np.median(v):.5f}", f"{np.percentile(v, 25):.5f}", f"{np.percentile(v, 75):.5f}"))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{kd.split()[-1][:4]} {p}" for kd, p in groups], rotation=45, fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("descriptor change")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, alpha=.6) for _, _, c in norms], [l for _, l, _ in norms])
    fig.tight_layout(); save(fig, "FigureS2_descriptor_stability")
    write("FigureS1_S2_perturbations_summary", ["perturbation", "param", "normalization", "n", "median", "q25", "q75"], out)

# ---------------------------------------------------------------- Figures S3, S4
def figures_s3_s4():
    print("Figures S3, S4: discretization error by order")
    p1 = BASE / "results_amc_phase1" / "moments_all.npz"; p4 = BASE / "results_amc_phase4" / "moments_phase4.npz"
    pe = BASE / "results_amc_phase1b" / "moments_extra.npz"
    if not need(p1, p4): return
    M = np.load(p1, allow_pickle=True); P4 = np.load(p4, allow_pickle=True); Pn = M["pairs"][:, 0]; N = len(M["labels"])
    assert [str(x) for x in P4["files"]][:N] == [str(x) for x in M["files"]][:N], "shape order differs between phase 1 and phase 4"
    D = {k: M[k] for k in M.files if k[:3] in ("LEG", "INS", "CIR", "REF")}
    if Path(pe).exists():
        E = np.load(pe); D.update({k: E[k] for k in E.files})
    for G in (128, 256, 512):
        k = f"BOX_CIR_EX_{G}__orig"
        if k in P4.files: D[f"EX_{G}"] = P4[k][:N]
    ref = {"INS": desc(D["REF_INS"]), "CIR": desc(D["REF_CIR"])}
    refof = lambda key: ref["CIR"] if key.startswith(("CIR", "EX")) else ref["INS"]
    fam = [("LEG", "baseline (BOX-INS, endpoint grid)"), ("INS_NN", "BOX-INS, nearest neighbor"),
           ("INS_AR", "BOX-INS, subsampled coverage"), ("CIR_NN", "BOX-CIR, nearest neighbor"),
           ("CIR_AR", "BOX-CIR, subsampled coverage"), ("EX", "BOX-CIR, exact coverage")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True); out = []; eta = {}
    for ax, G in zip(axs, (128, 256, 512)):
        for key, lab in fam:
            k = f"{key}_{G}"
            if k not in D: continue
            ORD, e = eta_by_order(desc(D[k]), refof(k), Pn); eta[k] = e
            ax.semilogy(ORD, e, marker="o", ms=2.3, lw=1.1, label=lab); out += [(G, lab, n, f"{x:.6g}") for n, x in zip(ORD, e)]
        ax.set_title(f"G = {G}"); ax.set_xlabel("radial order n")
    axs[0].set_ylabel("relative block error")
    h, l = axs[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); save(fig, "FigureS3_error_by_order")
    write("FigureS3_error_by_order", ["G", "implementation", "n", "relative_block_error"], out)
    fams = [("baseline (BOX-INS, endpoint grid)", ["LEG_128", "LEG_256", "LEG_512"]),
            ("BOX-INS, nearest neighbor", ["INS_NN_128", "INS_NN_256", "INS_NN_512"]),
            ("BOX-CIR, nearest neighbor", ["CIR_NN_128", "CIR_NN_256", "CIR_NN_512"]),
            ("BOX-INS, subsampled s = 4, 2, 2", ["INS_AR_128", "INS_AR_256", "INS_AR_512"]),
            ("BOX-CIR, subsampled s = 4, 2, 2", ["CIR_AR_128", "CIR_AR_256", "CIR_AR_512"]),
            ("BOX-INS, subsampled s = 4, 8, 8", ["INS_AR_128", "INS_AR_256_ss8", "INS_AR_512_ss8"]),
            ("BOX-CIR, subsampled s = 4, 8, 8", ["CIR_AR_128", "CIR_AR_256_ss8", "CIR_AR_512_ss8"]),
            ("BOX-CIR, exact coverage", ["EX_128", "EX_256", "EX_512"])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True); out = []
    for ax, j, ttl in [(axs[0], 0, "G = 128 → 256"), (axs[1], 1, "G = 256 → 512")]:
        for lab, ks in fams:
            if not all(k in D for k in ks): continue
            e = [eta.get(k) if k in eta else eta_by_order(desc(D[k]), refof(k), Pn)[1] for k in ks]
            o = np.log2(e[j] / e[j + 1]); ax.plot(np.arange(len(o)), o, marker="o", ms=2.3, lw=1.1, label=lab)
            out += [(ttl, lab, n, f"{x:.4f}") for n, x in enumerate(o)]
        ax.axhline(1, color="gray", ls=":"); ax.axhline(2, color="gray", ls=":"); ax.set_title(ttl); ax.set_xlabel("radial order n")
    axs[0].set_ylabel("observed convergence order")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); save(fig, "FigureS4_convergence_order")
    write("FigureS4_convergence_order", ["refinement", "implementation", "n", "observed_order"], out)

# ---------------------------------------------------------------- Figure S5
def figure_s5():
    print("Figure S5: rotation by order")
    p4 = BASE / "results_amc_phase4" / "moments_phase4.npz"
    if not need(p4): return
    P = np.load(p4, allow_pickle=True); Pn = P["pairs"][:, 0]
    items = [("LEG_128", "BOX-INS, baseline implementation"), ("BOX_INS_EX_256", "BOX-INS, exact coverage"),
             ("BOX_CIR_EX_256", "BOX-CIR, exact coverage"), ("CEN_MAX_EX_256", "CEN-MAX, exact coverage"),
             ("CEN_RG_EX_256", "CEN-RG (k = 2), exact coverage")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8)); out = []
    for k, lab in items:
        d0, d1 = desc(P[f"{k}__orig"]), desc(P[f"{k}__rot"]); ORD, e = eta_by_order(d1, d0, Pn)
        ax.semilogy(ORD, e, marker="o", ms=2.3, lw=1.1, label=lab); out += [(lab, n, f"{x:.6g}") for n, x in zip(ORD, e)]
    ax.set_xlabel("radial order n"); ax.set_ylabel("relative change under rotation"); ax.legend()
    save(fig, "FigureS5_rotation_by_order"); write("FigureS5_rotation_by_order", ["normalization", "n", "relative_change"], out)

# ---------------------------------------------------------------- Figure S6
def figure_s6():
    print("Figure S6: mAP as a function of k")
    p = BASE / "results_amc_phase6" / "k_curve.csv"
    if not need(p): return
    R = read_csv(p); dss = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5)); out = []
    for j, ds in enumerate(dss):
        for row, meas, ttl in ((0, "fused_mAP", "fused mAP"), (1, "zernike_mAP", "Zernike-only mAP")):
            ax = axs[row, j]
            for i, bb in enumerate([b for d, b in COMBOS if d == ds]):
                col = f"C{i}"
                for kind, ls, mk in (("trunc", "-", "o"), ("full", "--", ".")):
                    r = [x for x in R if x["dataset"] == ds and x["backbone"] == bb and x["kind"] == kind and x["measure"] == meas][0]
                    v = [float(r[f"k{k:g}"]) for k in KS]
                    ax.plot(KS, v, ls=ls, marker=mk, color=col, label=f"{bb}" if kind == "trunc" else None)
                    out += [(ds, bb, meas, kind, f"{k:g}", f"{x:.5f}") for k, x in zip(KS, v)]
                ax.axhline(float(r["LEG"]), color=col, ls=":", lw=1); out.append((ds, bb, meas, "baseline", "", r["LEG"]))
            ax.set_title(f"{ds}: {ttl}"); ax.set_xlabel("k")
        axs[0, j].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "FigureS6_k_extended")
    write("FigureS6_k_extended", ["dataset", "backbone", "measure", "kind", "k", "mAP"], out)

if __name__ == "__main__":
    for f in (figure1, figure2, figure3, figure4, figures_s1_s2, figures_s3_s4, figure_s5, figure_s6):
        try:
            f()
        except Exception as e:
            import traceback; print(f"  FAILED: {e}"); traceback.print_exc()
    print(f"done; figures in {OUT}")
