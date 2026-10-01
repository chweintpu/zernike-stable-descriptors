"""Phase 9: controlled perturbation test of Proposition 4.2 (scale-functional stability).
Erosion and dilation by 1-5 native pixels (symmetric-difference perturbations) and a small remote blob
at distance 2 R_max (the construction in the proof of Proposition 4.2(b)).
For each perturbation: relative change of r_g, sqrt(area) and R_max, centroid shift, ratio of the observed
r_g change to the bound (12), and the change of the normalized descriptor (N = 32, G = 256, exact coverage)
for CEN-RG, CEN-AREA (k = 2) and CEN-MAX.  CPU only.  Spyder: F5.  Writes results_amc_phase9/"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase5_data import load_dataset
from phase9_worker import work, NORMS

WORKERS = None
DATASETS = ["MPEG-7"]            # add "ETHZ", "Kimia-216", "Kimia-99" if desired (needs the dataset-loading environment)
COLS = ["shape", "perturbation", "param", "delta_rel", "within_assumption", "d_rg_rel", "d_sqrtA_rel", "d_Rmax_rel",
        "centroid_shift_rel", "rg_change_over_bound"] + [f"desc_change_{n}" for n, _, _ in NORMS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase9"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for ds in DATASETS:
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        tasks = [(i, D["items"][i], 2026) for i in range(len(D["labels"]))]
        rows = []
        with Pool(a.workers) as pool:
            for c, r in enumerate(pool.imap_unordered(work, tasks, chunksize=4), 1):
                rows += r
                if c % 200 == 0 or c == len(tasks):
                    print(f"[{ds}] {c}/{len(tasks)} shapes ({(time.time()-t0)/60:.1f} min)", flush=True)
        R = np.array([r[3:] for r in rows], float); kind = np.array([r[1] for r in rows]); par = np.array([r[2] for r in rows])
        with open(out / f"stability_per_shape_{ds}.csv", "w") as f:
            f.write(",".join(COLS) + "\n")
            f.writelines(",".join(map(str, r[:3])) + "," + ",".join(f"{x:.6g}" for x in r[3:]) + "\n" for r in rows)
        lines = ["perturbation,param,n_valid,delta_rel_median,d_rg_rel_median,d_rg_rel_p95,d_sqrtA_rel_median,"
                 "d_Rmax_rel_median,d_Rmax_rel_p95,d_Rmax_rel_max,max_rg_change_over_bound,frac_within_assumption,"
                 + ",".join(f"desc_change_{n}_median,desc_change_{n}_p95" for n, _, _ in NORMS)]
        for kd in ("erosion", "dilation", "remote blob"):
            for p in sorted(set(par[kind == kd])):
                sel = (kind == kd) & (par == p) & ~np.isnan(R[:, 0]); X = R[sel]
                ok = X[:, 1] > 0.5
                q = lambda j, f=np.nanmedian: f(X[:, j]) if len(X) else np.nan
                lines.append(f"{kd},{p},{sel.sum()},{q(0):.4g},{q(2):.4g},{np.nanpercentile(X[:,2],95):.4g},{q(3):.4g},{q(4):.4g},"
                             f"{np.nanpercentile(X[:,4],95):.4g},{np.nanmax(X[:,4]):.4g},{(np.nanmax(X[ok,6]) if ok.any() else np.nan):.4g},{ok.mean():.3f},"
                             + ",".join(f"{np.nanmedian(X[:,7+j]):.4g},{np.nanpercentile(X[:,7+j],95):.4g}" for j in range(len(NORMS))))
        (out / f"stability_summary_{ds}.csv").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
        nd = [r for r in rows if not np.isnan(r[3]) and any(np.isnan(r[10:]))]
        (out / f"degenerate_descriptors_{ds}.csv").write_text("shape,perturbation,param\n" + "".join(f"{r[0]},{r[1]},{r[2]}\n" for r in nd))
        print(f"[{ds}] perturbations with a degenerate (zero) descriptor: {len(nd)} (listed in degenerate_descriptors_{ds}.csv)")
        # Figure A: relative scale change vs relative symmetric difference (erosion and dilation)
        ed = np.isin(kind, ["erosion", "dilation"]) & ~np.isnan(R[:, 0])
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for j, lab, c in ((2, "radius of gyration", "C0"), (3, "sqrt(area)", "C2"), (4, "maximal radius", "C3")):
            axs[0].scatter(R[ed, 0], R[ed, j], s=2, alpha=.25, color=c, label=lab)
        axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("relative symmetric difference |Ω Δ Ω'| / |Ω|")
        axs[0].set_ylabel("relative change of the scale functional"); axs[0].legend(markerscale=5, fontsize=8)
        axs[0].set_title(f"{ds}: erosion and dilation by 1–5 pixels", fontsize=9)
        bl = (kind == "remote blob") & ~np.isnan(R[:, 0])
        data = [R[bl, j][np.isfinite(R[bl, j])] for j in (2, 3, 4)]
        axs[1].boxplot(data, labels=["radius of gyration", "sqrt(area)", "maximal radius"], showfliers=False)
        axs[1].set_yscale("log"); axs[1].set_ylabel("relative change of the scale functional")
        axs[1].set_title(f"{ds}: remote blob of radius 1–3 pixels at distance 2 R_max", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"fig_scale_stability_{ds}.png", dpi=200); plt.close(fig)
        # Figure B: descriptor change by normalization
        fig, ax = plt.subplots(figsize=(8, 3.8)); labels = []; pos = 0
        for kd in ("erosion", "dilation", "remote blob"):
            for p in sorted(set(par[kind == kd])):
                sel = (kind == kd) & (par == p) & ~np.isnan(R[:, 0])
                for j, (n, _, _) in enumerate(NORMS):
                    ax.boxplot(R[sel, 7 + j][np.isfinite(R[sel, 7 + j])], positions=[pos + j * 0.27], widths=0.22, showfliers=False,
                               patch_artist=True, boxprops=dict(facecolor=f"C{[0,2,3][j]}", alpha=.6))
                labels.append((pos + 0.27, f"{kd[:4]} {p}")); pos += 1
        ax.set_xticks([x for x, _ in labels]); ax.set_xticklabels([l for _, l in labels], fontsize=7, rotation=45)
        ax.set_yscale("log"); ax.set_ylabel("descriptor change")
        ax.set_title(f"{ds}: descriptor change (blue CEN-RG, green CEN-AREA, red CEN-MAX)", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"fig_descriptor_stability_{ds}.png", dpi=200); plt.close(fig)
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
