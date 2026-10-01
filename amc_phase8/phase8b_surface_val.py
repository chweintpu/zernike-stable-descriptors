"""Phase 8b (response to the v6 review): validation-only classification bandwidth surface
(MPEG-7 / ResNet-18 / Deep + CEN-RG, all ten splits) and a precisely defined variance-explained
statistic.  The test part is never used.  Spyder: F5.  Writes results_amc_phase8/surface_val/"""
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
from phase8_classification import build
from phase8_worker import init, run_surface_val

WORKERS = None

def r2(y, groups):
    """ANOVA-type R^2 = 1 - sum (y_i - mean of its group)^2 / sum (y_i - overall mean)^2, and the
    adjusted value 1 - (1 - R^2)(n - 1)/(n - g), with g the number of non-empty groups."""
    y = np.asarray(y); gm = {g: y[groups == g].mean() for g in np.unique(groups)}
    fit = np.array([gm[g] for g in groups]); n, g = len(y), len(gm)
    R = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return R, 1 - (1 - R) * (n - 1) / (n - g), g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase8" / "surface_val"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); data = build("MPEG-7", base)
    data["deep"] = {"ResNet-18": data["deep"]["ResNet-18"]}
    seeds = sorted(data["splits"]); ks = sorted(data["reps"]["CEN-RG"])
    tasks = [("ResNet-18", s, k) for s in seeds for k in ks]
    rows = []
    with Pool(min(a.workers, len(tasks)), initializer=init, initargs=(data,)) as pool:
        for r in pool.imap_unordered(run_surface_val, tasks):
            rows += r
    with open(out / "surface_val_per_split.csv", "w") as f:
        f.write("seed,k,N,val_acc,selected_C\n")
        f.writelines(f"{s},{k:g},{N},{v:.5f},{C:g}\n" for s, k, N, v, C in sorted(rows))
    cells = sorted({(k, N) for _, k, N, _, _ in rows})
    y = np.array([np.mean([v for s, k, N, v, _ in rows if (k, N) == c]) for c in cells])   # mean over splits
    K = np.array([c[0] for c in cells]); Nn = np.array([c[1] for c in cells]); L = (Nn + 1) / K
    with open(out / "surface_val_mean.csv", "w") as f:
        f.write("k,N,bandwidth,mean_val_acc\n")
        f.writelines(f"{k:g},{N},{l:.3f},{v:.5f}\n" for k, N, l, v in zip(K, Nn, L, y))
    lines = ["Variance explained by group means (cells = (k, N) pairs, y = validation accuracy averaged over 10 splits)",
             "grouping,groups,R2,adjusted_R2"]
    for name, g in [("bandwidth, unit bins ceil((N+1)/k)", np.ceil(L)), ("bandwidth, bins of width 0.5", np.ceil(2 * L)),
                    ("bandwidth, bins of width 2", np.ceil(L / 2)), ("k", K), ("N", Nn)]:
        R, Ra, ng = r2(y, g); lines.append(f"{name},{ng},{R:.3f},{Ra:.3f}")
    i = int(np.argmax(y)); lines.append(f"maximum mean validation accuracy {y[i]:.4f} at k={K[i]:g}, N={Nn[i]}, bandwidth {L[i]:.2f}")
    for lo, hi in [(0, 3), (3, 5), (5, 10), (10, 17)]:
        m = (L > lo) & (L <= hi); lines.append(f"bandwidth ({lo},{hi}]: mean {y[m].mean():.4f}, min {y[m].min():.4f}, max {y[m].max():.4f}, cells {m.sum()}")
    (out / "surface_val_report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    Ns = sorted(set(Nn)); H = np.array([[y[cells.index((k, N))] for k in ks] for N in Ns])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[-.5, len(ks) - .5, Ns[0] - 1, Ns[-1] + 1])
    for bwv in (6, 8, 10):
        ax.plot(range(len(ks)), [bwv * k - 1 for k in ks], "w--", lw=.8)
    ax.set_ylim(Ns[0] - 1, Ns[-1] + 1); ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks])
    ax.set_xlabel("k"); ax.set_ylabel("N"); fig.colorbar(im, ax=ax, label="validation accuracy")
    ax.set_title("Validation accuracy, MPEG-7 / ResNet-18 / Deep + CEN-RG, linear SVM\n"
                 "mean over 10 splits; white dashed: (N+1)/k = 6, 8, 10", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_classification_surface_val.png", dpi=200); plt.close(fig)
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
