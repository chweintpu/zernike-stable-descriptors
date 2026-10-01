"""Phase 4a: moments for new normalisations (exact area resampling) + rotated copies.
Spyder: F5.   Terminal: python phase4a_moments.py [--workers 12]
Also reports (i) O(h^2) convergence of exact area resampling vs the phase-1 reference,
(ii) rotation-invariance error of every normalisation.
Writes <base>/results_amc_phase4/moments_phase4.npz and analysis files.
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
import zernike_amc as za
from phase4a_worker import KEYS, ROT, work

WORKERS = None

def desc(C):
    d = np.abs(C); return d / np.linalg.norm(d, axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=1400)
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase4"; per = out / "per_image_4a"
    per.mkdir(parents=True, exist_ok=True)
    fs = za.list_mpeg7(base / "data" / "mpeg7"); assert len(fs) == 1400
    N = min(a.limit, 1400)
    tasks = [(i, str(fs[i]), str(per)) for i in range(N) if not (per / f"{i:04d}.npz").exists()]
    print(f"{N-len(tasks)} done, {len(tasks)} to compute, workers={a.workers}")
    t0 = time.time()
    if tasks:
        with Pool(a.workers) as pool:
            for k, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                if k % 50 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)}  {el/60:.1f} min, ETA {el/k*(len(tasks)-k)/60:.1f} min", flush=True)
    D = {k: np.stack([np.load(per / f"{i:04d}.npz")[k.replace('|', '__')] for i in range(N)]) for k in KEYS}
    extra = np.stack([np.load(per / f"{i:04d}.npz")["extra"] for i in range(N)])
    np.savez(out / "moments_phase4.npz", keys=np.array(KEYS), files=np.array([f.name for f in fs[:N]]),
             labels=np.array([za.label_of(f) for f in fs[:N]]), pairs=np.array(za.PAIRS),
             cenrg_outside=extra[:, 0], angles=extra[:, 1], **{k.replace("|", "__"): v for k, v in D.items()})
    Pn = np.array(za.PAIRS)[:, 0]; ORD = np.arange(Pn.max() + 1)
    lines = []

    # (i) convergence of exact area resampling against the phase-1 native reference REF_CIR
    p1 = base / "results_amc_phase1" / "moments_all.npz"
    if p1.exists():
        ref = desc(np.load(p1)["REF_CIR"][:N])
        errs = {G: np.linalg.norm(desc(D[f"BOX_CIR_EX_{G}|orig"]) - ref, axis=1).mean() for G in (128, 256, 512)}
        lines.append("exact-area BOX_CIR vs native reference: mean descriptor error " +
                     ", ".join(f"G={G}: {e:.5f}" for G, e in errs.items()) +
                     f";  observed order 128->256: {np.log2(errs[128]/errs[256]):.2f}, 256->512: {np.log2(errs[256]/errs[512]):.2f}")
    # (ii) rotation invariance
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, _, _ in ROT:
        d0, d1 = desc(D[f"{name}|orig"]), desc(D[f"{name}|rot"])
        e = np.linalg.norm(d1 - d0, axis=1)
        eta = [np.sqrt(np.mean(np.sum((d1[:, Pn == n] - d0[:, Pn == n]) ** 2, 1))) /
               np.sqrt(np.mean(np.sum(d0[:, Pn == n] ** 2, 1))) for n in ORD]
        rows.append((name, e.mean(), np.percentile(e, 95), e.max()))
        ax.plot(ORD, eta, marker="o", ms=2.5, lw=1.2, label=name)
    ax.set_yscale("log"); ax.set_xlabel("radial order n"); ax.set_ylabel("relative change under rotation")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "fig_rotation_invariance_by_order.png", dpi=200)
    with open(out / "rotation_invariance.csv", "w") as f:
        f.write("normalisation,mean_change,p95_change,max_change\n")
        f.writelines(f"{r[0]},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f}\n" for r in rows)
    co = extra[:, 0]
    lines.append(f"CEN_RG (k=2) foreground outside disk: mean {co.mean():.4f}, max {co.max():.4f}, n>1%: {(co>.01).sum()}")
    lines += [f"rotation change {r[0]:16s} mean {r[1]:.5f}  p95 {r[2]:.5f}  max {r[3]:.5f}" for r in rows]
    (out / "phase4a_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"outputs in {out}  ({(time.time()-t0)/60:.1f} min)")

if __name__ == "__main__":
    main()
