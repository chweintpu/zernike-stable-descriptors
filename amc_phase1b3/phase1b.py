"""Phase 1b: (1) sharper ranking-stability bound, (2) convergence-rate fits,
(3) recompute area resampling with 8x8 sub-sampling (G=256, 512) to test saturation.
Spyder: press F5.   Terminal: python phase1b.py [--workers 6] [--skip-recompute]
Reads  <base>/results_amc_phase1/moments_all.npz
Writes <base>/results_amc_phase1b/
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zernike_amc as za
from phase1b_worker import EXTRA_NAMES, work

WORKERS = None   # Spyder users: None = CPU cores - 1

def desc(C):
    d = np.abs(C); return d / np.linalg.norm(d, axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-recompute", action="store_true")
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates for rate CIs")
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase1b"; per = out / "per_image"
    per.mkdir(parents=True, exist_ok=True)
    M = np.load(base / "results_amc_phase1" / "moments_all.npz")
    names = [str(s) for s in M["names"]]; labels = M["labels"]; Pn = M["pairs"][:, 0]; N = len(labels)
    ORD = np.arange(Pn.max() + 1)

    # ---------------------------------------------------------- (3) recompute
    D = {k: M[k] for k in names}
    if not a.skip_recompute:
        fs = za.list_mpeg7(base / "data" / "mpeg7")
        assert [f.name for f in fs[:N]] == [str(x) for x in M["files"]], "file order mismatch"
        tasks = [(i, str(fs[i]), str(per)) for i in range(N) if not (per / f"{i:04d}.npz").exists()]
        print(f"[recompute] {N-len(tasks)} done, {len(tasks)} to compute, workers={a.workers}")
        t0 = time.time()
        if tasks:
            with Pool(a.workers) as pool:
                for k, _ in enumerate(pool.imap_unordered(work, tasks, chunksize=2), 1):
                    if k % 50 == 0 or k == len(tasks):
                        el = time.time() - t0
                        print(f"  {k}/{len(tasks)}  {el/60:.1f} min, ETA {el/k*(len(tasks)-k)/60:.1f} min", flush=True)
        for k in EXTRA_NAMES:
            D[k] = np.stack([np.load(per / f"{i:04d}.npz")[k] for i in range(N)])
        np.savez(out / "moments_extra.npz", **{k: D[k] for k in EXTRA_NAMES})
    elif (out / "moments_extra.npz").exists():
        E = np.load(out / "moments_extra.npz"); D.update({k: E[k] for k in E.files})

    Dsc = {k: desc(v) for k, v in D.items()}
    ref_of = lambda k: "REF_CIR" if k.startswith("CIR") else "REF_INS"
    configs = [k for k in Dsc if not k.startswith("REF")]

    # ---------------------------------------------------------- (1) ranking-stability bounds
    # top-1 of the reference ranking is certified unchanged if, for every candidate j != t,
    #   S(q,t) - S(q,j) > B(q,t,j).
    # crude  : B = 2 e_q + e_t + e_j
    # sharp  : B = e_q (|t-j| + e_t + e_j) + e_t |q-t| + e_j |q-j| + |e_t^2 - e_j^2| / 2
    #          (uses <t, e_t> = -|e_t|^2/2 for unit vectors; all quantities a-priori)
    rows = []
    for k in configs:
        r = Dsc[ref_of(k)]; e = np.linalg.norm(Dsc[k] - r, axis=1)
        S = r @ r.T; np.fill_diagonal(S, -np.inf)
        Dist = np.sqrt(np.maximum(0, 2 - 2 * np.clip(r @ r.T, -1, 1)))
        t = np.argmax(S, 1); idx = np.arange(N)
        gap = S[idx, t][:, None] - S                         # N x N
        B_crude = 2 * e[:, None] + e[t][:, None] + e[None, :]
        B_sharp = (e[:, None] * (Dist[t] + e[t][:, None] + e[None, :])
                   + (e[t] * Dist[idx, t])[:, None] + e[None, :] * Dist
                   + 0.5 * np.abs(e[t][:, None] ** 2 - e[None, :] ** 2))
        mask = np.ones((N, N), bool); mask[idx, idx] = False; mask[idx, t] = False
        cert_c = np.all((gap > B_crude) | ~mask, 1).mean()
        cert_s = np.all((gap > B_sharp) | ~mask, 1).mean()
        St = Dsc[k] @ Dsc[k].T; np.fill_diagonal(St, -np.inf)
        agree = (np.argmax(St, 1) == t).mean()
        # certified "top-1 class unchanged": every j of another class stays below t
        diffc = labels[None, :] != labels[t][:, None]
        cert_cls = np.all((gap > B_sharp) | ~(mask & diffc), 1).mean()
        cls_agree = (labels[np.argmax(St, 1)] == labels[t]).mean()
        rows.append([k, e.mean(), agree, cert_c, cert_s, cls_agree, cert_cls])
    hdr = "config,err_mean,top1_agree_observed,certified_crude,certified_sharp,top1_class_agree_observed,certified_class_sharp"
    with open(out / "ranking_stability.csv", "w") as f:
        f.write(hdr + "\n" + "\n".join(",".join([r[0]] + [f"{x:.4f}" for x in r[1:]]) for r in rows) + "\n")

    # ---------------------------------------------------------- (2) convergence-rate fits
    # eta_n(G) = C n^a h^b,  h = 2/G, fitted on orders 2..32 (orders 0,1 have tiny magnitudes)
    def eta_blocks(k, sel=slice(None)):
        d, r = Dsc[k][sel], Dsc[ref_of(k)][sel]
        e = np.stack([np.sqrt(np.mean(np.sum((d[:, Pn == n] - r[:, Pn == n]) ** 2, 1))) for n in ORD])
        s = np.stack([np.sqrt(np.mean(np.sum(r[:, Pn == n] ** 2, 1))) for n in ORD])
        return e / s
    fams = {"LEG": ["LEG_128", "LEG_256", "LEG_512"],
            "INS_NN": ["INS_NN_128", "INS_NN_256", "INS_NN_512"],
            "CIR_NN": ["CIR_NN_128", "CIR_NN_256", "CIR_NN_512"],
            "INS_AR(ss4/2)": ["INS_AR_128", "INS_AR_256", "INS_AR_512"],
            "CIR_AR(ss4/2)": ["CIR_AR_128", "CIR_AR_256", "CIR_AR_512"],
            "INS_AR(ss8)": ["INS_AR_128", "INS_AR_256_ss8", "INS_AR_512_ss8"],
            "CIR_AR(ss8)": ["CIR_AR_128", "CIR_AR_256_ss8", "CIR_AR_512_ss8"]}
    fams = {f: ks for f, ks in fams.items() if all(k in Dsc for k in ks)}
    Gs = np.array([128, 256, 512]); nn = ORD[2:]
    def fit(etas, use):
        X = []; y = []
        for gi in use:
            for n in nn:
                X.append([1, np.log(n), np.log(2 / Gs[gi])]); y.append(np.log(etas[gi][n]))
        coef = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)[0]
        return coef[1], coef[2]
    rng = np.random.default_rng(0); rate_rows = []; per_order = {}
    for f, ks in fams.items():
        etas = [eta_blocks(k) for k in ks]
        per_order[f] = [np.log2(etas[0] / etas[1]), np.log2(etas[1] / etas[2])]
        for use, tag in [((0, 1), "128-256"), ((1, 2), "256-512"), ((0, 1, 2), "128-512")]:
            a_, b_ = fit(etas, use); bs = []
            for _ in range(a.boot):
                sel = rng.integers(0, N, N)
                bs.append(fit([eta_blocks(k, sel) for k in ks], use))
            bs = np.array(bs)
            rate_rows.append([f, tag, a_, *np.percentile(bs[:, 0], [2.5, 97.5]), b_, *np.percentile(bs[:, 1], [2.5, 97.5])])
    with open(out / "convergence_rates.csv", "w") as fh:
        fh.write("family,grids,order_exponent_a,a_lo,a_hi,h_exponent_b,b_lo,b_hi\n")
        for r in rate_rows:
            fh.write(",".join([r[0], r[1]] + [f"{x:.3f}" for x in r[2:]]) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, j, ttl in [(axs[0], 0, "G 128 -> 256"), (axs[1], 1, "G 256 -> 512")]:
        for f in fams:
            ax.plot(ORD, per_order[f][j], marker="o", ms=2.5, lw=1.1, label=f)
        ax.axhline(1, color="gray", ls=":"); ax.axhline(2, color="gray", ls=":")
        ax.set_title(f"observed order  log2(eta_h / eta_h/2),  {ttl}"); ax.set_xlabel("radial order n")
    axs[0].set_ylabel("empirical convergence order"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_convergence_order_by_n.png", dpi=200); plt.close(fig)
    print(open(out / "ranking_stability.csv").read()); print(open(out / "convergence_rates.csv").read())
    print(f"outputs in {out}")

if __name__ == "__main__":
    main()
