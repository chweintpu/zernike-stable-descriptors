"""Phase 7c: (1) accuracy of four evaluation methods up to order 100 against a 60-digit reference,
(2) run time for all moments on the grids used in the paper, (3) run time of the resampling variants,
(4) the Richardson-based practical certificate (uses phase 1 and phase 4 results).
Spyder: F5 (single process; set NUM_THREADS below for BLAS).  Writes <base>/results_amc_phase7/numerics/"""
import os
NUM_THREADS = "1"                       # BLAS threads used in the timing experiments
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = NUM_THREADS
import argparse, time
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zernike_amc as za
import zernike_methods as zm

def mp_reference(N, r_vals, dps=60):
    """Kintner recurrence in 60-digit arithmetic (exact to double precision)."""
    mp.mp.dps = dps
    out = {}
    for m in range(N + 1):
        for i, rv in enumerate(r_vals):
            r = mp.mpf(rv)
            prev2 = r ** m
            out.setdefault((m, m), np.zeros(len(r_vals)))[i] = float(prev2)
            if m + 2 > N:
                continue
            prev1 = (m + 2) * r ** (m + 2) - (m + 1) * r ** m
            out.setdefault((m + 2, m), np.zeros(len(r_vals)))[i] = float(prev1)
            for n in range(m + 4, N + 1, 2):
                K1 = mp.mpf((n + m) * (n - m) * (n - 2)) / 2; K2 = 2 * n * (n - 1) * (n - 2)
                K3 = -m * m * (n - 1) - n * (n - 1) * (n - 2); K4 = mp.mpf(-n * (n + m - 2) * (n - m - 2)) / 2
                cur = ((K2 * r * r + K3) * prev1 + K4 * prev2) / K1
                out.setdefault((n, m), np.zeros(len(r_vals)))[i] = float(cur)
                prev2, prev1 = prev1, cur
    return out

def timeit(f, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return min(ts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--nmax", type=int, default=100)
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase7" / "numerics"; out.mkdir(parents=True, exist_ok=True)
    lines = []

    # (1) accuracy of radial polynomials, n <= nmax
    r = np.concatenate([np.linspace(0.02, 0.9, 45), np.linspace(0.9, 1.0, 56)])
    print("[1] 60-digit reference ...", flush=True)
    ref = mp_reference(a.nmax, r)
    err = {}
    for name, f in zm.RADIAL.items():
        t0 = time.time(); R = f(a.nmax, r)
        e = np.zeros(a.nmax + 1)
        for (n, m), v in R.items():
            e[n] = max(e[n], np.max(np.abs(v - ref[(n, m)])))
        err[name] = e; print(f"    {name}: {time.time()-t0:.1f}s", flush=True)
    with open(out / "radial_accuracy.csv", "w") as fh:
        fh.write("n," + ",".join(err) + "\n")
        fh.writelines(f"{n}," + ",".join(f"{err[k][n]:.3e}" for k in err) + "\n" for n in range(a.nmax + 1))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, e in err.items():
        ax.semilogy(range(a.nmax + 1), np.maximum(e, 1e-17), label=k)
    ax.set_xlabel("order n"); ax.set_ylabel("max |error| of $R_n^m$ on [0,1]"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_radial_accuracy.png", dpi=200); plt.close(fig)
    for n in (32, 48, 64, 80, 100):
        if n <= a.nmax:
            lines.append(f"radial max error at n={n}: " + ", ".join(f"{k} {err[k][n]:.1e}" for k in err))

    # (2) run time and accuracy of all moments on disk grids
    rng = np.random.default_rng(0)
    rr = np.sqrt(rng.random(300)) * 0.999; tt = rng.random(300) * 2 * np.pi
    xs, ys, ws = rr * np.cos(tt), rr * np.sin(tt), rng.random(300)
    rows = ["N,G,points,method,seconds,max_rel_moment_error"]
    for N in (32, 48, 64):
        # moment accuracy on 300 random points against 60-digit moments
        mp.mp.dps = 60
        refR = mp_reference(N, rr)
        Zref = np.zeros(len(zm.pairs(N)), complex)
        for p, (n, m) in enumerate(zm.pairs(N)):
            Zref[p] = (n + 1) / np.pi * np.sum(refR[(n, m)] * ws * np.exp(-1j * m * tt))
        scale = np.abs(Zref).max()
        acc = {k: np.abs(zm.moments_via_radial(k, N, xs, ys, ws) - Zref).max() / scale for k in zm.RADIAL}
        acc["Chebyshev (batched, eq. 6)"] = np.abs(zm.moments_chebyshev_batched(N, xs, ys, ws) - Zref).max() / scale
        zm.cheb_coeffs(N)                                   # exclude the one-off rational set-up from timings
        for G in (128, 256, 512):
            c = -1 + (2 * np.arange(G) + 1) / G; X, Y = np.meshgrid(c, c); keep = X * X + Y * Y <= 1
            x, y = X[keep], Y[keep]; w = np.ones_like(x)
            for k in ("factorial", "Kintner", "q-recursive", "Chebyshev"):
                t = timeit(lambda: zm.moments_via_radial(k, N, x, y, w), reps=1 if G == 512 else 2)
                rows.append(f"{N},{G},{x.size},{k} (per polynomial),{t:.3f},{acc[k]:.2e}")
            t = timeit(lambda: zm.moments_chebyshev_batched(N, x, y, w))
            rows.append(f"{N},{G},{x.size},Chebyshev (batched eq. 6),{t:.3f},{acc['Chebyshev (batched, eq. 6)']:.2e}")
            print(f"[2] N={N} G={G} done", flush=True)
    (out / "moment_runtime.csv").write_text("\n".join(rows) + "\n")

    # (3) run time of the resampling variants (resampling + moments, N = 32), 50 MPEG-7 shapes
    fs = za.list_mpeg7(base / "data" / "mpeg7")
    cvs = [za.legacy_canvas(za.load_foreground_mask(fs[i])) for i in range(0, 1400, 28)]
    rows = ["variant,G,seconds_per_shape"]
    for G in (128, 256, 512):
        for name, f in [("endpoint + nearest (baseline)", lambda cv: za.grid_moments(cv, "inscribed", G, "nearest", "endpoint")),
                        ("centre + nearest", lambda cv: za.grid_moments(cv, "inscribed", G, "nearest", "center")),
                        ("centre + subsampled coverage s=8", lambda cv: za.grid_moments(cv, "inscribed", G, "area", "center", ss=8)),
                        ("centre + exact coverage (9)", lambda cv: za.exact_grid_moments(cv, "BOX_INS", G))]:
            t0 = time.perf_counter()
            for cv in cvs:
                f(cv)
            rows.append(f"{name},{G},{(time.perf_counter()-t0)/len(cvs):.4f}")
    (out / "resampling_runtime.csv").write_text("\n".join(rows) + "\n")
    print("[3] done", flush=True)

    # (4) Richardson-based practical certificate
    p1 = base / "results_amc_phase1" / "moments_all.npz"; p4 = base / "results_amc_phase4" / "moments_phase4.npz"
    if p1.exists() and p4.exists():
        desc = lambda C: np.abs(C) / np.linalg.norm(np.abs(C), axis=1, keepdims=True)
        ref = desc(np.load(p1)["REF_CIR"]); M4 = np.load(p4)
        d = {G: desc(M4[f"BOX_CIR_EX_{G}__orig"]) for G in (128, 256, 512)}
        labels = np.array([str(x) for x in np.load(p1)["labels"]]); N = len(labels); idx = np.arange(N)
        rows = ["G,median_ratio_est_over_true,frac_est_ge_true,certified_true_eps,certified_est_eps,"
                "violations_est,top1_agree_observed,class_certified_est,class_violations_est"]
        for G in (128, 256):
            dG = d[G]; eps = np.linalg.norm(dG - ref, axis=1); est = 4 / 3 * np.linalg.norm(dG - d[2 * G], axis=1)
            S = dG @ dG.T; np.fill_diagonal(S, -np.inf); t = np.argmax(S, 1)
            Dist = np.sqrt(np.maximum(0, 2 - 2 * np.clip(dG @ dG.T, -1, 1)))
            Sr = ref @ ref.T; np.fill_diagonal(Sr, -np.inf); t_true = np.argmax(Sr, 1)
            gap = S[idx, t][:, None] - S
            mask = np.ones((N, N), bool); mask[idx, idx] = False; mask[idx, t] = False
            diffc = labels[None, :] != labels[t][:, None]
            def cert(e, m):
                B = (e[:, None] * (Dist[t] + e[t][:, None] + e[None, :]) + (e[t] * Dist[idx, t])[:, None]
                     + e[None, :] * Dist + 0.5 * np.abs(e[t][:, None] ** 2 - e[None, :] ** 2))
                return np.all((gap > B) | ~m, 1)
            ct, ce = cert(eps, mask), cert(est, mask); cce = cert(est, mask & diffc)
            viol = np.mean(ce & (t_true != t)); cviol = np.mean(cce & (labels[t_true] != labels[t]))
            rows.append(f"{G},{np.median(est/eps):.3f},{np.mean(est >= eps):.3f},{ct.mean():.4f},{ce.mean():.4f},"
                        f"{viol:.4f},{np.mean(t_true == t):.4f},{cce.mean():.4f},{cviol:.4f}")
        (out / "richardson_certificate.csv").write_text("\n".join(rows) + "\n")
        lines += rows
    else:
        lines.append("[4] skipped: phase 1 / phase 4 moment files not found")
    (out / "phase7c_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"outputs in {out}")

if __name__ == "__main__":
    main()
