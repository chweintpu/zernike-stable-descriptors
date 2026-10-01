"""Phase 7f (response to the v4 review):
(1) symmetric CEN-RG vs CEN-AREA comparison: CEN-RG with k in {2,...,8} (phase 6 moments),
    CEN-AREA with k in {2,...,12} (phase 7 / 7e moments), k, N and omega selected on validation;
(2) 95th percentile of the rotation change of CEN-AREA (Table 8);
(3) hardware / software information for the run-time tables.
Needs results_amc_phase6/moments_<ds>.npz, results_amc_phase7/moments_<ds>.npz and
results_amc_phase7/per_image_7e_<ds>/.  Spyder: F5."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, io, contextlib, platform, subprocess, sys, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from phase5_data import load_dataset
from phase7e_worker import run_eval, K_EXTRA
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

def system_info():
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}", f"processor {platform.processor()}"]
    try:
        if os.name == "nt":
            cpu = subprocess.run(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            lines.append(f"cpu {cpu}")
    except Exception:
        pass
    try:
        import psutil
        lines.append(f"memory {psutil.virtual_memory().total / 2**30:.1f} GiB, logical cores {psutil.cpu_count()}, "
                     f"physical cores {psutil.cpu_count(logical=False)}")
    except ImportError:
        lines.append(f"logical cores {os.cpu_count()} (psutil not installed: memory unknown)")
    lines.append(f"numpy {np.__version__}")
    try:
        import scipy; lines.append(f"scipy {scipy.__version__}")
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_info
        for d in threadpool_info():
            lines.append(f"threadpool: {d.get('internal_api')} {d.get('version')} ({d.get('user_api')}), "
                         f"num_threads={d.get('num_threads')}")
    except ImportError:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        lines.append("numpy config:\n" + buf.getvalue())
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase7"; t0 = time.time(); tasks = []; notes = []
    (out / "system_info.txt").write_text(system_info() + "\n")
    for ds in a.datasets.split(","):
        f6, f7 = base / "results_amc_phase6" / f"moments_{ds}.npz", out / f"moments_{ds}.npz"
        per = out / f"per_image_7e_{ds}"
        if not (f6.exists() and f7.exists() and per.exists()):
            print(f"[{ds}] missing phase 6 / 7 / 7e results"); continue
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        labels = np.array([str(x) for x in D["labels"]]); n = len(labels)
        M6, M7 = np.load(f6, allow_pickle=True), np.load(f7, allow_pickle=True)
        for M in (M6, M7):
            assert [str(x) for x in M["labels"]] == list(labels), f"{ds}: label order mismatch"
        mags = {f"CEN_RG_k{float(k):g}": np.abs(M6[f"k{float(k):g}"]) for k in M6["k_ext"]}
        mags.update({k: np.abs(M7[k]) for k in M7.files if k.startswith("CEN_AREA_k")})
        E = [np.load(per / f"{i:04d}.npz") for i in range(n)]
        for k in K_EXTRA:
            mags[f"CEN_AREA_k{k:g}"] = np.abs(np.stack([e[f"CEN_AREA_k{k:g}"] for e in E]))
        mags["LEG"] = np.load(D["leg_cache"]).astype(np.float64)
        if ds == "MPEG-7" and "ROT_CEN_AREA_k2" in M7.files:
            l2 = lambda X: np.abs(X) / np.linalg.norm(np.abs(X), axis=1, keepdims=True)
            for name in ("CEN_AREA_k2", "CEN_RG_k2", "CEN_MAX", "LEG_128"):
                ch = np.linalg.norm(l2(M7[name]) - l2(M7[f"ROT_{name}"]), axis=1)
                notes.append(f"rotation change {name}: mean {ch.mean():.4f}, 95th pct {np.percentile(ch, 95):.4f}, max {ch.max():.4f}")
        for bb, p in D["deep"].items():
            if Path(p).exists():
                tasks.append((ds, bb, str(p), mags, labels, D["splits"], M7["pairs"][:, 0]))
    print(f"{len(tasks)} jobs", flush=True)
    recs = []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r in pool.imap_unordered(run_eval, tasks):
            recs += r; print(f"  {r[0][0]} / {r[0][1]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "phase7f_records.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,seed,deep_mAP,zernike_mAP,fused_mAP,omega,param\n")
        f.writelines(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.6f}" for x in r[5:9]) + f",{r[9]}\n" for r in recs)
    R = {}
    for r in recs:
        R.setdefault(r[:4], {})[r[4]] = r
    combos = sorted({k[:2] for k in R}, key=lambda c: (DATASETS.index(c[0]), c[1]))
    get = lambda ds, bb, p, kind, i=7: np.array([R[(ds, bb, p, kind)][s][i] for s in sorted(R[(ds, bb, p, kind)])])
    rows = ["dataset,backbone,comparison,mean_A,mean_B,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_t,selected_RG,selected_AREA"]
    for ds, bb in combos:
        for kind in ("trunc", "full"):
            s = paired(get(ds, bb, "CEN_RG_kval", kind), get(ds, bb, "CEN_AREA_kval", kind))
            sr = "/".join(R[(ds, bb, "CEN_RG_kval", kind)][x][9] for x in sorted(R[(ds, bb, "CEN_RG_kval", kind)]))
            sa = "/".join(R[(ds, bb, "CEN_AREA_kval", kind)][x][9] for x in sorted(R[(ds, bb, "CEN_AREA_kval", kind)]))
            rows.append(f"{ds},{bb},CEN_RG kval(2..8) {kind} - CEN_AREA kval(2..12) {kind},"
                        f"{get(ds,bb,'CEN_RG_kval',kind).mean():.5f},{get(ds,bb,'CEN_AREA_kval',kind).mean():.5f},"
                        f"{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[3]},{s[4]},{s[5]},{s[6]:.3g},{sr},{sa}")
    (out / "phase7f_comparisons.csv").write_text("\n".join(rows) + "\n")
    ks = sorted({float(p.rsplit('_k', 1)[1]) for (_, _, p, _) in R if p.startswith("CEN_RG_k") and p[-1].isdigit()})
    curve = ["dataset,backbone,kind," + ",".join(f"RG_k{k:g}" for k in ks)]
    for ds, bb in combos:
        for kind in ("full", "trunc"):
            curve.append(f"{ds},{bb},{kind}," + ",".join(f"{get(ds,bb,f'CEN_RG_k{k:g}',kind).mean():.4f}" for k in ks))
    (out / "phase7f_rg_curves.csv").write_text("\n".join(curve) + "\n")
    (out / "phase7f_notes.txt").write_text("\n".join(notes) + "\n")
    print("\n".join(rows)); print("\n".join(notes)); print(open(out / "system_info.txt").read())
    print(f"done in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
