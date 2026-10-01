"""Phase 7b: CEN_AREA (classical) vs CEN_RG vs CEN_MAX vs bounding-box baseline, on four datasets;
rotation robustness on MPEG-7. Spyder: F5.  Reads results_amc_phase7/moments_<dataset>.npz."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from phase5_data import load_dataset
from phase7a_worker import K_LIST
from phase7b_worker import run, ap_mean, l2n
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
    base = Path(a.base); out = base / "results_amc_phase7"; t0 = time.time(); tasks = []; rot_lines = []
    for ds in a.datasets.split(","):
        f = out / f"moments_{ds}.npz"
        if not f.exists():
            print(f"[{ds}] run phase7a first"); continue
        try:
            D = load_dataset(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        M = np.load(f, allow_pickle=True)
        assert [str(x) for x in M["labels"]] == [str(x) for x in D["labels"]], f"{ds}: label order mismatch"
        Pn = M["pairs"][:, 0]; labels = np.array([str(x) for x in D["labels"]])
        mags = {k: np.abs(M[k]) for k in M.files if k.startswith(("CEN_",))}
        mags["LEG"] = np.load(D["leg_cache"]).astype(np.float64)
        if ds == "MPEG-7":                            # rotation robustness, Zernike only
            rot_lines.append("normalisation,mean_descriptor_change,mAP_upright,mAP_rotated_queries,drop,drop_lo,drop_hi,p")
            for name in ("LEG_128", "CEN_MAX", "CEN_AREA_k2", "CEN_RG_k2"):
                Z0, Zr = l2n(np.abs(M[name])), l2n(np.abs(M[f"ROT_{name}"]))
                u, r = [], []
                for seed, tr, va, te in D["splits"]:
                    gal = np.concatenate([tr, va])
                    u.append(ap_mean(Z0[te] @ Z0[gal].T, labels[te], labels[gal]))
                    r.append(ap_mean(Zr[te] @ Z0[gal].T, labels[te], labels[gal]))
                s = paired(u, r)
                rot_lines.append(f"{name},{np.linalg.norm(Z0 - Zr, axis=1).mean():.5f},{np.mean(u):.5f},{np.mean(r):.5f},"
                                 f"{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[6]:.3g}")
            (out / "rotation_phase7.csv").write_text("\n".join(rot_lines) + "\n")
            print("\n".join(rot_lines))
        for bb, p in D["deep"].items():
            if Path(p).exists():
                tasks.append((ds, bb, str(p), mags, labels, D["splits"], Pn, K_LIST))
    print(f"{len(tasks)} jobs", flush=True)
    recs = []
    with Pool(min(a.workers, len(tasks))) as pool:
        for r in pool.imap_unordered(run, tasks):
            recs += r; print(f"  {r[0][0]} / {r[0][1]} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "phase7_records.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,seed,deep_mAP,zernike_mAP,fused_mAP,omega,param\n")
        f.writelines(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.6f}" for x in r[5:9]) + f",{r[9]}\n" for r in recs)
    R = {}
    for r in recs:
        R.setdefault(r[:4], {})[r[4]] = r
    combos = sorted({k[:2] for k in R}, key=lambda c: (DATASETS.index(c[0]), c[1]))
    get = lambda ds, bb, p, kind, i=7: np.array([R[(ds, bb, p, kind)][s][i] for s in sorted(R[(ds, bb, p, kind)])])
    # Table-8 style summary (untruncated, k = 2)
    with open(out / "normalisation_table.csv", "w") as f:
        f.write("dataset,backbone,deep,zernike_BOX-INS,zernike_CEN-AREA,zernike_CEN-MAX,zernike_CEN-RG,"
                "fused_BOX-INS,fused_CEN-AREA,fused_CEN-MAX,fused_CEN-RG\n")
        for ds, bb in combos:
            ps = ["LEG", "CEN_AREA_k2", "CEN_MAX", "CEN_RG_k2"]
            f.write(f"{ds},{bb},{get(ds,bb,'LEG','full',5).mean():.4f}," +
                    ",".join(f"{get(ds,bb,p,'full',6).mean():.4f}" for p in ps) + "," +
                    ",".join(f"{get(ds,bb,p,'full',7).mean():.4f}" for p in ps) + "\n")
    comps = [("CEN_AREA k2 full - BOX-INS full", ("CEN_AREA_k2", "full"), ("LEG", "full")),
             ("CEN_RG k2 full - CEN_AREA k2 full", ("CEN_RG_k2", "full"), ("CEN_AREA_k2", "full")),
             ("CEN_RG k2 full - CEN_MAX full", ("CEN_RG_k2", "full"), ("CEN_MAX", "full")),
             ("CEN_AREA k2 full - CEN_MAX full", ("CEN_AREA_k2", "full"), ("CEN_MAX", "full")),
             ("CEN_RG kval full - CEN_AREA kval full", ("CEN_RG_kval", "full"), ("CEN_AREA_kval", "full")),
             ("CEN_RG kval trunc - CEN_AREA kval trunc", ("CEN_RG_kval", "trunc"), ("CEN_AREA_kval", "trunc")),
             ("CEN_AREA kval trunc - BOX-INS full", ("CEN_AREA_kval", "trunc"), ("LEG", "full")),
             ("CEN_RG kval trunc - BOX-INS full", ("CEN_RG_kval", "trunc"), ("LEG", "full"))]
    with open(out / "phase7_comparisons.csv", "w") as f:
        f.write("dataset,backbone,comparison,mean_delta,ci_lo,ci_hi,wins,ties,losses,p_t\n")
        for ds, bb in combos:
            for name, A_, B_ in comps:
                s = paired(get(ds, bb, *A_), get(ds, bb, *B_))
                f.write(f"{ds},{bb},{name},{s[0]:.5f},{s[1]:.5f},{s[2]:.5f},{s[3]},{s[4]},{s[5]},{s[6]:.3g}\n")
    with open(out / "selected_params.csv", "w") as f:
        f.write("dataset,backbone,pipeline,kind,fused_mAP,selected\n")
        for ds, bb in combos:
            for p in ("CEN_AREA_kval", "CEN_RG_kval"):
                for kind in ("full", "trunc"):
                    sel = "/".join(R[(ds, bb, p, kind)][s][9] for s in sorted(R[(ds, bb, p, kind)]))
                    f.write(f"{ds},{bb},{p},{kind},{get(ds,bb,p,kind).mean():.4f},{sel}\n")
    print(open(out / "normalisation_table.csv").read())
    print(open(out / "phase7_comparisons.csv").read())
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
