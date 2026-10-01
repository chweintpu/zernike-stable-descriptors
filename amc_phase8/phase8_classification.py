"""Phase 8: linear-SVM classification with the retrieval splits (pilot first).
C1 Deep | C2 BOX-INS | C3 CEN-AREA | C4 CEN-RG | (C7 CEN-MAX) | C5 Deep+BOX-INS | C6 Deep+CEN-RG | (C8 Deep+CEN-AREA)
+ rotation test (MPEG-7) + optional (k, N) accuracy surface (MPEG-7 / ResNet-18).
Zernike (k, N) and SVM C are chosen on validation only; the final model is refitted on train+val.
Needs phase 6 and phase 7 / 7e moment files.  Spyder: F5."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse, time, traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase5_data import load_dataset
from phase8_worker import init, run_zernike, run_deep, run_surface
try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None

# ------------------------------------------------------------------ settings
PILOT = False                # True: MPEG-7 / ResNet-18 only.  False: all datasets and backbones
SURFACE = False             # (k, N) accuracy surface for MPEG-7 / ResNet-18 / Deep + CEN-RG
SURFACE_SEEDS = 3
WORKERS = None
DATASETS = ["MPEG-7", "Kimia-216", "Kimia-99", "ETHZ"]
K_AREA_EXTRA = [5.0, 6.0, 8.0, 10.0, 12.0]

def paired(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b; n = len(d); m = d.mean(); s = d.std(ddof=1)
    h = (2.262 if n == 10 else 1.96) * s / np.sqrt(n) if n > 1 else 0
    p = ttest_rel(a, b).pvalue if (ttest_rel is not None and s > 0) else np.nan
    return m, m - h, m + h, int((d > 1e-12).sum()), int((np.abs(d) <= 1e-12).sum()), int((d < -1e-12).sum()), p

def build(ds, base):
    D = load_dataset(ds, base); labels = np.array([str(x) for x in D["labels"]]); n = len(labels)
    M6 = np.load(base / "results_amc_phase6" / f"moments_{ds}.npz", allow_pickle=True)
    M7 = np.load(base / "results_amc_phase7" / f"moments_{ds}.npz", allow_pickle=True)
    per = base / "results_amc_phase7" / f"per_image_7e_{ds}"
    for M in (M6, M7):
        assert [str(x) for x in M["labels"]] == list(labels), f"{ds}: label order mismatch"
    reps = {"BOX-INS": {0.0: np.load(D["leg_cache"]).astype(np.float64)},
            "CEN-MAX": {0.0: np.abs(M7["CEN_MAX"])},
            "CEN-AREA": {float(k.split("_k")[1]): np.abs(M7[k]) for k in M7.files if k.startswith("CEN_AREA_k")},
            "CEN-RG": {float(k): np.abs(M6[f"k{float(k):g}"]) for k in M6["k_ext"]}}
    if per.exists():
        E = [np.load(per / f"{i:04d}.npz") for i in range(n)]
        for k in K_AREA_EXTRA:
            reps["CEN-AREA"][k] = np.abs(np.stack([e[f"CEN_AREA_k{k:g}"] for e in E]))
    data = dict(labels=labels, Pn=M7["pairs"][:, 0], reps=reps, fuse=["BOX-INS", "CEN-AREA", "CEN-RG"],
                splits={s: (tr, va, te) for s, tr, va, te in D["splits"]},
                deep={bb: str(p) for bb, p in D["deep"].items() if Path(p).exists()})
    if "ROT_CEN_RG_k2" in M7.files:
        data["rot"] = {"BOX-INS": np.abs(M7["ROT_LEG_128"]), "CEN-MAX": np.abs(M7["ROT_CEN_MAX"]),
                       "CEN-AREA": np.abs(M7["ROT_CEN_AREA_k2"]), "CEN-RG": np.abs(M7["ROT_CEN_RG_k2"])}
        data["rot_k"] = {"BOX-INS": 0.0, "CEN-MAX": 0.0, "CEN-AREA": 2.0, "CEN-RG": 2.0}
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--workers", type=int, default=WORKERS or max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    base = Path(a.base); out = base / "results_amc_phase8" / ("pilot" if PILOT else "full"); out.mkdir(parents=True, exist_ok=True)
    datasets = ["MPEG-7"] if PILOT else DATASETS
    recs, surf = [], []; t0 = time.time()
    for ds in datasets:
        try:
            data = build(ds, base)
        except Exception:
            print(f"[{ds}] LOADING FAILED:\n{traceback.format_exc()}"); continue
        if PILOT:
            data["deep"] = {"ResNet-18": data["deep"]["ResNet-18"]}
        seeds = sorted(data["splits"])
        with Pool(min(a.workers, len(seeds) * max(1, len(data["deep"]))), initializer=init, initargs=(data,)) as pool:
            sel = {}
            for seed, r, s in pool.imap_unordered(run_zernike, seeds):
                recs += [(ds,) + x for x in r]; sel[seed] = s
            print(f"[{ds}] Zernike-only done ({(time.time()-t0)/60:.1f} min)", flush=True)
            tasks = [(bb, seed, sel[seed]) for bb in data["deep"] for seed in seeds]
            for r in pool.imap_unordered(run_deep, tasks):
                recs += [(ds,) + x for x in r]
            print(f"[{ds}] Deep and Deep+Zernike done ({(time.time()-t0)/60:.1f} min)", flush=True)
            if SURFACE and ds == "MPEG-7":
                st = [("ResNet-18", s, k) for s in seeds[:SURFACE_SEEDS] for k in sorted(data["reps"]["CEN-RG"])]
                for r in pool.imap_unordered(run_surface, st):
                    surf += r
                print(f"[{ds}] surface done ({(time.time()-t0)/60:.1f} min)", flush=True)
    with open(out / "classification_records.csv", "w") as f:
        f.write("dataset,backbone,seed,representation,selected,val_acc,test_acc,test_macroF1,rot_acc,rot_macroF1\n")
        f.writelines(",".join(map(str, r[:5])) + "," + ",".join(f"{x:.5f}" for x in r[5:]) + "\n" for r in recs)
    R = {}
    for r in recs:
        R.setdefault((r[0], r[1], r[3]), {})[r[2]] = r
    val = lambda ds, bb, rep, i: np.array([R[(ds, bb, rep)][s][i] for s in sorted(R[(ds, bb, rep)])])
    rows = ["dataset,backbone,metric,comparison,mean_A,mean_B,delta,ci_lo,ci_hi,wins,ties,losses,p_t"]
    summ = ["dataset,backbone,representation,test_acc,test_macroF1,selected_params"]
    for (ds, bb, rep) in sorted(R):
        if rep.startswith("ROT:"):
            continue
        summ.append(f"{ds},{bb},{rep},{val(ds,bb,rep,6).mean():.4f},{val(ds,bb,rep,7).mean():.4f},"
                    + "/".join(R[(ds, bb, rep)][s][4] for s in sorted(R[(ds, bb, rep)])))
    comps = []
    for ds in datasets:
        comps += [(ds, "-", "CEN-RG", "BOX-INS"), (ds, "-", "CEN-RG", "CEN-AREA"), (ds, "-", "CEN-RG", "CEN-MAX")]
        for bb in sorted({k[1] for k in R if k[0] == ds and k[1] != "-"}):
            comps += [(ds, bb, "Deep+CEN-RG", "Deep"), (ds, bb, "Deep+CEN-RG", "Deep+BOX-INS"),
                      (ds, bb, "Deep+CEN-RG", "Deep+CEN-AREA"), (ds, bb, "Deep+BOX-INS", "Deep")]
    for ds, bb, A_, B_ in comps:
        if (ds, bb, A_) not in R or (ds, bb, B_) not in R:
            continue
        for metric, i in (("accuracy", 6), ("macroF1", 7)):
            x, y = val(ds, bb, A_, i), val(ds, bb, B_, i); s = paired(x, y)
            rows.append(f"{ds},{bb},{metric},{A_} - {B_},{x.mean():.4f},{y.mean():.4f},{s[0]:.4f},{s[1]:.4f},{s[2]:.4f},{s[3]},{s[4]},{s[5]},{s[6]:.3g}")
    rot = ["dataset,normalization,upright_acc,rotated_acc,drop,ci_lo,ci_hi,p_t"]
    for (ds, bb, rep) in sorted(R):
        if rep.startswith("ROT:"):
            u, r_ = val(ds, bb, rep, 6), val(ds, bb, rep, 8); s = paired(u, r_)
            rot.append(f"{ds},{rep[4:]},{u.mean():.4f},{r_.mean():.4f},{s[0]:.4f},{s[1]:.4f},{s[2]:.4f},{s[6]:.3g}")
    (out / "classification_summary.csv").write_text("\n".join(summ) + "\n")
    (out / "classification_comparisons.csv").write_text("\n".join(rows) + "\n")
    (out / "classification_rotation.csv").write_text("\n".join(rot) + "\n")
    if surf:
        S = {}
        for seed, k, N, v, acc in surf:
            S.setdefault((k, N), []).append(acc)
        ks = sorted({k for k, _ in S}); Ns = sorted({N for _, N in S})
        H = np.array([[np.mean(S[(k, N)]) for k in ks] for N in Ns])
        with open(out / "classification_surface.csv", "w") as f:
            f.write("k,N,bandwidth,test_acc\n")
            f.writelines(f"{k:g},{N},{(N+1)/k:.3f},{np.mean(S[(k,N)]):.4f}\n" for (k, N) in sorted(S))
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis", extent=[-.5, len(ks) - .5, Ns[0] - 1, Ns[-1] + 1])
        for bwv in (6, 8, 10):
            ax.plot(range(len(ks)), [bwv * k - 1 for k in ks], "w--", lw=.8)
        ax.set_ylim(Ns[0] - 1, Ns[-1] + 1); ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k:g}" for k in ks])
        ax.set_xlabel("k"); ax.set_ylabel("N"); fig.colorbar(im, ax=ax, label="test accuracy")
        ax.set_title("MPEG-7 / ResNet-18 / Deep + CEN-RG, linear SVM\nwhite dashed: (N+1)/k = 6, 8, 10", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fig_classification_surface.png", dpi=200); plt.close(fig)
    print("\n".join(summ)); print("\n".join(rows)); print("\n".join(rot))
    if PILOT:
        def mean(bb, rep): return val("MPEG-7", bb, rep, 6).mean()
        ok1 = mean("-", "CEN-RG") > mean("-", "BOX-INS"); ok2 = mean("ResNet-18", "Deep+CEN-RG") > mean("ResNet-18", "Deep+BOX-INS")
        verdict = (f"PILOT: CEN-RG > BOX-INS: {ok1};  Deep+CEN-RG > Deep+BOX-INS: {ok2}  ->  "
                   + ("expand to the full study (set PILOT = False)" if ok1 and ok2 else "classification does not support expansion"))
        (out / "pilot_verdict.txt").write_text(verdict + "\n"); print(verdict)
    print(f"done in {(time.time()-t0)/60:.1f} min; outputs in {out}")

if __name__ == "__main__":
    main()
