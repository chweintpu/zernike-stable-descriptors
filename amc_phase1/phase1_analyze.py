"""Phase 1 (analysis): per-order error, SNR, trusted orders, ranking stability.
Usage:  python phase1_analyze.py
Reads  <base>/results_amc_phase1/moments_all.npz
Writes <base>/results_amc_phase1/analysis/  (CSV, JSON, PNG)
Retrieval numbers are leave-one-out on all 1,400 shapes: a numerical diagnostic,
NOT the baseline split protocol.
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
ap.add_argument("--taus", default="5,10,20", help="SNR thresholds for trusted orders")
a = ap.parse_args()
base = Path(a.base); res = base / "results_amc_phase1"; out = res / "analysis"
out.mkdir(parents=True, exist_ok=True)
taus = [float(t) for t in a.taus.split(",")]

Z = np.load(res / "moments_all.npz")
names = [str(s) for s in Z["names"]]; labels = Z["labels"]; pairs = Z["pairs"]; extra = Z["extra"]
Pn = pairs[:, 0]; ORD = np.arange(Pn.max() + 1); N = len(labels)

def desc(C):
    d = np.abs(C); return d / np.linalg.norm(d, axis=1, keepdims=True)
Dsc = {k: desc(Z[k]) for k in names}
ref_of = lambda k: "REF_CIR" if k.startswith("CIR") else "REF_INS"
configs = [k for k in names if not k.startswith("REF")]

# ---------------- signal: between-class spread of each order block (reference)
rng = np.random.default_rng(0)
I = rng.integers(0, N, 60000); J = rng.integers(0, N, 60000)
keep = labels[I] != labels[J]; I, J = I[keep], J[keep]
signal = {}
for r in ("REF_INS", "REF_CIR"):
    d = Dsc[r]
    signal[r] = np.array([np.sqrt(np.mean(np.sum((d[I][:, Pn == n] - d[J][:, Pn == n]) ** 2, 1))) for n in ORD])

# ---------------- per-order error table
rows, summ = [], {}
for k in configs:
    d, r = Dsc[k], Dsc[ref_of(k)]
    e_blk = np.stack([np.linalg.norm(d[:, Pn == n] - r[:, Pn == n], axis=1) for n in ORD], 1)   # N x orders
    r_blk = np.stack([np.linalg.norm(r[:, Pn == n], axis=1) for n in ORD], 1)
    rms_e = np.sqrt((e_blk ** 2).mean(0)); eta = rms_e / np.sqrt((r_blk ** 2).mean(0))
    snr = signal[ref_of(k)] / rms_e
    for n in ORD:
        rows.append([k, n, eta[n], rms_e[n], snr[n]] + [int(snr[n] >= t) for t in taus])
    summ[k] = {"eta": eta, "snr": snr,
               "full_err": np.linalg.norm(d - r, axis=1)}
with open(out / "per_order_error.csv", "w") as f:
    f.write("config,order,eta_rel,rms_abs_err,snr," + ",".join(f"trusted_tau{int(t)}" for t in taus) + "\n")
    for row in rows:
        f.write(",".join(str(x if not isinstance(x, float) else f"{x:.6g}") for x in row) + "\n")

# ---------------- retrieval diagnostics (LOO, cosine)
def loo(S):
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)[:, :N - 1]
    rel = labels[order] == labels[:, None]
    cum = np.cumsum(rel, 1)
    ap_ = (cum / np.arange(1, N) * rel).sum(1) / np.maximum(rel.sum(1), 1)
    return ap_.mean(), rel[:, 0].mean(), order[:, 0]

def certified_top1(Sref, eps):
    """Top-1 of the reference ranking provably unchanged if for every j:
       S(q,t)-S(q,j) > 2 eps_q + eps_t + eps_j   (|<a',b'>-<a,b>| <= |a'-a| + |b'-b| for unit vectors)."""
    S = Sref.copy(); np.fill_diagonal(S, -np.inf)
    t = np.argmax(S, 1)
    gap = S[np.arange(N), t][:, None] - S - eps[None, :]
    gap[np.arange(N), t] = np.inf; np.fill_diagonal(gap, np.inf)
    return (gap.min(1) > 2 * eps + eps[t]).mean()

deep = {}
for nm, p in [("ResNet18", base / "results_mpeg7_unified_preprocessing/feature_cache/resnet18_mpeg7_unified.npy"),
              ("DINOv2", base / "results_mpeg7_dinov2_zernike_foundation/feature_cache/dinov2_vitb14_mpeg7_224.npy")]:
    if p.exists():
        x = np.load(p).astype(np.float64)[:N]; x /= np.linalg.norm(x, axis=1, keepdims=True); deep[nm] = x @ x.T

ret = {}
for k in ["REF_INS", "REF_CIR"] + configs:
    S = Dsc[k] @ Dsc[k].T
    mAP, nn1, top = loo(S)
    item = {"zernike_mAP": mAP, "zernike_1NN": nn1}
    for dn, SD in deep.items():
        item[f"fused_{dn}_w0.5_mAP"] = loo(0.5 * SD + 0.5 * S)[0]
    if not k.startswith("REF"):
        Sr = Dsc[ref_of(k)] @ Dsc[ref_of(k)].T
        item["top1_agree_with_ref"] = float(np.mean(top == ret[ref_of(k)]["_top"]))
        item["top1_certified"] = float(certified_top1(Sr, summ[k]["full_err"]))
    item["_top"] = top
    ret[k] = item

# ---------------- summary JSON / CSV
def contiguous(snr, t):
    bad = np.where(snr < t)[0]; return int(bad[0] - 1) if bad.size else int(ORD[-1])
summary = {"n_shapes": int(N),
           "inscribed_outside_disk_fraction": {"mean": float(extra[:, 0].mean()), "max": float(extra[:, 0].max()),
                                               "n_gt_1pct": int((extra[:, 0] > .01).sum()), "n_gt_5pct": int((extra[:, 0] > .05).sum())},
           "configs": {}}
hdr = ["config", "err_mean", "err_p95", "err_max"] + [f"Nstar_tau{int(t)}" for t in taus] + \
      [c for c in ret["REF_INS"] if not c.startswith("_")] + ["top1_agree_with_ref", "top1_certified"]
lines = [",".join(hdr)]
for k in ["REF_INS", "REF_CIR"] + configs:
    it = {c: v for c, v in ret[k].items() if not c.startswith("_")}
    if k in summ:
        fe = summ[k]["full_err"]
        it.update({"err_mean": fe.mean(), "err_p95": np.percentile(fe, 95), "err_max": fe.max()})
        it.update({f"Nstar_tau{int(t)}": contiguous(summ[k]["snr"], t) for t in taus})
    summary["configs"][k] = {c: float(v) for c, v in it.items()}
    lines.append(",".join([k] + [f"{it[c]:.5f}" if c in it else "" for c in hdr[1:]]))
(out / "summary.csv").write_text("\n".join(lines) + "\n")
(out / "summary.json").write_text(json.dumps(summary, indent=2))

# ---------------- figures
groups = [("LEG", "Legacy (baseline)"), ("INS_NN", "Inscribed, nearest"), ("INS_AR", "Inscribed, area"),
          ("CIR_NN", "Circumscribed, nearest"), ("CIR_AR", "Circumscribed, area")]
for key, ylab, fname, logy in [("eta", "relative block error  $\\eta_n$", "fig_eta_by_order.png", True),
                               ("snr", "SNR$_n$", "fig_snr_by_order.png", True)]:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, G in zip(axs, (128, 256, 512)):
        for g, lab in groups:
            ax.plot(ORD, summ[f"{g}_{G}"][key], marker="o", ms=2.5, lw=1.2, label=lab)
        if key == "snr":
            for t in taus: ax.axhline(t, color="gray", ls=":", lw=.8)
        ax.set_title(f"G = {G}"); ax.set_xlabel("radial order n")
        if logy: ax.set_yscale("log")
    axs[0].set_ylabel(ylab); axs[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / fname, dpi=200); plt.close(fig)

print((out / "summary.csv").read_text())
print("trusted contiguous orders N* (tau=10):",
      {k: summary["configs"][k]["Nstar_tau10"] for k in configs if "Nstar_tau10" in summary["configs"][k]})
print(f"outputs in {out}")
