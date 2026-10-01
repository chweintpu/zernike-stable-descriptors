"""Worker for phase7b_eval.py: one (dataset, backbone) per process."""
import numpy as np

OMEGAS = [1.0, 0.75, 0.5, 0.25, 0.0]
N_GRID = list(range(32, 5, -1))            # ties -> larger N

def ap_mean(S, ql, gl):
    order = np.argsort(-S, axis=1, kind="stable")
    rel = gl[order] == ql[:, None]; cum = np.cumsum(rel, 1)
    return float((((cum / np.arange(1, S.shape[1] + 1)) * rel).sum(1) / np.maximum(rel.sum(1), 1)).mean())

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1; return x / n

def run(task):
    ds, bb, deep_path, mags, labels, splits, Pn, k_list = task
    labels = np.asarray(labels)
    Dp = l2n(np.load(deep_path).astype(np.float64))
    recs = []
    for seed, tr, va, te in splits:
        gal = np.concatenate([tr, va])
        lv, lt, lg, lte = labels[va], labels[tr], labels[gal], labels[te]
        SDv, SDt = Dp[va] @ Dp[tr].T, Dp[te] @ Dp[gal].T
        deep = ap_mean(SDt, lte, lg)

        def evaluate(Z):
            SZv = Z[va] @ Z[tr].T; best = (-1.0, None)
            for w in OMEGAS:
                m = ap_mean(w * SDv + (1 - w) * SZv, lv, lt)
                if m > best[0] + 1e-12:
                    best = (m, w)
            SZt = Z[te] @ Z[gal].T
            return dict(val=best[0], w=best[1], fused=ap_mean(best[1] * SDt + (1 - best[1]) * SZt, lte, lg),
                        z=ap_mean(SZt, lte, lg))

        res = {}
        for p, A in mags.items():
            cand = []
            for N in N_GRID:
                r = evaluate(l2n(A[:, Pn <= N])); r["prm"] = N; cand.append(r)
            res[(p, "full")] = cand[0]
            best = cand[0]
            for r in cand[1:]:
                if r["val"] > best["val"] + 1e-12:
                    best = r
            res[(p, "trunc")] = best
        for fam in ("CEN_AREA", "CEN_RG"):
            for kind in ("full", "trunc"):
                best = None
                for k in k_list:                     # k_list starts with 2 -> ties favour k = 2
                    r = res[(f"{fam}_k{k:g}", kind)]
                    if best is None or r["val"] > best[1]["val"] + 1e-12:
                        best = (k, r)
                res[(f"{fam}_kval", kind)] = dict(best[1], prm=f"k{best[0]:g}/{best[1]['prm']}")
        for (p, kind), r in res.items():
            recs.append((ds, bb, p, kind, seed, deep, r["z"], r["fused"], r["w"], str(r["prm"])))
    return recs
