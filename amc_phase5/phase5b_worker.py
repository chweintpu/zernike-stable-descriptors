"""Worker for phase5b_eval.py: one (dataset, backbone) per process."""
import numpy as np

OMEGAS = [1.0, 0.75, 0.5, 0.25, 0.0]          # ties -> larger omega
N_GRID = list(range(32, 5, -1))                # ties -> larger N (baseline first)

def ap_mean(S, ql, gl):
    order = np.argsort(-S, axis=1, kind="stable")
    rel = gl[order] == ql[:, None]; cum = np.cumsum(rel, 1)
    return float((((cum / np.arange(1, S.shape[1] + 1)) * rel).sum(1) / np.maximum(rel.sum(1), 1)).mean())

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1; return x / n

def cka_feat(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return np.linalg.norm(Xc.T @ Yc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))

def run(task):
    ds, bb, deep_path, mags, labels, splits, Pn, Pm, k_grid = task
    labels = np.asarray(labels)
    Dp = l2n(np.load(deep_path).astype(np.float64))
    assert Dp.shape[0] == len(labels), f"{ds}/{bb}: Deep cache has {Dp.shape[0]} rows, labels {len(labels)}"
    recs, cka = [], []
    for p in ("LEG", "CEN_RG_k2"):
        Z = l2n(mags[p])
        for m in range(Pm.max() + 1):
            cka.append((ds, bb, p, f"m={m}", cka_feat(Dp, Z[:, Pm == m])))
        cka.append((ds, bb, p, "even m>0", cka_feat(Dp, Z[:, (Pm > 0) & (Pm % 2 == 0)])))
        cka.append((ds, bb, p, "odd m", cka_feat(Dp, Z[:, Pm % 2 == 1])))
    for seed, tr, va, te in splits:
        gal = np.concatenate([tr, va])
        lv, lt, lg, lte = labels[va], labels[tr], labels[gal], labels[te]
        SDv, SDt = Dp[va] @ Dp[tr].T, Dp[te] @ Dp[gal].T
        deep = ap_mean(SDt, lte, lg)

        def choose(cands):
            best = (-1.0, None, None, None)
            for prm, Z in cands:
                SZv = Z[va] @ Z[tr].T
                for w in OMEGAS:
                    m = ap_mean(w * SDv + (1 - w) * SZv, lv, lt)
                    if m > best[0] + 1e-12:
                        best = (m, prm, w, Z)
            val, prm, w, Z = best
            SZt = Z[te] @ Z[gal].T
            return dict(val=val, fused=ap_mean(w * SDt + (1 - w) * SZt, lte, lg),
                        z=ap_mean(SZt, lte, lg), w=w, prm=prm)

        res = {}
        for p, A in mags.items():
            res[f"{p}|full"] = choose([(32, l2n(A))])
            res[f"{p}|trunc"] = choose([(N, l2n(A[:, Pn <= N])) for N in N_GRID])
        # validation-selected k (k_grid is in tie-preference order)
        for kind in ("full", "trunc"):
            best = None
            for k in k_grid:
                r = res[f"CEN_RG_k{k:g}|{kind}"]
                if best is None or r["val"] > best[1]["val"] + 1e-12:
                    best = (k, r)
            res[f"CEN_RG_kval|{kind}"] = dict(best[1], prm=f"k{best[0]:g}/{best[1]['prm']}")
        for name, r in res.items():
            p, kind = name.split("|")
            recs.append((ds, bb, p, kind, seed, deep, r["z"], r["fused"], r["fused"] - deep, r["w"], str(r["prm"])))
    return recs, cka
