"""Worker for phase4b_select.py: one backbone per process."""
from collections import defaultdict
import numpy as np

TRAIN, VAL = 12, 4
SEEDS = list(range(3001, 3011))    # independent split set for the AMC study
OMEGAS = [1.0, 0.75, 0.5, 0.25, 0.0]          # scanned in this order: ties -> larger omega
N_GRID = list(range(32, 3, -1))                 # truncation orders, ties -> larger N (= baseline first)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # power weights (n+1)^-alpha, ties -> alpha=0 first
M_GROUPS = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 16), (17, 32)]

def stratified_split(labels, seed):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[str(lab)].append(i)
    tr, va, te = [], [], []
    for lab in sorted(by):
        ids = np.asarray(by[lab], np.int64).copy(); rng.shuffle(ids)
        tr += list(ids[:TRAIN]); va += list(ids[TRAIN:TRAIN + VAL]); te += list(ids[TRAIN + VAL:])
    return np.array(tr), np.array(va), np.array(te)

def ap_mean(S, ql, gl):
    order = np.argsort(-S, axis=1, kind="stable")
    rel = gl[order] == ql[:, None]; cum = np.cumsum(rel, 1)
    return float((((cum / np.arange(1, S.shape[1] + 1)) * rel).sum(1) / np.maximum(rel.sum(1), 1)).mean())

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1; return x / n

def parseval_weights(Pn, Pm):
    """sqrt of the Zernike Parseval weights pi/(n+1); x sqrt(2) for m>0 (the -m twin of a real image)."""
    return np.sqrt(np.pi / (Pn + 1)) * np.where(Pm > 0, np.sqrt(2.0), 1.0)

def cka_feat(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    return np.linalg.norm(Xc.T @ Yc) ** 2 / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))

def run_backbone(task):
    bb, deep_path, mags, labels, Pn, Pm = task
    labels = np.asarray(labels)
    Dp = l2n(np.load(deep_path).astype(np.float64))
    recs, cka = [], []
    pw = parseval_weights(Pn, Pm)
    for p, A in mags.items():                     # CKA split by m = 0 / m > 0 within each order
        Z = l2n(A)
        for n in range(Pn.max() + 1):
            s0, s1 = (Pn == n) & (Pm == 0), (Pn == n) & (Pm > 0)
            if s0.any(): cka.append((bb, p, "m0", n, cka_feat(Dp, Z[:, s0])))
            if s1.any(): cka.append((bb, p, "mpos", n, cka_feat(Dp, Z[:, s1])))
    for seed in SEEDS:
        tr, va, te = stratified_split(labels, seed); gal = np.concatenate([tr, va])
        lv, lt, lg, lte = labels[va], labels[tr], labels[gal], labels[te]
        SDv, SDt = Dp[va] @ Dp[tr].T, Dp[te] @ Dp[gal].T
        deep = ap_mean(SDt, lte, lg)

        def choose(cands):
            """cands: list of (param, Z). Validation picks (param, omega); returns test record."""
            best = (-1, None, None, None)
            for prm, Z in cands:
                SZv = Z[va] @ Z[tr].T
                for w in OMEGAS:
                    m = ap_mean(w * SDv + (1 - w) * SZv, lv, lt)
                    if m > best[0] + 1e-12:
                        best = (m, prm, w, Z)
            _, prm, w, Z = best
            SZt = Z[te] @ Z[gal].T
            fused = ap_mean(w * SDt + (1 - w) * SZt, lte, lg)
            return fused, ap_mean(SZt, lte, lg), w, prm

        for p, A in mags.items():
            res = {
                "full": choose([(32, l2n(A))]),
                "trunc": choose([(N, l2n(A[:, Pn <= N])) for N in N_GRID]),
                "parseval": choose([("pw", l2n(A * pw))]),
                "power": choose([(al, l2n(A * (Pn + 1.0) ** -al)) for al in ALPHAS]),
                "parseval_trunc": choose([(N, l2n((A * pw)[:, Pn <= N])) for N in N_GRID]),
                "m0_only": choose([("m0", l2n(A[:, Pm == 0]))]),
                "no_m0": choose([("no_m0", l2n(A[:, Pm > 0]))]),
            }
            for g, (lo, hi) in enumerate(M_GROUPS):
                sel = (Pm >= lo) & (Pm <= hi)
                res[f"mgrp_only_{g}"] = choose([(g, l2n(A[:, sel]))])
                res[f"mgrp_out_{g}"] = choose([(g, l2n(A[:, ~sel]))])
            for meth, (fused, zap, w, prm) in res.items():
                recs.append((bb, p, meth, seed, deep, zap, fused, fused - deep, w, str(prm)))
    return recs, cka
