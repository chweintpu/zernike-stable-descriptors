"""Worker for phase3_spectrum.py: one backbone per process (importable -> works from Spyder)."""
from collections import defaultdict
import numpy as np

TRAIN, VAL = 12, 4
SEEDS = list(range(3001, 3011))    # independent split set for the AMC study
OMEGAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BANDS = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 23), (24, 27), (28, 32)]

def stratified_split(labels, seed):
    """Identical to the baseline split (12/4/4 per class, one RNG over sorted class names)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, lab in enumerate(labels):
        by[str(lab)].append(i)
    tr, va, te = [], [], []
    for lab in sorted(by):
        ids = np.asarray(by[lab], np.int64).copy(); rng.shuffle(ids)
        tr += list(ids[:TRAIN]); va += list(ids[TRAIN:TRAIN + VAL]); te += list(ids[TRAIN + VAL:])
    return np.array(tr), np.array(va), np.array(te)

def ap_rows(S, ql, gl):
    order = np.argsort(-S, axis=1, kind="stable")
    rel = gl[order] == ql[:, None]
    cum = np.cumsum(rel, 1)
    return ((cum / np.arange(1, S.shape[1] + 1)) * rel).sum(1) / np.maximum(rel.sum(1), 1)

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1; return x / n

def variants(Pn):
    """(type, id, column mask) of every descriptor variant."""
    out = [("full", -1, np.ones_like(Pn, bool))]
    out += [("cum", n, Pn <= n) for n in range(Pn.max() + 1)]
    out += [("band", b, (Pn >= lo) & (Pn <= hi)) for b, (lo, hi) in enumerate(BANDS)]
    out += [("leaveout", b, ~((Pn >= lo) & (Pn <= hi))) for b, (lo, hi) in enumerate(BANDS)]
    return out

def cka_feat(X, Y):
    """Linear CKA between feature matrices (rows = samples)."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    return num / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc))

def run_backbone(task):
    bb, deep_path, mags, labels, Pn = task          # mags: {pipeline: N x 289 magnitudes}
    Dp = l2n(np.load(deep_path).astype(np.float64))
    labels = np.asarray(labels)
    recs, cka = [], []
    # ---- unsupervised spectrum on all shapes: CKA(Deep, order block) with the globally
    #      normalised descriptor, so that K_Z = sum_n K_n exactly (order decomposition)
    for p, A in mags.items():
        Z = l2n(A)
        cka.append((bb, p, "full", -1, cka_feat(Dp, Z)))
        for n in range(Pn.max() + 1):
            cka.append((bb, p, "order", n, cka_feat(Dp, Z[:, Pn == n])))
            cka.append((bb, p, "cum", n, cka_feat(Dp, Z[:, Pn <= n])))
    # ---- retrieval spectrum under the baseline protocol
    V = variants(Pn)
    for seed in SEEDS:
        tr, va, te = stratified_split(labels, seed)
        gal = np.concatenate([tr, va])
        SDv, SDt = Dp[va] @ Dp[tr].T, Dp[te] @ Dp[gal].T
        deep_ap = ap_rows(SDt, labels[te], labels[gal]).mean()
        deep_nn_ok = labels[gal][np.argmax(SDt, 1)] == labels[te]
        for p, A in mags.items():
            for vt, vid, cols in V:
                Z = l2n(A[:, cols])
                SZv, SZt = Z[va] @ Z[tr].T, Z[te] @ Z[gal].T
                best, bw = -1, None
                for w in OMEGAS:                       # validation selection, ties -> larger omega
                    m = ap_rows(w * SDv + (1 - w) * SZv, labels[va], labels[tr]).mean()
                    if m >= best - 1e-12:
                        best, bw = m, w
                fused = ap_rows(bw * SDt + (1 - bw) * SZt, labels[te], labels[gal]).mean()
                z_ap = ap_rows(SZt, labels[te], labels[gal]).mean()
                z_nn_ok = labels[gal][np.argmax(SZt, 1)] == labels[te]
                rescue = (z_nn_ok & ~deep_nn_ok).sum() / max(1, (~deep_nn_ok).sum())
                recs.append((bb, p, vt, vid, seed, deep_ap, z_ap, fused, fused - deep_ap, bw, rescue))
    return recs, cka
