"""Worker for phase8_classification.py (linear-SVM classification with the retrieval splits)."""
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

C_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
N_SEL = [32, 28, 24, 20, 16, 12, 8]          # ties -> larger N (listed first)
G = {}

def init(data):
    G.update(data)

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n == 0] = 1; return x / n

def svm(Xtr, ytr, C):
    clf = LinearSVC(C=C, dual=Xtr.shape[0] < Xtr.shape[1], max_iter=20000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Xtr, ytr)
    return clf

def select(cands, tr, va, y):
    """cands: list of (param, X). Chooses (param, C) by validation accuracy; ties -> first."""
    best = (-1.0, None, None, None)
    for prm, X in cands:
        for C in C_GRID:
            acc = accuracy_score(y[va], svm(X[tr], y[tr], C).predict(X[va]))
            if acc > best[0] + 1e-12:
                best = (acc, prm, C, X)
    return best

def final(X, C, trva, te, y, Xte_alt=None):
    clf = svm(X[trva], y[trva], C); p = clf.predict(X[te])
    out = dict(acc=accuracy_score(y[te], p), f1=f1_score(y[te], p, average="macro"))
    if Xte_alt is not None:
        pr = clf.predict(Xte_alt[te])
        out.update(rot_acc=accuracy_score(y[te], pr), rot_f1=f1_score(y[te], pr, average="macro"))
    return out

def zernike_cands(rep):
    ks = sorted(G["reps"][rep])
    Pn = G["Pn"]
    return [((k, N), l2n(G["reps"][rep][k][:, Pn <= N])) for k in ks for N in N_SEL]

def run_zernike(seed):
    """Zernike-only classification for every normalization; returns records and selected (k, N)."""
    tr, va, te = G["splits"][seed]; y = G["labels"]; trva = np.concatenate([tr, va]); recs = []; sel = {}
    for rep in G["reps"]:
        val, prm, C, X = select(zernike_cands(rep), tr, va, y)
        alt = None
        if rep in G.get("rot", {}):                         # rotation test: k fixed at the rotated scale
            kr = G["rot_k"][rep]; Pn = G["Pn"]
            cands = [((kr, N), l2n(G["reps"][rep][kr][:, Pn <= N])) for N in N_SEL]
            v2, p2, C2, X2 = select(cands, tr, va, y)
            Xr = l2n(G["rot"][rep][:, Pn <= p2[1]])
            r = final(X2, C2, trva, te, y, Xte_alt=Xr)
            recs.append(("-", seed, f"ROT:{rep}", f"k{p2[0]:g}/N{p2[1]}/C{C2:g}", v2, r["acc"], r["f1"], r["rot_acc"], r["rot_f1"]))
        r = final(X, C, trva, te, y)
        recs.append(("-", seed, rep, f"k{prm[0]:g}/N{prm[1]}/C{C:g}", val, r["acc"], r["f1"], np.nan, np.nan))
        sel[rep] = prm
    return seed, recs, sel

def run_deep(task):
    bb, seed, sel = task
    tr, va, te = G["splits"][seed]; y = G["labels"]; trva = np.concatenate([tr, va]); Pn = G["Pn"]
    D = l2n(np.load(G["deep"][bb]).astype(np.float64)); recs = []
    val, _, C, X = select([("deep", D)], tr, va, y); r = final(X, C, trva, te, y)
    recs.append((bb, seed, "Deep", f"C{C:g}", val, r["acc"], r["f1"], np.nan, np.nan))
    for rep in G["fuse"]:
        k, N = sel[rep]
        X = np.hstack([D, l2n(G["reps"][rep][k][:, Pn <= N])])
        val, _, C, X = select([("cat", X)], tr, va, y); r = final(X, C, trva, te, y)
        recs.append((bb, seed, f"Deep+{rep}", f"k{k:g}/N{N}/C{C:g}", val, r["acc"], r["f1"], np.nan, np.nan))
    return recs

def run_surface(task):
    """(k, N) accuracy surface for Deep + CEN-RG; C chosen on validation for every cell."""
    bb, seed, k = task
    tr, va, te = G["splits"][seed]; y = G["labels"]; trva = np.concatenate([tr, va]); Pn = G["Pn"]
    D = l2n(np.load(G["deep"][bb]).astype(np.float64)); out = []
    for N in range(6, 33, 2):
        X = np.hstack([D, l2n(G["reps"]["CEN-RG"][k][:, Pn <= N])])
        val, _, C, X = select([("cat", X)], tr, va, y); r = final(X, C, trva, te, y)
        out.append((seed, k, N, val, r["acc"]))
    return out

def run_surface_val(task):
    """Validation-only (k, N) surface for Deep + CEN-RG: for every cell, the best validation
    accuracy over C of a classifier trained on the training part. The test part is not used."""
    bb, seed, k = task
    tr, va, _ = G["splits"][seed]; y = G["labels"]; Pn = G["Pn"]
    D = l2n(np.load(G["deep"][bb]).astype(np.float64)); out = []
    for N in range(6, 33, 2):
        X = np.hstack([D, l2n(G["reps"]["CEN-RG"][k][:, Pn <= N])])
        val, _, C, _ = select([("cat", X)], tr, va, y)
        out.append((seed, k, N, val, C))
    return out
