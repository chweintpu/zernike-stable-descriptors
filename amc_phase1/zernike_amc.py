"""
zernike_amc.py -- Zernike moment engine for the AMC follow-up study.

Key design:
  * V*_nm(x,y) = (x - i y)^m * P_nm(t),  t = 2 r^2 - 1,
    where P_nm is a polynomial of degree J=(n-m)/2 in t, expanded EXACTLY
    (rational arithmetic) in Chebyshev polynomials T_j(t).
  * Moments of any weighted point set are then obtained from the
    matrix  M[m, j] = sum_p w_p (x_p - i y_p)^m T_j(t_p)   (one BLAS-3 product),
    followed by  Z_nm = (n+1)/pi * sum_j c_{nm,j} M[m, j].
  * No factorial-form cancellation: |T_j| <= 1, |x - i y|^m <= 1 on the disk,
    and the Chebyshev coefficients of P_nm are O(1).
"""
import math, re
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image

N_MAX = 32

# ------------------------------------------------------------------ indices
def valid_pairs(N=N_MAX):
    return [(n, m) for n in range(N + 1) for m in range(n + 1) if (n - m) % 2 == 0]

def _binom(a, b):
    return math.comb(a, b)

def _power_to_cheb(k):
    """t^k = sum_j coef_j T_j(t), exact."""
    out = {}
    for i in range(k // 2 + 1):
        j = k - 2 * i
        c = Fraction(_binom(k, i), 2 ** (k - 1)) if k > 0 else Fraction(1)
        if j == 0 and k > 0:
            c /= 2
        out[j] = out.get(j, Fraction(0)) + c
    return out

def cheb_coefficients(N=N_MAX):
    pairs = valid_pairs(N)
    J = N // 2
    C = np.zeros((len(pairs), J + 1))
    for p, (n, m) in enumerate(pairs):
        # R_nm(r) = sum_s a_s r^{n-2s} = r^m * sum_s a_s ((t+1)/2)^{(n-m)/2 - s}
        poly_t = {}  # power basis in t
        for s in range((n - m) // 2 + 1):
            a = Fraction((-1) ** s * math.factorial(n - s),
                         math.factorial(s) * math.factorial((n + m) // 2 - s)
                         * math.factorial((n - m) // 2 - s))
            k = (n - m) // 2 - s
            for i in range(k + 1):          # ((t+1)/2)^k expansion
                poly_t[i] = poly_t.get(i, Fraction(0)) + a * Fraction(_binom(k, i), 2 ** k)
        cheb = {}
        for i, c in poly_t.items():
            for j, d in _power_to_cheb(i).items():
                cheb[j] = cheb.get(j, Fraction(0)) + c * d
        for j, c in cheb.items():
            C[p, j] = float(c)
    return pairs, C

PAIRS, CHEB = cheb_coefficients(N_MAX)
P_N = np.array([n for n, _ in PAIRS])
P_M = np.array([m for _, m in PAIRS])
J_MAX = N_MAX // 2

def moments(x, y, w, chunk=150_000):
    """Complex Zernike moments Z_nm (len(PAIRS),) of the weighted point set.
    Points with r > 1 must already carry zero weight (they are skipped)."""
    x = np.asarray(x, np.float64).ravel(); y = np.asarray(y, np.float64).ravel()
    w = np.asarray(w, np.float64).ravel()
    keep = (w != 0) & (x * x + y * y <= 1.0 + 1e-12)
    x, y, w = x[keep], y[keep], w[keep]
    M = np.zeros((N_MAX + 1, J_MAX + 1), np.complex128)
    js = np.arange(J_MAX + 1)[:, None]
    for s in range(0, x.size, chunk):
        xc, yc, wc = x[s:s + chunk], y[s:s + chunk], w[s:s + chunk]
        zc = xc - 1j * yc
        A = np.empty((N_MAX + 1, xc.size), np.complex128)
        A[0] = wc
        for m in range(1, N_MAX + 1):
            A[m] = A[m - 1] * zc
        t = np.clip(2.0 * (xc * xc + yc * yc) - 1.0, -1.0, 1.0)
        T = np.cos(js * np.arccos(t)[None, :])
        M += A @ T.T
    return (P_N + 1) / np.pi * np.einsum("pj,pj->p", CHEB, M[P_M, :])

def descriptor(Z):
    """Rotation-invariant magnitude descriptor, L2-normalized (as in the baseline pipeline)."""
    d = np.abs(Z).astype(np.float64)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else d

# ---------------------------------------------------------- data & geometry
def list_mpeg7(root):
    """Windows-compatible (case-insensitive) ordering -> aligned with cached Deep features."""
    fs = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() == ".gif"
          and re.match(r"^(.*?)-(\d+)$", p.stem)]
    return sorted(fs, key=lambda p: str(p).lower())

def label_of(p):
    return re.match(r"^(.*?)-(\d+)$", p.stem).group(1).lower()

def load_foreground_mask(path):
    arr = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    mask = arr < 0.5 if float(np.median(border)) >= 0.5 else arr >= 0.5
    frac = float(mask.mean())
    if frac < 0.005 or frac > 0.95:
        dark, bright = arr < 0.5, arr >= 0.5
        mask = dark if dark.mean() < bright.mean() else bright
    return mask.astype(np.uint8)

def legacy_canvas(mask):
    """Exactly the baseline canvas: bbox crop, square, 10% margin, centred."""
    ys, xs = np.where(mask > 0)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    margin = max(2, int(round(side * 0.10)))
    S = side + 2 * margin
    canvas = np.zeros((S, S), np.uint8)
    yo, xo = (S - h) // 2, (S - w) // 2
    canvas[yo:yo + h, xo:xo + w] = crop
    return canvas

def mapping(canvas, kind):
    """Return (centre c, radius R): unit coords x = (u - c)/R, u = canvas column coord."""
    S = canvas.shape[0]
    c = S / 2.0
    if kind == "inscribed":
        return c, S / 2.0
    ii, jj = np.nonzero(canvas)
    dx = np.maximum(np.abs(jj - c), np.abs(jj + 1 - c))
    dy = np.maximum(np.abs(ii - c), np.abs(ii + 1 - c))
    return c, float(np.sqrt(dx * dx + dy * dy).max()) * (1 + 1e-9)

# ------------------------------------------------------------ reference
def gauss01(Q):
    g, wq = np.polynomial.legendre.leggauss(Q)
    return (g + 1) / 2, wq / 2

# ------------------------------------------------------------ test pipelines
def _lookup(canvas, u, v):
    S = canvas.shape[0]
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ok = (iu >= 0) & (iu < S) & (iv >= 0) & (iv < S)
    out = np.zeros(u.shape, np.float64)
    out[ok] = canvas[iv[ok], iu[ok]]
    return out

def grid_moments(canvas, kind, G, resample="nearest", sampling="center", ss=4):
    """Discrete pipeline.
    resample : 'nearest' (value at the output-pixel centre = PIL NEAREST)
               'area'    (foreground coverage of the output pixel, ss x ss subsamples)
    sampling : 'center'   basis evaluated at true pixel centres -1+(2k+1)/G
               'endpoint' basis evaluated at linspace(-1,1,G)  (baseline legacy pipeline)"""
    c, R = mapping(canvas, kind)
    ctr = -1.0 + (2 * np.arange(G) + 1) / G
    if resample == "nearest":
        X, Y = np.meshgrid(ctr, ctr)
        f = _lookup(canvas, c + X * R, c + Y * R)
    else:
        off = (-1.0 + (2 * np.arange(ss) + 1) / ss) / G      # sub-offsets inside a pixel
        f = np.zeros((G, G))
        for oy in off:
            for ox in off:
                X, Y = np.meshgrid(ctr + ox, ctr + oy)
                f += _lookup(canvas, c + X * R, c + Y * R)
        f /= ss * ss
    if sampling == "center":
        X, Y = np.meshgrid(ctr, ctr)
    else:
        ax = np.linspace(-1.0, 1.0, G)
        X, Y = np.meshgrid(ax, ax)
    w = f * (2.0 / G) ** 2
    w = np.where(X * X + Y * Y <= 1.0, w, 0.0)
    return moments(X, Y, w)

def reference_moments_split(canvas, kind, Q=4, Qb=32):
    """As reference_moments, but pixels crossing the unit circle (only possible for
    the inscribed mapping) are integrated with a much finer QbxQb rule."""
    c, R = mapping(canvas, kind)
    ii, jj = np.nonzero(canvas)
    # nearest and farthest distance of each pixel square to the centre (unit coords)
    x0, x1 = (jj - c) / R, (jj + 1 - c) / R
    y0, y1 = (ii - c) / R, (ii + 1 - c) / R
    nx = np.where((x0 <= 0) & (x1 >= 0), 0, np.minimum(abs(x0), abs(x1)))
    ny = np.where((y0 <= 0) & (y1 >= 0), 0, np.minimum(abs(y0), abs(y1)))
    fx, fy = np.maximum(abs(x0), abs(x1)), np.maximum(abs(y0), abs(y1))
    inside = fx**2 + fy**2 <= 1.0
    cross = (~inside) & (nx**2 + ny**2 < 1.0)
    Z = np.zeros(len(PAIRS), np.complex128)
    for sel, q in [(inside, Q), (cross, Qb)]:
        if not sel.any():
            continue
        g, wq = gauss01(q)
        ga, gb = np.meshgrid(g, g); wa, wb = np.meshgrid(wq, wq)
        ga, gb, ww = ga.ravel(), gb.ravel(), (wa * wb).ravel()
        I, Jc = ii[sel], jj[sel]
        step = max(1, 150_000 // (q * q))
        for s in range(0, I.size, step):
            i, j = I[s:s + step, None], Jc[s:s + step, None]
            x = ((j + ga[None]) - c) / R; y = ((i + gb[None]) - c) / R
            w = np.broadcast_to(ww[None] / (R * R), x.shape)
            Z += moments(x, y, w)
    return Z
