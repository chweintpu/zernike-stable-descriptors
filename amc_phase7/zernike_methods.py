"""Four ways to evaluate Zernike radial polynomials / all moments up to order N:
factorial form, Kintner's recurrence in n, the q-recursive method (Chong et al. 2003),
and the batched Chebyshev representation used in this paper."""
import math
from fractions import Fraction
import numpy as np

def pairs(N):
    return [(n, m) for n in range(N + 1) for m in range(n + 1) if (n - m) % 2 == 0]

# ---------------------------------------------------------------- radial polynomials
def radial_factorial(N, r):
    R = {}
    for n, m in pairs(N):
        s_ = np.zeros_like(r)
        for s in range((n - m) // 2 + 1):
            c = (-1) ** s * math.factorial(n - s) / (math.factorial(s) * math.factorial((n + m) // 2 - s)
                                                     * math.factorial((n - m) // 2 - s))
            s_ = s_ + c * r ** (n - 2 * s)
        R[(n, m)] = s_
    return R

def radial_kintner(N, r):
    R = {}
    for m in range(N + 1):
        R[(m, m)] = r ** m
        if m + 2 <= N:
            R[(m + 2, m)] = (m + 2) * r ** (m + 2) - (m + 1) * r ** m
        for n in range(m + 4, N + 1, 2):
            K1 = (n + m) * (n - m) * (n - 2) / 2; K2 = 2 * n * (n - 1) * (n - 2)
            K3 = -m * m * (n - 1) - n * (n - 1) * (n - 2); K4 = -n * (n + m - 2) * (n - m - 2) / 2
            R[(n, m)] = ((K2 * r * r + K3) * R[(n - 2, m)] + K4 * R[(n - 4, m)]) / K1
    return R

def radial_qrec(N, r):
    """requires r > 0"""
    R = {}
    for n in range(N + 1):
        R[(n, n)] = r ** n
        if n >= 2:
            R[(n, n - 2)] = n * r ** n - (n - 1) * r ** (n - 2)
        for m in range(n, 3, -2):
            H3 = -4 * (m - 2) * (m - 3) / ((n + m - 2) * (n - m + 4))
            H2 = H3 * (n + m) * (n - m + 2) / (4 * (m - 1)) + (m - 2)
            H1 = m * (m - 1) / 2 - m * H2 + H3 * (n + m + 2) * (n - m) / 8
            R[(n, m - 4)] = H1 * R[(n, m)] + (H2 + H3 / (r * r)) * R[(n, m - 2)]
    return R

_CHEB = {}
def cheb_coeffs(N):
    if N in _CHEB:
        return _CHEB[N]
    P = pairs(N); J = N // 2
    def p2c(k):
        out = {}
        for i in range(k // 2 + 1):
            j = k - 2 * i
            c = Fraction(math.comb(k, i), 2 ** (k - 1)) if k > 0 else Fraction(1)
            if j == 0 and k > 0:
                c /= 2
            out[j] = out.get(j, 0) + c
        return out
    P2C = [p2c(k) for k in range(J + 1)]
    C = np.zeros((len(P), J + 1))
    for p, (n, m) in enumerate(P):
        poly = {}
        for s in range((n - m) // 2 + 1):
            a = Fraction((-1) ** s * math.factorial(n - s), math.factorial(s) * math.factorial((n + m) // 2 - s)
                         * math.factorial((n - m) // 2 - s))
            k = (n - m) // 2 - s
            for i in range(k + 1):
                poly[i] = poly.get(i, 0) + a * Fraction(math.comb(k, i), 2 ** k)
        cheb = {}
        for i, c in poly.items():
            for j, d in P2C[i].items():
                cheb[j] = cheb.get(j, 0) + c * d
        for j, c in cheb.items():
            C[p, j] = float(c)
    _CHEB[N] = (P, C)
    return _CHEB[N]

def radial_chebyshev(N, r):
    P, C = cheb_coeffs(N)
    t = np.clip(2 * r * r - 1, -1, 1)
    T = np.cos(np.arange(N // 2 + 1)[:, None] * np.arccos(t)[None, :])
    return {(n, m): r ** m * (C[p] @ T) for p, (n, m) in enumerate(P)}

RADIAL = {"factorial": radial_factorial, "Kintner": radial_kintner, "q-recursive": radial_qrec,
          "Chebyshev": radial_chebyshev}

# ---------------------------------------------------------------- all moments of a point set
def moments_via_radial(method, N, x, y, w, chunk=20000):
    P = pairs(N); Z = np.zeros(len(P), complex); f = RADIAL[method]
    for s in range(0, x.size, chunk):
        xc, yc, wc = x[s:s + chunk], y[s:s + chunk], w[s:s + chunk]
        r = np.hypot(xc, yc); th = np.arctan2(yc, xc)
        R = f(N, r)
        for p, (n, m) in enumerate(P):
            Z[p] += (n + 1) / np.pi * np.dot(R[(n, m)] * wc, np.exp(-1j * m * th))
    return Z

def moments_chebyshev_batched(N, x, y, w, chunk=150000):
    """The batched form (6): one complex matrix product per chunk."""
    P, C = cheb_coeffs(N); J = N // 2
    Pn = np.array([n for n, _ in P]); Pm = np.array([m for _, m in P])
    M = np.zeros((N + 1, J + 1), complex); js = np.arange(J + 1)[:, None]
    for s in range(0, x.size, chunk):
        xc, yc, wc = x[s:s + chunk], y[s:s + chunk], w[s:s + chunk]
        zc = xc - 1j * yc
        A = np.empty((N + 1, xc.size), complex); A[0] = wc
        for m in range(1, N + 1):
            A[m] = A[m - 1] * zc
        T = np.cos(js * np.arccos(np.clip(2 * (xc * xc + yc * yc) - 1, -1, 1))[None, :])
        M += A @ T.T
    return (Pn + 1) / np.pi * np.einsum("pj,pj->p", C, M[Pm, :])

def moments_kintner_batched(N, x, y, w, chunk=50000):
    """Kintner's recurrence evaluated for all points at once (vectorized over points), with the
    angular accumulation done as one matrix-vector product per angular index m."""
    P = pairs(N); idx = {pm: i for i, pm in enumerate(P)}; Z = np.zeros(len(P), complex)
    for s in range(0, x.size, chunk):
        xc, yc, wc = x[s:s + chunk], y[s:s + chunk], w[s:s + chunk]
        r2 = xc * xc + yc * yc; zc = xc - 1j * yc
        rm = np.ones_like(r2); zm = wc.astype(complex)          # r^m and w (x - i y)^m / r^m ... built below
        E = wc.astype(complex)                                   # w * (x - i y)^m, updated with m
        for m in range(N + 1):
            if m > 0:
                E = E * zc
            # radial part divided by r^m: Q_n^m(r^2), via Kintner's recurrence on R/r^m
            rows = []
            q2 = np.ones_like(r2)                                # R_m^m / r^m
            rows.append(q2)
            if m + 2 <= N:
                q1 = (m + 2) * r2 - (m + 1)                      # R_{m+2}^m / r^m
                rows.append(q1)
                for n in range(m + 4, N + 1, 2):
                    K1 = (n + m) * (n - m) * (n - 2) / 2; K2 = 2 * n * (n - 1) * (n - 2)
                    K3 = -m * m * (n - 1) - n * (n - 1) * (n - 2); K4 = -n * (n + m - 2) * (n - m - 2) / 2
                    q = ((K2 * r2 + K3) * q1 + K4 * q2) / K1
                    rows.append(q); q2, q1 = q1, q
            Qm = np.vstack(rows)                                 # (#n, points)
            vals = Qm @ E                                        # sum_p Q_n^m(p) w_p (x_p - i y_p)^m
            for j, n in enumerate(range(m, N + 1, 2)):
                Z[idx[(n, m)]] += (n + 1) / np.pi * vals[j]
    return Z
