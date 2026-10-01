"""Phase 7d: controlled convergence test on manufactured shapes (off-centre disk, rotated ellipse,
axis-parallel rectangle) with reference moments computed to double precision by polar integration.
Spyder: F5.  Writes <base>/results_amc_phase7/manufactured/"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import argparse
from pathlib import Path
import numpy as np
import zernike_amc as za
from zernike_methods import radial_kintner

N = za.N_MAX
PAIRS = za.PAIRS

# ------------------------------------------------------------------ shapes (normalized coordinates)
class Ellipse:
    def __init__(self, cx, cy, a, b, phi):
        self.c = np.array([cx, cy]); self.a, self.b = a, b
        self.cp, self.sp = np.cos(phi), np.sin(phi)
    def _local(self, x, y):
        dx, dy = x - self.c[0], y - self.c[1]
        return dx * self.cp + dy * self.sp, -dx * self.sp + dy * self.cp
    def inside(self, x, y):
        u, v = self._local(x, y); return (u / self.a) ** 2 + (v / self.b) ** 2 <= 1
    def rho(self, th):                       # distance from the origin to the boundary along direction th
        ux, uy = np.cos(th), np.sin(th)
        du, dv = self._local(np.zeros_like(th), np.zeros_like(th))         # origin in local coords
        eu, ev = ux * self.cp + uy * self.sp, -ux * self.sp + uy * self.cp
        A = (eu / self.a) ** 2 + (ev / self.b) ** 2
        B = 2 * (du * eu / self.a ** 2 + dv * ev / self.b ** 2)
        C = (du / self.a) ** 2 + (dv / self.b) ** 2 - 1
        return (-B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    def y_interval(self, x):                 # vertical chord at abscissa x (vectorized)
        # solve for y: inside(x,y) is a quadratic inequality in y
        cp, sp, a, b = self.cp, self.sp, self.a, self.b
        dx = x - self.c[0]
        A = (sp / a) ** 2 + (cp / b) ** 2
        B = 2 * (dx * cp * sp / a ** 2 - dx * sp * cp / b ** 2)
        C = (dx * cp / a) ** 2 + (dx * sp / b) ** 2 - 1
        disc = B * B - 4 * A * C
        ok = disc > 0; sq = np.sqrt(np.where(ok, disc, 0))
        lo = np.where(ok, (-B - sq) / (2 * A), 0) + self.c[1]; hi = np.where(ok, (-B + sq) / (2 * A), 0) + self.c[1]
        return lo, hi, ok
    kinks = None

class Rect:
    def __init__(self, x0, x1, y0, y1):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
    def inside(self, x, y):
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)
    def rho(self, th):
        ux, uy = np.cos(th), np.sin(th)
        with np.errstate(divide="ignore"):
            tx = np.where(ux > 0, self.x1 / ux, np.where(ux < 0, self.x0 / ux, np.inf))
            ty = np.where(uy > 0, self.y1 / uy, np.where(uy < 0, self.y0 / uy, np.inf))
        return np.minimum(tx, ty)
    @property
    def kinks(self):
        return sorted(np.mod([np.arctan2(y, x) for x in (self.x0, self.x1) for y in (self.y0, self.y1)], 2 * np.pi))

SHAPES = {"disk": Ellipse(0.10, 0.05, 0.60, 0.60, 0.0),
          "ellipse": Ellipse(0.05, -0.03, 0.75, 0.40, np.pi / 6),
          "rectangle": Rect(-0.60, 0.50, -0.35, 0.45)}

# ------------------------------------------------------------------ reference moments
def reference(shape, n_theta=2048, n_r=24):
    gr, wr = np.polynomial.legendre.leggauss(n_r)
    if shape.kinks is None:                              # smooth periodic rho: trapezoid rule
        th = 2 * np.pi * np.arange(n_theta) / n_theta; wt = np.full(n_theta, 2 * np.pi / n_theta)
    else:                                                # split at corner angles, Gauss-Legendre per panel
        k = shape.kinks; edges = list(k) + [k[0] + 2 * np.pi]
        gt, wg = np.polynomial.legendre.leggauss(128); th, wt = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            th.append((b - a) / 2 * gt + (a + b) / 2); wt.append((b - a) / 2 * wg)
        th, wt = np.concatenate(th), np.concatenate(wt)
    rho = shape.rho(th)
    r = (rho[:, None] * (gr[None, :] + 1) / 2); w_r = (rho[:, None] / 2) * wr[None, :] * r   # r dr
    R = radial_kintner(N, r.ravel())
    Z = np.zeros(len(PAIRS), complex)
    for p, (n, m) in enumerate(PAIRS):
        F = (R[(n, m)].reshape(r.shape) * w_r).sum(1)                                      # int_0^rho R r dr
        Z[p] = (n + 1) / np.pi * np.sum(wt * F * np.exp(-1j * m * th))
    return Z

# ------------------------------------------------------------------ discrete moments
def coverage(shape, G, method, ss=8, nsub=32, ngl=8):
    h = 2.0 / G; c = -1 + (2 * np.arange(G) + 1) / G
    X, Y = np.meshgrid(c, c)
    if method == "nearest":
        return shape.inside(X, Y).astype(float)
    if method == "subsampled":
        off = (-1 + (2 * np.arange(ss) + 1) / ss) * h / 2; W = np.zeros((G, G))
        for oy in off:
            for ox in off:
                W += shape.inside(X + ox, Y + oy)
        return W / ss ** 2
    # exact coverage
    if isinstance(shape, Rect):
        e = -1 + 2 * np.arange(G + 1) / G
        ox = np.clip(np.minimum(e[1:], shape.x1) - np.maximum(e[:-1], shape.x0), 0, None) / h
        oy = np.clip(np.minimum(e[1:], shape.y1) - np.maximum(e[:-1], shape.y0), 0, None) / h
        return oy[:, None] * ox[None, :]
    W = np.zeros((G, G))
    corners = [shape.inside(X + sx * h / 2, Y + sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)]
    full = corners[0] & corners[1] & corners[2] & corners[3]          # convex shape: cell inside
    W[full] = 1.0
    near = (~full) & (np.hypot(X - shape.c[0], Y - shape.c[1]) <= max(shape.a, shape.b) + h)
    iy, ix = np.nonzero(near)
    g, wg = np.polynomial.legendre.leggauss(ngl)
    t = ((np.arange(nsub)[:, None] + (g[None, :] + 1) / 2) / nsub).ravel(); wt = np.tile(wg / 2 / nsub, nsub)
    for s in range(0, iy.size, 4000):
        yy0 = c[iy[s:s + 4000]] - h / 2; xx0 = c[ix[s:s + 4000]] - h / 2
        xs = xx0[:, None] + h * t[None, :]
        lo, hi, ok = shape.y_interval(xs)
        length = np.clip(np.minimum(hi, yy0[:, None] + h) - np.maximum(lo, yy0[:, None]), 0, None) * ok
        W[iy[s:s + 4000], ix[s:s + 4000]] = (length * wt[None, :]).sum(1) / h
    return W

def discrete(shape, G, method, ss=8):
    c = -1 + (2 * np.arange(G) + 1) / G; X, Y = np.meshgrid(c, c)
    W = coverage(shape, G, method, ss) * (2.0 / G) ** 2
    W = np.where(X * X + Y * Y <= 1, W, 0)
    return za.moments(X, Y, W)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    out = Path(a.base) / "results_amc_phase7" / "manufactured"; out.mkdir(parents=True, exist_ok=True)
    Gs = [64, 128, 256, 512]
    methods = [("nearest", "nearest", 1), ("subsampled s=4", "subsampled", 4), ("subsampled s=8", "subsampled", 8),
               ("exact coverage", "exact", 1)]
    desc = lambda Z: np.abs(Z) / np.linalg.norm(np.abs(Z))
    rows = ["shape,method,measure," + ",".join(f"G{G}" for G in Gs) + ",order_64_128,order_128_256,order_256_512"]
    for sname, shape in SHAPES.items():
        Zr = reference(shape); Zr2 = reference(shape, n_theta=4096, n_r=32)
        print(f"[{sname}] reference self-consistency: {np.abs(Zr - Zr2).max() / np.abs(Zr).max():.1e}", flush=True)
        for label, meth, ss in methods:
            e_mom, e_des = [], []
            for G in Gs:
                Z = discrete(shape, G, meth, ss)
                e_mom.append(np.linalg.norm(Z - Zr) / np.linalg.norm(Zr))
                e_des.append(np.linalg.norm(desc(Z) - desc(Zr)))
            for meas, e in (("moment rel. error", e_mom), ("descriptor error", e_des)):
                orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
                rows.append(f"{sname},{label},{meas}," + ",".join(f"{x:.3e}" for x in e) + "," +
                            ",".join(f"{o:.2f}" for o in orders))
            print(f"   {label}: moment errors " + ", ".join(f"{x:.2e}" for x in e_mom), flush=True)
    (out / "manufactured_convergence.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows)); print(f"outputs in {out}")

if __name__ == "__main__":
    main()
