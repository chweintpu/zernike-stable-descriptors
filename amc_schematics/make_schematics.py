"""Schematic figures for the manuscript (candidate Figures A, B, C).
All quantities are computed from their definitions in the manuscript on a fine raster.
Requires numpy, scipy, matplotlib.  Spyder: F5.  Output: PNG (300 dpi) and PDF in ./output"""
from pathlib import Path
import numpy as np
from scipy.special import j0, j1
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

OUT = Path(__file__).resolve().parent / "output"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "savefig.dpi": 300})
C_SHAPE, C_BOX, C_RG, C_MAX, C_AREA, C_CLIP = "#8fb3d9", "#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#d7301f"

# ------------------------------------------------------------------ test shape on a fine grid
def grid(half=6.0, n=1201):
    t = np.linspace(-half, half, n); X, Y = np.meshgrid(t, t); return X, Y, t[1] - t[0]

def shape_mask(X, Y, angle=0.0, blob=None):
    """elongated shape: a capsule (length 6, width 1) with an asymmetric side protrusion"""
    ca, sa = np.cos(-angle), np.sin(-angle); x = ca * X - sa * Y; y = sa * X + ca * Y
    capsule = (np.abs(x) <= 2.5) & (np.abs(y) <= 0.5) | ((x - 2.5) ** 2 + y ** 2 <= 0.25) | ((x + 2.5) ** 2 + y ** 2 <= 0.25)
    bump = ((x - 1.2) ** 2 / 0.6 ** 2 + (y - 0.7) ** 2 / 0.55 ** 2) <= 1
    m = capsule | bump
    if blob is not None:
        bx, by, br = blob; m |= (X - bx) ** 2 + (Y - by) ** 2 <= br ** 2
    return m

def stats(m, X, Y, dA):
    A = m.sum() * dA; cx, cy = X[m].mean(), Y[m].mean()
    rg = np.sqrt(np.mean((X[m] - cx) ** 2 + (Y[m] - cy) ** 2)); rmax = np.sqrt((X[m] - cx) ** 2 + (Y[m] - cy) ** 2).max()
    return A, cx, cy, rg, rmax

def draw_shape(ax, m, X, Y, color=C_SHAPE, alpha=1.0, edge="k", lw=0.8):
    ax.contourf(X, Y, m.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha)
    ax.contour(X, Y, m.astype(float), levels=[0.5], colors=[edge], linewidths=lw)

def bbox(m, X, Y):
    return X[m].min(), X[m].max(), Y[m].min(), Y[m].max()

def square_canvas(m, X, Y, margin=0.10):
    """BOX-INS: bounding box, square canvas with 10% margin, inscribed disk"""
    x0, x1, y0, y1 = bbox(m, X, Y); side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; return cx, cy, side

def label(ax, text, y=-0.04):
    """panel label and description below the panel"""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)

def finish(ax, lim):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")

# ================================================================== Figure A
def figure_A():
    X, Y, h = grid(7.0, 1401); dA = h * h
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    # (a) rotation: bounding-box canvas changes, centroid and r_g circle rotate with the shape
    ax = axs[0]
    for ang, alpha, ls in ((0.0, 0.35, "--"), (np.deg2rad(40), 1.0, "-")):
        m = shape_mask(X, Y, ang); A, cx, cy, rg, _ = stats(m, X, Y, dA)
        draw_shape(ax, m, X, Y, alpha=alpha, edge="k" if alpha == 1 else "0.5")
        bx, by, side = square_canvas(m, X, Y)
        ax.add_patch(Rectangle((bx - side / 2, by - side / 2), side, side, fill=False, ec=C_BOX, ls=ls, lw=1.2))
        ax.add_patch(Circle((bx, by), side / 2, fill=False, ec=C_BOX, ls=ls, lw=1.0))
        ax.plot(bx, by, "s", color=C_BOX, ms=3.5)
    ax.add_patch(Circle((cx, cy), 2 * rg, fill=False, ec=C_RG, lw=1.6)); ax.plot(cx, cy, "o", color=C_RG, ms=4)
    finish(ax, 6.2); label(ax, "(a) Rotation: bounding-box canvas vs\ncentroid and radius of gyration")
    # (b) remote blob: maximal radius jumps, radius of gyration barely changes
    ax = axs[1]
    m0 = shape_mask(X, Y, 0.0); A, cx, cy, rg0, rmax0 = stats(m0, X, Y, dA)
    blob = (cx - 2 * rmax0 * np.cos(np.deg2rad(20)), cy + 2 * rmax0 * np.sin(np.deg2rad(20)), 0.07)
    m1 = shape_mask(X, Y, 0.0, blob); _, cx1, cy1, rg1, rmax1 = stats(m1, X, Y, dA)
    draw_shape(ax, m1, X, Y)
    ax.add_patch(Circle((cx, cy), rmax0, fill=False, ec=C_MAX, ls="--", lw=1.2))
    ax.add_patch(Circle((cx1, cy1), rmax1, fill=False, ec=C_MAX, lw=1.2))
    ax.add_patch(Circle((cx, cy), rg0, fill=False, ec=C_RG, ls="--", lw=1.2))
    ax.add_patch(Circle((cx1, cy1), rg1, fill=False, ec=C_RG, lw=1.6))
    ax.annotate("remote blob", xy=blob[:2], xytext=(blob[0] + 0.3, blob[1] + 1.6), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    finish(ax, 8.2); label(ax, f"(b) Remote blob: $R_{{\\max}}$ +{100*(rmax1/rmax0-1):.0f}%,\n$r_g$ +{100*(rg1/rg0-1):.1f}%")
    # (c), (d) normalization into the unit disk at k = 2: area scale clips, r_g scale does not
    m = shape_mask(X, Y, np.deg2rad(20)); A, cx, cy, rg, _ = stats(m, X, Y, dA)
    for ax, (R, name, col) in zip(axs[2:], ((2 * np.sqrt(A / (2 * np.pi)), "(c) Area scale, $R=2\\sqrt{|\\Omega|/2\\pi}$", C_AREA),
                                              (2 * rg, "(d) Radius of gyration, $R=2r_g$", C_RG))):
        Xn, Yn = (X - cx) / R, (Y - cy) / R
        inside = m & (Xn ** 2 + Yn ** 2 <= 1); outside = m & (Xn ** 2 + Yn ** 2 > 1)
        draw_shape(ax, m, Xn, Yn, edge="k")
        if outside.any():
            ax.contourf(Xn, Yn, outside.astype(float), levels=[0.5, 1.5], colors=[C_CLIP], alpha=0.85)
        ax.add_patch(Circle((0, 0), 1, fill=False, ec=col, lw=1.8))
        frac = outside.sum() / m.sum()
        finish(ax, 1.9 if frac > 0 else 1.25); label(ax, f"{name}\nclipped fraction {100*frac:.0f}%")
    handles = [plt.Line2D([], [], color=C_BOX, lw=1.2, label="bounding-box canvas and inscribed disk"),
               plt.Line2D([], [], color=C_MAX, lw=1.2, label="maximal radius $R_{\\max}$"),
               plt.Line2D([], [], color=C_RG, lw=1.6, label="radius of gyration: circle of radius $r_g$ in (b), $2r_g$ in (a) and (d)"),
               plt.Line2D([], [], color="0.3", ls="--", lw=1.0, label="before rotation or perturbation"),
               plt.Line2D([], [], color=C_AREA, lw=1.8, label="area-scale disk"), Rectangle((0, 0), 1, 1, fc=C_CLIP, label="clipped foreground")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.22, 1, 1)); fig.savefig(OUT / "FigA_normalization.png", bbox_inches="tight"); fig.savefig(OUT / "FigA_normalization.pdf", bbox_inches="tight"); plt.close(fig)
    print("Figure A written")

# ================================================================== Figure B
def fourier_bessel(m, X, Y, dA, nus, mm=0):
    """nu*|B_m(nu)| of the shape in radius-of-gyration units, m = 0 or 1"""
    A, cx, cy, rg, _ = stats(m, X, Y, dA)
    vx, vy = (X[m] - cx) / rg, (Y[m] - cy) / rg; r = np.hypot(vx, vy); th = np.arctan2(vy, vx); dv = dA / rg ** 2
    J = j0 if mm == 0 else j1
    return np.array([nu * abs(np.sum(J(nu * r) * np.exp(-1j * mm * th)) * dv) for nu in nus])

def figure_B():
    X, Y, h = grid(5.0, 801); dA = h * h; m = shape_mask(X, Y, 0.0)
    nus = np.linspace(0.0, 12.0, 481); F = fourier_bessel(m, X, Y, dA, nus); F /= F.max()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]; ax.plot(nus, F, color="0.25", lw=1.3, label="$\\nu\\,|B_0(\\nu)|$ (normalized)")
    for (k, N, col, mk) in ((2, 15, "#1b9e77", "o"), (4, 31, "#d95f02", "s")):
        nn = np.arange(0, N + 1, 2); nu = (nn + 1) / k
        ax.plot(nu, np.interp(nu, nus, F), mk, color=col, ms=5, label=f"$(k,N)=({k},{N})$: $\\nu_n=(n+1)/k$, step $2/k={2/k:g}$")
    ax.axvspan(0, 8, color="0.92", zorder=0); ax.axvline(8, color="k", ls=":", lw=1)
    ax.text(8.15, 0.92, "bandwidth\n$\\Lambda=(N+1)/k=8$", fontsize=8, va="top")
    ax.set_xlabel("radial frequency $\\nu$"); ax.set_ylabel("spectral magnitude"); ax.set_xlim(0, 12); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.72)); label(ax, "(a) Zernike magnitudes as samples of the Fourier–Bessel spectrum ($m=0$)", -0.16)
    ax = axs[1]; ks = np.linspace(1.0, 6.0, 100)
    for L, ls in ((6, "--"), (8, "-"), (10, "--")):
        ax.plot(ks, L * ks - 1, ls, color="0.35", lw=1.0)
        kk = (34 + 1) / L; ax.text(min(kk, 5.3) + 0.08, min(L * min(kk, 5.3) - 1, 34) - 1.6, f"$\\Lambda={L}$", fontsize=8)
    ax.plot(2, 15, "o", color="#1b9e77", ms=7, zorder=3); ax.plot(4, 31, "s", color="#d95f02", ms=7, zorder=3)
    ax.annotate("", xy=(3.93, 30.4), xytext=(2.07, 15.6), arrowprops=dict(arrowstyle="->", lw=1.0, color="k"))
    ax.text(3.15, 21.5, "same bandwidth,\nfiner sampling", fontsize=7.5, rotation=0)
    ax.annotate("", xy=(3.0, 15), xytext=(2.1, 15), arrowprops=dict(arrowstyle="->", lw=0.9, color="0.4"))
    ax.text(2.25, 12.6, "larger $k$: lower $\\Lambda$", fontsize=7, color="0.3")
    ax.annotate("", xy=(2, 22.5), xytext=(2, 16), arrowprops=dict(arrowstyle="->", lw=0.9, color="0.4"))
    ax.text(1.12, 24.0, "larger $N$:\nhigher $\\Lambda$", fontsize=7, color="0.3")
    ax.set_xlim(1.0, 6.0); ax.set_ylim(0, 35); ax.set_xlabel("scale $k$"); ax.set_ylabel("maximal order $N$")
    label(ax, "(b) Lines of constant bandwidth $\\Lambda=(N+1)/k$", -0.16)
    fig.tight_layout(); fig.savefig(OUT / "FigB_bandwidth.png", bbox_inches="tight"); fig.savefig(OUT / "FigB_bandwidth.pdf", bbox_inches="tight"); plt.close(fig)
    print("Figure B written")

# ================================================================== Figure C
def figure_C():
    # native raster of a smooth curve (disk of radius 9.3 native pixels, centre off the lattice)
    nat = 1.0; S = 40; c0 = (-3.4, -4.1); R0 = 23.0
    I, J = np.meshgrid(np.arange(S), np.arange(S)); px = I + 0.5; py = J + 0.5
    raster = (px - 5.7 - c0[0]) ** 2 + (py - 5.2 - c0[1]) ** 2 <= R0 ** 2
    # computation grid: cell size h = 4 native pixels, offset from the native lattice
    h = 4.0; off = (1.3, 0.6); nx = int((S - off[0]) // h); ny = int((S - off[1]) // h)
    def cover_exact(x0, y0):
        xs = np.arange(int(np.floor(x0)), int(np.ceil(x0 + h))); ys = np.arange(int(np.floor(y0)), int(np.ceil(y0 + h))); a = 0.0
        for i in xs:
            for j in ys:
                if 0 <= i < S and 0 <= j < S and raster[j, i]:
                    a += max(0, min(i + 1, x0 + h) - max(i, x0)) * max(0, min(j + 1, y0 + h) - max(j, y0))
        return a / h ** 2
    def inside(x, y):
        i, j = int(np.floor(x)), int(np.floor(y)); return 0 <= i < S and 0 <= j < S and raster[j, i]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    titles = ("(a) Nearest neighbor: $\\hat w=\\mathbf{1}_\\Omega(\\mathbf{x}_p)$", "(b) Subsampled coverage, $s=4$",
              "(c) Exact coverage: $\\hat w=|Q_p\\cap\\Omega|/h^2$")
    for ax, mode, ttl in zip(axs, ("nn", "ss", "ex"), titles):
        # native pixels
        for i in range(S):
            for j in range(S):
                if raster[j, i]:
                    ax.add_patch(Rectangle((i, j), 1, 1, fc=C_SHAPE, ec="none"))
        for a in range(nx):
            for b in range(ny):
                x0, y0 = off[0] + a * h, off[1] + b * h; ex = cover_exact(x0, y0)
                if mode == "nn": w = float(inside(x0 + h / 2, y0 + h / 2))
                elif mode == "ss":
                    g = x0 + (np.arange(4) + 0.5) * h / 4; gy = y0 + (np.arange(4) + 0.5) * h / 4
                    w = np.mean([inside(u, v) for u in g for v in gy])
                else: w = ex
                boundary = 0 < ex < 1
                if w > 0:
                    ax.add_patch(Rectangle((x0, y0), h, h, fc="k", alpha=0.35 * w, ec="none"))
                ax.add_patch(Rectangle((x0, y0), h, h, fill=False, ec="#d95f02" if boundary else "0.55", lw=1.3 if boundary else 0.5))
                if mode == "nn": ax.plot(x0 + h / 2, y0 + h / 2, ".", color="k", ms=3)
                if mode == "ss" and boundary:
                    for u in g:
                        for v in gy: ax.plot(u, v, ".", color="k", ms=1.6)
                if boundary and mode != "nn":
                    ax.text(x0 + h / 2, y0 + h / 2, f"{w:.2f}", ha="center", va="center", fontsize=5.5, color="k")
        ax.set_xlim(off[0], off[0] + nx * h); ax.set_ylim(off[1], off[1] + ny * h); ax.set_aspect("equal"); ax.axis("off"); label(ax, ttl, -0.03)
    handles = [Rectangle((0, 0), 1, 1, fc=C_SHAPE, label="native pixels of the silhouette"),
               Rectangle((0, 0), 1, 1, fill=False, ec="#d95f02", lw=1.3, label="boundary cells of the computation grid"),
               Rectangle((0, 0), 1, 1, fc="k", alpha=0.35, label="estimated coverage $\\hat w$ (shading)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(OUT / "FigC_discretization.png", bbox_inches="tight"); fig.savefig(OUT / "FigC_discretization.pdf", bbox_inches="tight"); plt.close(fig)
    print("Figure C written")

if __name__ == "__main__":
    figure_A(); figure_B(); figure_C()
    print("output in", OUT)
