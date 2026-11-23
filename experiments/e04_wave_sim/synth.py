"""E04: synthetic laser-spot frames whose speckle has the statistics the wave picture demands.

What reaches a camera pixel is a sum of very many path fields, so the field is a Gaussian random field and is fully
described by its covariance. The transport Monte Carlo gives that covariance: for the light leaving the surface at x
it gives how much there is, with what polarization, after what path length. Here the object-plane field is drawn with
exactly that covariance,

    E(x; exposure) = sum over path nodes f of  L_f(x) g_f(x) exp(i dk n s_f)      (plus the surface term at s = 0),

with g_f independent white complex Gaussian fields (the object's microstructure), L_f the 2 x 2 Cholesky factor of the
polarization coherency W_f(x), and the phase exp(i dk n s_f) for each laser line or wavelength step dk. It is then
passed through the lens pupil (which makes the speckle grains), detected as intensity summed over polarizations and
laser lines, integrated on pixels and given shot noise, read noise and a 12-bit converter. Nothing about speckle
contrast is put in by hand: it comes out of the path statistics.

Units: millimetres. The kernel's lengths are in mean free paths and are scaled by the material's transport length.
"""
import numpy as np
import scipy.fft as sfft

import theory as th

WORKERS = 6
LAM = 635e-6


class Kernel:
    def __init__(self, k):
        self.re, self.se = k["rho_edges"], k["s_edges"]
        H = k["H"]
        self.n_phi = H.shape[2]
        self.dphi = (np.pi / 2) / self.n_phi
        self.rc = np.sqrt(self.re[1:] * self.re[:-1])
        self.sc = np.sqrt(self.se[1:] * self.se[:-1])
        area = 2 * self.dphi * (self.re[1:] ** 2 - self.re[:-1] ** 2)       # the four folded sectors, in ell^2
        self.dens = H[:, 1:] / area[1:, None, None]                          # continuous part, per ell^2
        self.delta = H[:, 0].sum(1)                                          # mass at the entry point (single scattering)
        self.mass_s = (H[0] + H[1]).sum(1)                                   # (rho, s)
        self.dlr = np.log(self.re[1] / self.re[0])


class Rig:
    def __init__(self, N=320, dx=0.025, pix=2, grain=0.15, w=1.0, lam=LAM, n_index=1.4):
        self.N, self.dx, self.pix, self.grain, self.w, self.lam, self.n = N, dx, pix, grain, w, lam, n_index
        self.k = 2 * np.pi / lam
        self.x = (np.arange(N) - N / 2 + 0.5) * dx
        self.X, self.Y = np.meshgrid(self.x, self.x)
        f = sfft.fftfreq(N, dx)
        FX, FY = np.meshgrid(f, f)
        self.na = 1.22 * lam / grain                                         # Airy first-zero diameter = grain
        self.pupil = (np.hypot(FX, FY) <= 1.22 / grain).astype(np.float32)
        self.pupil /= np.sqrt(self.pupil.mean())                             # so the image keeps the object's mean brightness
        psf2 = np.abs(sfft.ifft2(self.pupil)) ** 2
        self.otf = (sfft.fft2(psf2) / psf2.sum()).real
        self.xp = self.x.reshape(-1, pix).mean(1)                            # camera pixel centres

    def beam(self, c=(0.0, 0.0), Rc=None):
        """beam field on the object, unit power, 1/e^2 intensity radius w, wavefront radius Rc (None = flat)"""
        r2 = (self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2
        A = np.sqrt(2 / (np.pi * self.w ** 2)) * np.exp(-r2 / self.w ** 2)
        if Rc is None:
            return A.astype(complex)
        return A * np.exp(1j * self.k * r2 / (2 * Rc))

    def blur(self, a):
        return sfft.ifft2(sfft.fft2(a, workers=WORKERS) * self.otf, workers=WORKERS)

    def image(self, E):
        """object-plane field (2, N, N) to image-plane intensity summed over the two polarizations"""
        U = sfft.ifft2(sfft.fft2(E, axes=(-2, -1), workers=WORKERS) * self.pupil, axes=(-2, -1), workers=WORKERS)
        return (np.abs(U) ** 2).sum(0)

    def to_pixels(self, I):
        p, n = self.pix, self.N // self.pix
        return I.reshape(n, p, n, p).mean((1, 3))


class Material:
    def __init__(self, name, ell=None, mua=0.0, f_surf=0.0):
        self.name, self.ell, self.mua, self.f_surf = name, ell, mua, (1.0 if ell is None else f_surf)


def cnormal(shape, rng):
    return ((rng.standard_normal(shape, dtype=np.float32) + 1j * rng.standard_normal(shape, dtype=np.float32))
            * np.float32(np.sqrt(0.5)))


class KernelMaps:
    """the transport kernel of one material on the grid of offsets between object points"""

    def __init__(self, kern, rig, mat, rng, keep=0.999, sub=5, core=4):
        self.kern, self.rig, self.mat = kern, rig, mat
        N, dx, ell = rig.N, rig.dx, mat.ell
        P = 2 * N
        self.P = P
        off = (np.arange(P) - P // 2) * dx
        OX, OY = np.meshgrid(off, off)
        self.idx = self._index(OX, OY)
        c0 = P // 2
        sl = slice(c0 - core, c0 + core + 1)
        self.core = (sl, sl)
        ss = (np.arange(sub) - (sub - 1) / 2) / sub * dx
        SX = OX[sl, sl][..., None, None] + ss[None, None, None, :]
        SY = OY[sl, sl][..., None, None] + ss[None, None, :, None]
        self.idx_core = self._index(SX, SY)
        # path nodes: the path-length bins that carry the light inside the field of view, one random length in each
        s_mm = kern.sc * ell
        att = np.exp(-mat.mua * s_mm)
        in_fov = kern.rc <= (N * dx * 0.75) / ell
        mass = kern.mass_s[in_fov].sum(0) * att
        order = np.argsort(mass)[::-1]
        n_keep = int(np.searchsorted(np.cumsum(mass[order]), keep * mass.sum())) + 1
        self.bins = np.sort(order[:n_keep])
        u = rng.uniform(size=len(self.bins))
        self.s = np.exp(np.log(kern.se[self.bins]) * (1 - u) + np.log(kern.se[self.bins + 1]) * u) * ell
        self.att = np.exp(-mat.mua * self.s)

    def _index(self, OX, OY):
        k = self.kern
        rho = np.hypot(OX, OY) / self.mat.ell
        u = (np.log(np.maximum(rho, 1e-12)) - np.log(k.rc[1])) / k.dlr
        i0 = np.clip(np.floor(u).astype(int), 0, len(k.rc) - 3)
        fr = np.clip(u - i0, 0.0, 1.0)
        v = np.arctan2(np.abs(OY), np.abs(OX)) / k.dphi - 0.5
        j0 = np.clip(np.floor(v).astype(int), 0, k.n_phi - 2)
        fp = np.clip(v - j0, 0.0, 1.0)
        inside = rho <= k.re[-1]
        return i0, fr, j0, fp, inside, np.sign(OX * OY)

    def _interp(self, table, idx):
        i0, fr, j0, fp, inside, _ = idx
        v = ((1 - fr) * (1 - fp) * table[i0, j0] + fr * (1 - fp) * table[i0 + 1, j0]
             + (1 - fr) * fp * table[i0, j0 + 1] + fr * fp * table[i0 + 1, j0 + 1])
        return v * inside

    def node(self, f):
        """kernel maps (xx, yy, xy) of path node f, per mm^2, on the offset grid (zero offset at index P/2)"""
        k, b = self.kern, self.bins[f]
        out = np.empty((3, self.P, self.P))
        for c in range(3):
            t = k.dens[c][:, :, b]
            m = self._interp(t, self.idx)
            mc = self._interp(t, self.idx_core)
            if c == 2:
                m = m * self.idx[5]
                mc = mc * self.idx_core[5]
            m[self.core] = mc.mean((-2, -1))
            m /= self.mat.ell ** 2
            if c < 2:
                m[self.P // 2, self.P // 2] += k.delta[c, b] / self.rig.dx ** 2
            out[c] = m * self.att[f]
        return out

    def total(self):
        return sum(self.node(f) for f in range(len(self.s)))


def convolve(rig, Kmap, FB):
    """(K * B)(x) on the object grid; FB = fft2 of B zero-padded to (2N, 2N)"""
    N = rig.N
    F = sfft.fft2(sfft.ifftshift(Kmap, axes=(-2, -1)), axes=(-2, -1), workers=WORKERS)
    return sfft.ifft2(F * FB, axes=(-2, -1), workers=WORKERS)[..., :N, :N] * rig.dx ** 2


def padded_fft(rig, B):
    Bp = np.zeros((2 * rig.N, 2 * rig.N), complex)
    Bp[:rig.N, :rig.N] = B
    return sfft.fft2(Bp, workers=WORKERS)


class Scene:
    """one material under one beam position: the weight maps W_f(x) of every path node (and the surface term)"""

    def __init__(self, kern, rig, mat, rng):
        self.rig, self.mat = rig, mat
        Ib = np.abs(rig.beam()) ** 2
        if mat.ell is None:
            self.s = np.zeros(1)
            self.W = np.zeros((1, 3, rig.N, rig.N), np.float32)
            self.W[0, 0] = Ib
            return
        km = KernelMaps(kern, rig, mat, rng)
        FB = padded_fft(rig, Ib)
        F = len(km.s)
        W = np.empty((F + 1, 3, rig.N, rig.N), np.float32)
        for f in range(F):
            W[f + 1] = convolve(rig, km.node(f), FB).real
        np.maximum(W[:, :2], 0, out=W[:, :2])
        p_vol = float((W[1:, 0] + W[1:, 1]).sum()) * rig.dx ** 2
        W[0] = 0
        W[0, 0] = Ib * (mat.f_surf / (1 - mat.f_surf) * p_vol)
        self.W = W
        self.s = np.concatenate([[0.0], km.s])
        self.p_vol = p_vol

    def with_surface(self, f_surf):
        """the same volume scatterer with a share f_surf of the detected light coming from the surface"""
        import copy
        c = copy.copy(self)
        c.W = self.W.copy()
        c.W[0] = 0
        c.W[0, 0] = np.abs(self.rig.beam()) ** 2 * (f_surf / (1 - f_surf) * self.p_vol)
        if hasattr(c, "_Wb"):
            del c._Wb
        return c

    def mean_image(self):
        return self.rig.blur((self.W[:, 0] + self.W[:, 1]).sum(0)).real

    def fields(self, phases, rng):
        """object-plane fields for each row of spectral phases (rows = laser lines or exposures; columns = nodes)"""
        N = self.rig.N
        E = np.zeros((phases.shape[0], 2, N, N), np.complex64)
        ph = np.exp(1j * phases).astype(np.complex64)
        for f in range(len(self.s)):
            wxx, wyy, wxy = self.W[f]
            l11 = np.sqrt(wxx)
            g1 = cnormal((N, N), rng)
            ex = l11 * g1
            E[:, 0] += ph[:, f, None, None] * ex
            if wyy.any():
                l21 = np.divide(wxy, l11, out=np.zeros_like(wxy), where=l11 > 0)
                l22 = np.sqrt(np.maximum(wyy - l21 * l21, 0))
                ey = l21 * g1 + l22 * cnormal((N, N), rng)
                E[:, 1] += ph[:, f, None, None] * ey
        return E

    def blurred(self):
        if not hasattr(self, "_Wb"):
            self._Wb = np.stack([[self.rig.blur(self.W[f, c]).real.astype(np.float32) for c in range(3)]
                                 for f in range(len(self.s))])
        return self._Wb

    def k2_theory(self, dk, wts):
        """F1 on the pixel grid: speckle contrast squared of one exposure with laser lines (dk, wts)"""
        V = self.blurred().reshape(len(self.s), 3, -1)
        G2 = th.coherence2(dk, wts, self.rig.n * (self.s[:, None] - self.s[None, :]))
        num = sum(wc * np.einsum("fp,fp->p", V[:, c], G2 @ V[:, c]) for c, wc in ((0, 1), (1, 1), (2, 2)))
        den = (V[:, 0] + V[:, 1]).sum(0) ** 2
        return (num / den).reshape(self.rig.N, self.rig.N)

    def step_theory(self, dk_a, dk_b):
        """F3 on the pixel grid: numerator |Gamma_ab|^2 and the two self terms, for single-line exposures"""
        V = self.blurred().reshape(len(self.s), 3, -1)

        def g2(d):
            ph = np.exp(1j * d * self.rig.n * self.s)
            return sum(wc * np.abs(ph @ V[:, c]) ** 2 for c, wc in ((0, 1), (1, 1), (2, 2)))
        N = self.rig.N
        return g2(dk_a - dk_b).reshape(N, N), g2(0.0).reshape(N, N)


def camera(I, rig, peak, rng, exposure=0.6, full_well=10000.0, read=3.0, bits=12, gains=(8.0, 1.0, 0.125), pedestal=64):
    """two frames (independent noise, same speckle), each merged from three bracketed exposures: every pixel takes the
    longest exposure that stays below 90% of full well. `exposure` is the mean signal at the spot centre, as a share
    of full well, in the middle exposure. The converter has a black-level pedestal, subtracted afterwards (a dark frame),
    so read noise is not clipped at zero. Returned in units of the mean signal at the spot centre."""
    Ip = np.clip(rig.to_pixels(I) / peak, 0, None)
    top = 2 ** bits - 1
    frames = []
    for _ in range(2):
        best = np.zeros_like(Ip)
        done = np.zeros(Ip.shape, bool)
        for g in sorted(gains, reverse=True):
            e = rng.poisson(Ip * (exposure * g * full_well)) + rng.normal(0, read, Ip.shape)
            dn = np.clip(np.round(e / full_well * top) + pedestal, 0, top)
            ok = (dn < 0.9 * top) & ~done
            best[ok] = (dn[ok] - pedestal) / top / (exposure * g)
            done |= ok
        best[~done] = 0.9 / (exposure * min(gains))          # clipped even in the shortest exposure
        frames.append(best)
    return frames


def _local_fit(y, w, sig, order):
    """value at each bin of a weighted local polynomial fit (Gaussian window of width sig bins)"""
    g = np.arange(-15, 16)
    ker = np.exp(-0.5 * (g / sig) ** 2)

    def mom(v, k):
        return np.convolve(np.pad(v, 15, mode="edge"), (ker * g ** k)[::-1], mode="valid")
    n = order + 1
    S = [mom(w, k) for k in range(2 * order + 1)]
    T = [mom(w * y, k) for k in range(n)]
    M = np.stack([np.stack([S[i + j] for j in range(n)], -1) for i in range(n)], -2)
    return np.linalg.solve(M, np.stack(T, -1)[..., None])[..., 0, 0]


def spot_trend(img, cx, cy, sig=3.0, sig_h=5.0):
    """smooth model of the spot's mean brightness about (cx, cy) in pixels:
    a0(r) [1 + c2(r) cos 2phi + s2(r) sin 2phi].
    log a0 is a local parabola fit to the log of the ring means (the log of a Gaussian edge is a parabola, so the fit
    does not bend it); the 2-phi terms follow the oval glow that polarized light makes in a scattering material (no
    light is scattered along the direction of the polarization)."""
    yy, xx = np.indices(img.shape)
    ddx, ddy = xx - cx, yy - cy
    r = np.hypot(ddx, ddy)
    ph = np.arctan2(ddy, ddx)
    c2, s2 = np.cos(2 * ph), np.sin(2 * ph)
    rb = np.floor(r).astype(int).ravel()
    cnt = np.bincount(rb).astype(float)
    v = img.ravel()
    a0 = np.bincount(rb, v) / np.maximum(cnt, 1)
    a2 = 2 * np.bincount(rb, v * c2.ravel()) / np.maximum(cnt, 1)
    b2 = 2 * np.bincount(rb, v * s2.ravel()) / np.maximum(cnt, 1)
    pos = np.maximum(a0, 1e-7 * a0.max())
    a0s = np.exp(_local_fit(np.log(pos), cnt, sig, 2))
    taper = np.clip((np.arange(len(a0)) - 3) / 4.0, 0, 1)          # too few pixels near the centre for the harmonics
    q2, p2 = _local_fit(a2 / pos, cnt, sig_h, 0) * taper, _local_fit(b2 / pos, cnt, sig_h, 0) * taper
    rc = np.arange(len(a0)) + 0.5
    return np.interp(r, rc, a0s) * (1 + np.interp(r, rc, q2) * c2 + np.interp(r, rc, p2) * s2), r


def ring_products(fa, fb, rig, centre_a, centre_b, edges_mm):
    """ring means of the cross-frame products of normalized residuals: frame 1 of exposure a with frame 2 of b"""
    n = rig.N // rig.pix
    px = rig.dx * rig.pix
    to_px = lambda c: (c[0] / px + n / 2 - 0.5, c[1] / px + n / 2 - 0.5)
    ta, _ = spot_trend(0.5 * (fa[0] + fa[1]), *to_px(centre_a))
    tb, _ = spot_trend(0.5 * (fb[0] + fb[1]), *to_px(centre_b))
    yy, xx = np.indices((n, n))
    c0 = to_px((0.0, 0.0))
    r = np.hypot(xx - c0[0], yy - c0[1]) * px
    da = fa[0] / np.maximum(ta, 1e-9) - 1.0
    db = fb[1] / np.maximum(tb, 1e-9) - 1.0
    ib = np.searchsorted(edges_mm, r.ravel()) - 1
    ok = (ib >= 0) & (ib < len(edges_mm) - 1)
    cnt = np.bincount(ib[ok], minlength=len(edges_mm) - 1)
    return np.bincount(ib[ok], (da * db).ravel()[ok], minlength=len(edges_mm) - 1) / np.maximum(cnt, 1), cnt


def ring_mean(img_grid, rig, edges_mm, weight=None):
    r = np.hypot(rig.X, rig.Y).ravel()
    ib = np.searchsorted(edges_mm, r) - 1
    ok = (ib >= 0) & (ib < len(edges_mm) - 1)
    w = np.ones(r.size) if weight is None else weight.ravel()
    return (np.bincount(ib[ok], (img_grid.ravel() * w)[ok], minlength=len(edges_mm) - 1)
            / np.maximum(np.bincount(ib[ok], w[ok], minlength=len(edges_mm) - 1), 1e-30))
