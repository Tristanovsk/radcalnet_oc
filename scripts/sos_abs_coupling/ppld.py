"""
Photon path-length (equivalence-theorem) model of the gaseous transmittance of each term of the TOA signal.

By the equivalence theorem, for an absorber of optical thickness tau (vertical, per unit air mass) the
transmittance of a term X (path reflectance R0, water-leaving term A, irradiance Ed) is the Laplace transform of
the distribution of the absorber paths. With a Gamma distribution of mean eta*m and relative variance s2:

    T_X(tau) = (1 + s2 * eta * m * tau) ** (-1 / s2)          (-> exp(-eta m tau) when s2 -> 0)

This is a monochromatic model. A CKD interval of the SOS-ABS simulations is a mixture of such absorbers,
Tg(m) = sum_i a_i exp(-tau_i m): the weights a_i on a fixed grid of tau are recovered from the direct-beam table
Tg(m), m = 0..10 (non-negative least squares), and

    T_X = sum_i a_i (1 + s2 eta m tau_i) ** (-1 / s2).

eta and s2 are linear in (m - 2, aot - 0.1, ln(U/2)) per window and term, fitted on the absolute transmittance
error. In radcalnet_oc the model is applied channel by channel with tau m = -ln Tg(m) of the channel (no
smoothing needed: a REPTRAN channel is a single exponential).
"""
import numpy as np
import xarray as xr
from scipy.optimize import least_squares, nnls

import coupling as cp

TAU_GRID = np.r_[0., np.logspace(-3, np.log10(60.), 48)]   # optical thickness per unit air mass
S2_MIN = 1e-4
PARAMS = ("1", "m-2", "aot-0.1", "ln(U/2)")


def gamma_t(x, s2):
    """Gamma-path transmittance for a mean absorber optical path x: (1 + s2 x)^(-1/s2)."""
    s2 = np.maximum(s2, S2_MIN)
    return np.exp(-np.log1p(s2 * x) / s2)


def exponential_sum(tg):
    """Weights a(wl, h2o, tau) of Tg(m) = sum_j a_j exp(-tau_j m) on TAU_GRID (NNLS, sum a = 1), and the rms
    of the fit over m."""
    tg = tg.transpose("wl", "h2o", "m")
    m = tg.m.values
    A = np.exp(-np.outer(m, TAU_GRID))
    w = 1e3
    Aw = np.vstack([A, w * np.ones(len(TAU_GRID))])
    out = np.zeros(tg.shape[:2] + (len(TAU_GRID),))
    rms = np.zeros(tg.shape[:2])
    for i in range(tg.sizes["wl"]):
        for j in range(tg.sizes["h2o"]):
            t = tg.values[i, j]
            a, _ = nnls(Aw, np.r_[t, w])
            out[i, j] = a
            rms[i, j] = np.sqrt(np.mean((A @ a - t) ** 2))
    coords = {"wl": tg.wl, "h2o": tg.h2o, "tau": TAU_GRID}
    return (xr.DataArray(out, coords=coords, dims=("wl", "h2o", "tau")),
            xr.DataArray(rms, coords={"wl": tg.wl, "h2o": tg.h2o}, dims=("wl", "h2o")))


def linear(c, m, aot, h2o, w):
    out = c[0] + c[1] * (m - 2) + c[2] * (aot - 0.1)
    if w in cp.H2O_WINDOWS:
        out = out + c[3] * np.log(h2o / cp.H2O_REF)
    return out


def _clip(r, m, aot, h2o):
    return (np.clip(m, r["m_min"], r["m_max"]), np.clip(aot, r["aot_min"], r["aot_max"]),
            np.clip(h2o, r["h2o_min"], r["h2o_max"]))


def transmittance(a, m, eta, s2):
    """T = sum_j a_j gamma_t(eta m tau_j, s2); a (..., tau), eta and s2 broadcast with m."""
    out = 0.
    for j, tau in enumerate(a.tau.values):
        aj = a.isel(tau=j, drop=True)
        if float(aj.max()) == 0:
            continue
        out = out + aj * gamma_t(eta * m * tau, s2)
    return out


def _samples(eff, a, w, term, n, rng, exclude=None):
    """Random sample of (T observed, a_j, m, aot, U, weight) for the fit."""
    T, geo = eff["T"], eff["geo"]
    obs = {"atm": T.Tatm, "w": T.Tw, "Ed": T.TEd}[term]
    m = geo.m0 if term == "Ed" else geo.m
    t0 = T.Tg_m0 if term == "Ed" else T.Tg_m
    obs, t0 = cp.window(obs, w), cp.window(t0, w)
    obs = obs.where(t0 < cp.STRONG)
    obs = cp.subset(obs, exclude)
    st = obs.stack(pt=obs.dims).dropna("pt")
    idx = rng.choice(st.sizes["pt"], min(n, st.sizes["pt"]), replace=False)
    st = st.isel(pt=idx)
    sel = {d: xr.DataArray(st[d].values, dims="pt") for d in obs.dims}
    av = cp.window(a, w).sel(wl=sel["wl"], h2o=sel["h2o"]).transpose("pt", "tau").values
    mv = m.sel({d: sel[d] for d in m.dims}).values
    return dict(T=st.values, a=av, m=mv, aot=sel["aot_ref"].values, h2o=sel["h2o"].values)


def fit_term(eff, a, w, term, n=60000, seed=0, exclude=None):
    """Fit eta and s2 (linear in m, aot, ln U) by least squares on the absolute transmittance."""
    rng = np.random.default_rng(seed)
    d = _samples(eff, a, w, term, n, rng, exclude)
    h2o_term = w in cp.H2O_WINDOWS

    def unpack(p):
        ce = np.r_[p[0:3], p[3] if h2o_term else 0.]
        cs = np.r_[p[4:7], p[7] if h2o_term else 0.]
        return ce, cs

    def model(p):
        ce, cs = unpack(p)
        eta = linear(ce, d["m"], d["aot"], d["h2o"], w)
        s2 = linear(cs, d["m"], d["aot"], d["h2o"], w)
        g = gamma_t(eta[:, None] * d["m"][:, None] * TAU_GRID[None, :], s2[:, None])
        return (d["a"] * g).sum(1)

    p0 = np.array([0.5 if term == "atm" else 1.03, 0, 0, 0, 0.3 if term == "atm" else 0.02, 0, 0, 0])
    res = least_squares(lambda p: model(p) - d["T"], p0, method="trf", x_scale="jac")
    ce, cs = unpack(res.x)
    rms = float(np.sqrt(np.mean(res.fun ** 2)))
    return ce, cs, rms


def fit_all(eff, a, exclude=None, n=60000):
    """PPLD model of all the windows and terms: coefs[(w, term)] = (c_eta, c_s2), ranges, rms."""
    geo, eta = eff["geo"], eff["eta"]
    model = dict(coefs={}, rms={}, ranges={}, windows=dict(cp.WINDOWS))
    sel = lambda x: cp.subset(x, exclude)
    for w in cp.WINDOWS:
        for k in cp.TERMS:
            ce, cs, rms = fit_term(eff, a, w, k, n=n, exclude=exclude)
            model["coefs"][(w, k)] = (ce, cs)
            model["rms"][(w, k)] = rms
            m = sel(geo.m0 if k == "Ed" else geo.m.where(geo.valid))
            e = sel(eta)
            model["ranges"][(w, k)] = dict(m_min=float(m.min()), m_max=float(m.max()),
                                           aot_min=float(e.aot_ref.min()), aot_max=float(e.aot_ref.max()),
                                           h2o_min=float(e.h2o.min()), h2o_max=float(e.h2o.max()))
    return model


def parameters(model, w, term, m, aot, h2o):
    ce, cs = model["coefs"][(w, term)]
    mm, aa, uu = _clip(model["ranges"][(w, term)], m, aot, h2o)
    return linear(ce, mm, aa, uu, w), np.maximum(linear(cs, mm, aa, uu, w), S2_MIN)


def apply(eff, model, a, term):
    """Transmittance of a term with the PPLD model on the CKD intervals of the simulations."""
    T, geo = eff["T"], eff["geo"]
    m = geo.m0 if term == "Ed" else geo.m
    parts = []
    for w, (lo, hi) in model["windows"].items():
        eta, s2 = parameters(model, w, term, m, T.aot_ref, T.h2o)
        parts.append(transmittance(a.sel(wl=slice(lo, hi)), m, eta, s2))
    out = xr.concat(parts, "wl")
    t0 = T.Tg_m0 if term == "Ed" else T.Tg_m
    return out.broadcast_like(t0).where(np.isfinite(t0)) if term != "Ed" else out


TABLE = cp.os.path.join(cp.HERE, "..", "..", "radcalnet_oc", "data", "auxdata", "gas_coupling_ppld.csv")


def table_frame(model):
    """Table of radcalnet_oc (kernel.GasCoupling, form 'ppld'), pandas.DataFrame."""
    import pandas as pd
    rows = []
    for (w, k), (ce, cs) in model["coefs"].items():
        a, b = model["windows"][w]
        rows.append({"window": w, "wl_min": a, "wl_max": b, "term": k, "form": "ppld",
                     **{f"c{i}": v for i, v in enumerate(ce)}, **{f"s{i}": v for i, v in enumerate(cs)},
                     "rms": model["rms"][(w, k)], **model["ranges"][(w, k)]})
    return pd.DataFrame(rows)


def write_table(model, path=TABLE, source="v2"):
    table = table_frame(model)
    with open(path, "w") as f:
        f.write("# Photon path-length model of the gaseous transmittance of the scattered light\n"
                "# (radcalnet_oc.kernel.GasCoupling, form ppld), per absorption window and term (atm: path\n"
                "# reflectance, w: water-leaving term, m = 1/mu0 + 1/muv; Ed: downwelling irradiance, m = 1/mu0):\n"
                "#   T = (1 + s2 eta tau)^(-1/s2), tau = -ln Tg(m) of the channel (Gamma distribution of the paths)\n"
                "#   eta = c0 + c1 (m - 2) + c2 (aot550 - 0.1) + c3 ln(U / 2 g cm-2)\n"
                "#   s2  = s0 + s1 (m - 2) + s2 (aot550 - 0.1) + s3 ln(U / 2 g cm-2), s2 >= 1e-4\n"
                "# U water vapour column (g cm-2); m, aot550 and U clipped to the ranges; rms: absolute transmittance.\n"
                f"# Fitted on the SOS-ABS coupled simulations, set {source} of scripts/sos_abs_coupling/make_luts.py\n"
                "# (CKD 5 cm-1 decomposed into exponential sums), by scripts/sos_abs_coupling/ppld.py\n")
        table.to_csv(f, index=False, float_format="%.6g")
    return table


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "fit":
        sys.exit("usage: python ppld.py fit [LUT_DIR] [OUTPUT_CSV]")
    d = sys.argv[2] if len(sys.argv) > 2 else cp.DATA_V2
    out = sys.argv[3] if len(sys.argv) > 3 else TABLE
    eff = cp.effective(cp.load(d))
    a, rms = exponential_sum(eff["tg"])
    print(f"exponential sums of Tg(m): max rms {float(rms.max()):.1e}")
    table = write_table(fit_all(eff, a), path=out, source=cp.os.path.basename(cp.os.path.normpath(d)))
    print(table.round(4).to_string())
    print("written:", cp.os.path.abspath(out))
