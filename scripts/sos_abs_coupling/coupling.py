"""
Analysis of the SOS-ABS coupled absorption-scattering simulations (make_luts.py) and fit of the effective
air masses used by radcalnet_oc (kernel.GasCoupling).

The TOA reflectance R = I_up / mu0 is split, for each geometry, into a path term and a surface term,

    R(rho) = R0 + rho A / (1 - S rho),

for the simulations with (abs) and without (noabs) gaseous absorption. The direct sunglint (Cox-Munk, direct
transmittance of the path) is removed from R0, which is then the diffuse path reflectance of the LUTs of
radcalnet_oc. The effective transmittances
Tatm = R0_abs / R0_noabs, Tw = A_abs / A_noabs and TEd = Ed_abs / Ed_noabs are converted into effective air
masses m* with the direct transmittance table Tg_dir(wl, m) of the same CKD absorption, and eta = m* / m.

Command line: python coupling.py fit [LUT_DIR] [OUTPUT_CSV]
    fit the adopted model (FORMS_ADOPTED: effective air mass, single-scattering blend for the path
    reflectance in the O2-A head and the 940 nm core) on the LUTs of LUT_DIR (default data/v2) and write the
    table of radcalnet_oc
    (default radcalnet_oc/data/auxdata/gas_coupling_eta.csv).
"""
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
# simulations of SOS-ABS: $SOS_ABS_DATA, else the folder data/ next to the scripts if present, else
# /data/sos_abs_coupling (the development machine; no symbolic link in the repository: setuptools would follow it)
DATA = os.environ.get("SOS_ABS_DATA") or next(
    (d for d in (os.path.join(HERE, "data"), "/data/sos_abs_coupling") if os.path.isdir(d)), os.path.join(HERE, "data"))
DATA_V1 = DATA
DATA_V2 = os.path.join(DATA, "v2")
TABLE = os.path.join(HERE, "..", "..", "radcalnet_oc", "data", "auxdata", "gas_coupling_eta.csv")

# absorption windows of the parameterization; the O2-A band (head and R branch / P branch, either side of
# the band origin at 762 nm) and the 940 nm band (shoulder / core) are split, their parts saturating
# differently
WINDOWS = {"O2-B": (684, 698), "H2O-720": (712, 736), "O2-A-R": (757, 762.25), "O2-A-P": (762.25, 772),
           "H2O-820": (808, 836), "H2O-900": (890, 919.5), "H2O-940": (919.5, 990)}
# spectral ranges of the simulations
RANGES = [(684, 698), (712, 736), (757, 772), (808, 836), (890, 990)]
H2O_WINDOWS = ("H2O-720", "H2O-820", "H2O-900", "H2O-940")
TERMS = ("atm", "w", "Ed")
H2O_REF = 2.0       # g cm-2, H2O column of the v1 simulations (reference of the H2O term)
STRONG = 0.95       # channels kept for eta: Tg_dir(m) < STRONG
GLINT_MIN = 30.     # deg, minimum glint angle of the geometries kept
VZA_MAX = 60.       # deg


# ---------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------
def open_lut(path):
    """SOS-ABS LUT with the wavelengths in nm; an H2O dimension of length 1 is added if missing."""
    ds = xr.open_dataset(path)
    ds = ds.assign_coords(wl=np.round(ds.wl.values * 1e3, 2))
    return ds


def load(d=DATA_V2):
    """LUTs of a simulation set: dict abs, noabs, tg (Tg_dir(wl, [h2o], m)); viewing angles restricted to
    the Gauss angles <= VZA_MAX."""
    lut = {}
    for k in ("abs", "noabs"):
        ds = open_lut(os.path.join(d, f"{k}.nc"))
        keep = (ds.gauss_weight > 0) & (ds.vza <= VZA_MAX)
        lut[k] = ds.isel(vza=np.flatnonzero(keep.values))
    if "h2o" not in lut["abs"].dims:
        lut["abs"] = lut["abs"].expand_dims(h2o=[H2O_REF])
    return dict(lut, tg=tg_table(d))


def tg_table(d):
    """CKD direct transmittance Tg(wl, h2o, m) from the ratio of the direct fluxes (m = 1/mu0)."""
    a = open_lut(os.path.join(d, "tgdir_abs.nc")).flux_dir_down
    n = open_lut(os.path.join(d, "tgdir_noabs.nc")).flux_dir_down
    tg = (a / n).squeeze(drop=True)
    if "h2o" not in tg.dims:
        tg = tg.expand_dims(h2o=[H2O_REF])
    tg = tg.assign_coords(m=("sza", 1 / np.cos(np.radians(tg.sza.values)))).swap_dims(sza="m").drop_vars("sza")
    tg = tg.sortby("m").transpose("wl", "h2o", "m")
    # Tg(m = 0) = 1: effective air masses below 1 (path reflectance)
    one = xr.ones_like(tg.isel(m=0)).expand_dims(m=[0.])
    return xr.concat([one, tg], "m").transpose("wl", "h2o", "m")


def tg_of_m(tg, m):
    """Tg_dir at the air mass m (log-interpolation in m)."""
    return np.exp(np.log(tg).interp(m=m)).drop_vars("m", errors="ignore")


def m_of_tg(tg, t):
    """Effective air mass m* such that Tg_dir(m*) = t, per (wl, h2o); nan outside the table (m from 0 to 10)
    or where Tg depends weakly on m (Tg(m=1) > 0.99)."""
    lt = np.log(tg.transpose("wl", "h2o", "m"))
    t = t.transpose("wl", "h2o", ...)
    out = np.full(t.shape, np.nan)
    mm = tg.m.values[::-1]
    for i in range(lt.sizes["wl"]):
        for j in range(lt.sizes["h2o"]):
            x = lt.values[i, j, ::-1]                    # increasing when m decreases (last: m = 0)
            if np.all(np.diff(x) > 0) and x[-2] < np.log(0.99):   # x[-2]: m = 1
                out[i, j] = np.interp(np.log(t.values[i, j]), x, mm, left=np.nan, right=np.nan)
    return t.copy(data=out)


# ---------------------------------------------------------------------------
# decomposition of the TOA signal and effective transmittances
# ---------------------------------------------------------------------------
def decompose(ds):
    """R(rho) = R0 + rho A / (1 - S rho), A and S from the albedos rho > 0 (exact for two of them)."""
    R = ds.I_up / np.cos(np.radians(ds.sza))
    R0 = R.sel(albedo=0)
    dR = (R - R0).sel(albedo=ds.albedo[ds.albedo > 0])
    y, x = dR.albedo / dR, dR.albedo                     # y = 1/A - (S/A) rho
    xm, ym = x.mean(), y.mean("albedo")
    b = ((x - xm) * (y - ym)).sum("albedo") / ((x - xm) ** 2).sum()
    a = ym - b * xm
    A = 1 / a
    S = -b * A
    return xr.Dataset({"R": R, "R0": R0, "A": A, "S": S})


def glint_reflectance(ds, wind=5.0, n=1.33):
    """Cox-Munk (isotropic) reflectance of the direct sunlight, without atmosphere (radcalnet_oc.coxmunk),
    on the geometry of a LUT (SOS azimuth convention: 0 deg = specular side)."""
    from radcalnet_oc.coxmunk import Sunglint
    out = np.zeros((ds.sizes["sza"], ds.sizes["vza"], ds.sizes["azi"]))
    for i, s in enumerate(ds.sza.values):
        for j, v in enumerate(ds.vza.values):
            for k, a in enumerate(ds.azi.values):
                g = Sunglint(np.array([s]), np.array([v]), np.array([180. - a]), m=n)
                out[i, j, k] = g.sunglint(wind, stats="cm_iso", shadow=False)[0][0]
    return xr.DataArray(out, coords={"sza": ds.sza, "vza": ds.vza, "azi": ds.azi}, dims=("sza", "vza", "azi"))


def direct_glint(ds):
    """Sunglint of the direct sunlight at TOA without gaseous absorption, G = exp(-tau m) rho_glint, with the
    optical thickness tau of the direct flux of the LUT (normalised to E0 mu0: exp(-tau / mu0))."""
    mu0 = np.cos(np.radians(ds.sza))
    tau = -mu0 * np.log(ds.flux_dir_down.sel(albedo=0))
    m = 1 / mu0 + 1 / np.cos(np.radians(ds.vza))
    return np.exp(-tau * m) * glint_reflectance(ds)


def geometry(ds):
    """mu0, muv, m0 = 1/mu0, m = 1/mu0 + 1/muv and the mask of the geometries kept (Gauss angles,
    vza <= VZA_MAX, glint angle > GLINT_MIN)."""
    mu0, muv = np.cos(np.radians(ds.sza)), np.cos(np.radians(ds.vza))
    cos_glint = mu0 * muv + np.sqrt(1 - mu0 ** 2) * np.sqrt(1 - muv ** 2) * np.cos(np.radians(ds.azi))
    valid = ((ds.gauss_weight > 0) & (ds.vza <= VZA_MAX) & (np.degrees(np.arccos(cos_glint)) > GLINT_MIN)
             & (ds.azi < 360))
    return xr.Dataset({"mu0": mu0, "muv": muv, "m0": 1 / mu0, "m": 1 / mu0 + 1 / muv, "valid": valid})


def effective(lut):
    """Effective transmittances, CKD direct transmittances and eta of the three terms.

    :return: dict with dec (decompositions abs / noabs), geo (geometry), T (Dataset Tatm, Tw, TEd, Tg_m,
        Tg_m0) and eta (Dataset atm, w, Ed; nan where Tg < STRONG is not met or out of the table)
    """
    dec = {k: decompose(lut[k]) for k in ("abs", "noabs")}
    geo = geometry(lut["noabs"])
    tg = lut["tg"]
    # diffuse path reflectance (as the LUT of radcalnet_oc): direct sunglint removed, attenuated in the
    # absorbing case by the direct transmittance of the Sun-surface-sensor path
    G = direct_glint(lut["noabs"])
    dec["noabs"]["R0_glint"] = G
    dec["abs"]["R0_glint"] = G * tg_of_m(tg, geo.m)
    for k in ("abs", "noabs"):
        dec[k]["R0_total"] = dec[k]["R0"]
        dec[k]["R0"] = dec[k]["R0_total"] - dec[k]["R0_glint"]
    fl = {k: lut[k].flux_tot_down.sel(albedo=0) for k in ("abs", "noabs")}
    T = xr.Dataset({
        "Tatm": (dec["abs"].R0 / dec["noabs"].R0).where(geo.valid),
        "Tw": (dec["abs"].A / dec["noabs"].A).where(geo.valid),
        "TEd": fl["abs"] / fl["noabs"],
        "Tg_m": tg_of_m(tg, geo.m).where(geo.valid),
        "Tg_m0": tg_of_m(tg, geo.m0),
        "Ed_abs": fl["abs"],
        "Ed_noabs": fl["noabs"],
    })
    eta = xr.Dataset({
        "atm": (m_of_tg(tg, T.Tatm) / geo.m).where(T.Tg_m < STRONG),
        "w": (m_of_tg(tg, T.Tw) / geo.m).where(T.Tg_m < STRONG),
        "Ed": (m_of_tg(tg, T.TEd) / geo.m0).where(T.Tg_m0 < STRONG),
    })
    return dict(dec=dec, geo=geo, T=T, eta=eta, tg=tg)


def window(x, w):
    a, b = WINDOWS[w]
    return x.sel(wl=slice(a, b))


# ---------------------------------------------------------------------------
# parameterization (same forms as radcalnet_oc.kernel.GasCoupling)
# ---------------------------------------------------------------------------
# coefficient c = c0 + c1 (m - 2) + c2 (aot - 0.1) + c3 ln(U / H2O_REF) + c4 tau
#   (c3 = 0 in the O2 windows; tau = -ln Tg(m), absorption strength, blend form only)
#   form 'eta'  : T = Tg(eta m), eta = c
#   form 'blend': T = Tg(m)^(1-k) Tss^k, k = c, Tss = (1 - Tg(m)) / (-ln Tg(m))
FEATURES = ("1", "m-2", "aot-0.1", "ln(U/2)", "tau")
NO_TAU = ("1", "m-2", "aot-0.1", "ln(U/2)")
LINEAR = ("1", "m-2", "aot-0.1")            # model of the v1 table (no H2O term)
FORMS_ETA = {"atm": "eta", "w": "eta", "Ed": "eta"}
# weights of the fits: None for the absorption 1 - Tg (first version), otherwise squared absolute error of
# the transmittance (what matters for band averages), times WEIGHT(T)
WEIGHT = lambda t: 1.
FORMS_BLEND = {"atm": "blend", "w": "eta", "Ed": "eta"}
# forms adopted for the table of radcalnet_oc (choice per window on band-level errors, cross-validation and
# transfer to the gaseous table of the package: notebook/research/radcalnet_oc_gaseous_coupling_analysis),
# fitted with the features NO_TAU
FORMS_ADOPTED = {**FORMS_ETA, ("O2-A-R", "atm"): "blend", ("H2O-940", "atm"): "blend"}


def single_scattering(t):
    """Tss = (1 - T) / (-ln T)."""
    tau = -np.log(t)
    return xr.where(tau < 1e-12, 1., -np.expm1(-tau) / xr.where(tau < 1e-12, 1., tau))


def _like(x, ref):
    """x restricted to the coordinates of ref on their common dimensions."""
    return x.sel({d: ref[d].values for d in x.dims if d in ref.dims})


def _features(e, m, w, features, tau=None):
    """Columns of the design matrix (DataArrays broadcast with e); tau (absorption strength) for the
    blend form only (zeros otherwise)."""
    aot, h2o = e.aot_ref.broadcast_like(e), e.h2o.broadcast_like(e)
    m = _like(m, e).broadcast_like(e)
    cols = {"1": xr.ones_like(e), "m-2": m - 2, "aot-0.1": aot - 0.1,
            "ln(U/2)": np.log(h2o / H2O_REF) if w in H2O_WINDOWS else xr.zeros_like(e),
            "tau": _like(tau, e).broadcast_like(e) if tau is not None else xr.zeros_like(e)}
    return [cols[f] for f in FEATURES if f in features]


def _lstsq(y, X, weight):
    """Weighted least squares y = X c; columns null over the data get c = 0."""
    ok = np.isfinite(y) & np.isfinite(weight) & np.all(np.isfinite(X), axis=-1)
    y, X, sw = y[ok], X[ok], np.sqrt(weight[ok])
    used = np.ptp(X, axis=0) > 0
    used[0] = True
    c = np.zeros(X.shape[-1])
    c[used], *_ = np.linalg.lstsq(X[:, used] * sw[:, None], y * sw, rcond=None)
    rms = np.sqrt(np.average((y - X @ c) ** 2, weights=sw ** 2))
    return c, rms


def subset(x, exclude=None, only=None):
    """Drop (exclude) or keep (only) coordinate values, e.g. exclude={'sza': 40.} (cross-validation)."""
    for k, v in (exclude or {}).items():
        x = x.drop_sel({k: v}) if k in x.dims else x
    for k, v in (only or {}).items():
        x = x.sel({k: np.atleast_1d(v)}) if k in x.dims else x
    return x


def fit_term(eff, w, term, form, features=FEATURES, exclude=None, only=None):
    """Fit the coefficient of a term in a window; returns c (4 values, FEATURES order) and the rms of the
    weighted residual (on eta for 'eta', on ln T for 'blend')."""
    T, geo, eta = eff["T"], eff["geo"], eff["eta"]
    m = geo.m0 if term == "Ed" else geo.m
    t0 = window(T.Tg_m0 if term == "Ed" else T.Tg_m, w)
    if form == "eta":
        y = subset(window(eta[term], w), exclude, only)
        cols = _features(y, m, w, features)
        t = _like(t0, y).broadcast_like(y)
        # absolute error of the transmittance: dT = T tau deta
        weight = WEIGHT(t) * (t * -np.log(t)) ** 2 if WEIGHT is not None else (1 - t)
        mult = 1.
    else:
        ta = window({"atm": T.Tatm, "w": T.Tw, "Ed": T.TEd}[term], w)
        ta, t0 = xr.broadcast(subset(ta, exclude, only), subset(t0, exclude, only))
        y = np.log(ta / t0).where(t0 < STRONG)
        mult = np.log(single_scattering(t0) / t0)   # ln(Tatm/T) = k ln(Tss/T)
        cols = _features(y, m, w, features, tau=-np.log(t0))
        # absolute error of the transmittance: dT = T d(ln T)
        weight = WEIGHT(ta) * ta ** 2 if WEIGHT is not None else 1 - t0
    X = np.stack([(c * mult).values.ravel() for c in cols], axis=-1)
    c, rms = _lstsq(y.values.ravel(), X, weight.values.ravel())
    full = np.zeros(len(FEATURES))
    full[[FEATURES.index(f) for f in FEATURES if f in features]] = c
    return full, rms


def fit_all(eff, forms=FORMS_BLEND, features=FEATURES, exclude=None, only=None):
    """Model fitted on eff: dict with coefs[(w, term)], rms, ranges, forms."""
    geo, eta = eff["geo"], eff["eta"]
    coefs, rms, ranges = {}, {}, {}
    sel = lambda x: subset(x, exclude, only)
    forms = {(w, k): forms.get((w, k), forms.get(k)) for w in WINDOWS for k in TERMS}
    for w in WINDOWS:
        for k in TERMS:
            coefs[(w, k)], rms[(w, k)] = fit_term(eff, w, k, forms[(w, k)], features, exclude, only)
            m = sel(geo.m0 if k == "Ed" else geo.m.where(geo.valid))
            e = sel(eta)
            t0 = sel(window(eff["T"].Tg_m0 if k == "Ed" else eff["T"].Tg_m, w))
            ranges[(w, k)] = dict(m_min=float(m.min()), m_max=float(m.max()),
                                  aot_min=float(e.aot_ref.min()), aot_max=float(e.aot_ref.max()),
                                  h2o_min=float(e.h2o.min()), h2o_max=float(e.h2o.max()),
                                  tau_max=float((-np.log(t0)).max()))
    return dict(coefs=coefs, rms=rms, ranges=ranges, forms=forms, windows=dict(WINDOWS))


def from_table(path=TABLE):
    """Model read from a table of radcalnet_oc (e.g. the v1 table: no H2O term, form 'eta')."""
    import pandas as pd
    t = pd.read_csv(path, comment="#")
    model = dict(coefs={}, rms={}, ranges={}, forms={}, windows={})
    for r in t.itertuples():
        model["windows"][r.window] = (r.wl_min, r.wl_max)
        g = lambda k, d: getattr(r, k, d)
        model["coefs"][(r.window, r.term)] = np.array([r.c0, r.c1, r.c2, g("c3", 0.), g("c4", 0.)])
        model["ranges"][(r.window, r.term)] = dict(
            m_min=r.m_min, m_max=r.m_max, aot_min=r.aot_min, aot_max=r.aot_max,
            h2o_min=g("h2o_min", H2O_REF), h2o_max=g("h2o_max", H2O_REF), tau_max=g("tau_max", np.inf))
        model["forms"][(r.window, r.term)] = g("form", "eta")
    return model


def coefficient(model, w, term, m, aot, h2o, tau=None):
    c = model["coefs"][(w, term)]
    r = model["ranges"][(w, term)]
    m = np.clip(m, r["m_min"], r["m_max"])
    aot = np.clip(aot, r["aot_min"], r["aot_max"])
    out = c[0] + c[1] * (m - 2) + c[2] * (aot - 0.1)
    if c[3] != 0:
        out = out + c[3] * np.log(np.clip(h2o, r["h2o_min"], r["h2o_max"]) / H2O_REF)
    if len(c) > 4 and c[4] != 0 and tau is not None:
        out = out + c[4] * np.clip(tau, 0, r["tau_max"])
    return out


def apply(eff, model, term):
    """Transmittance of a term given by a model (None: decoupled), on the wavelengths of the LUTs."""
    T, geo, tg = eff["T"], eff["geo"], eff["tg"]
    m = geo.m0 if term == "Ed" else geo.m
    t0 = T.Tg_m0 if term == "Ed" else T.Tg_m
    if model is None:
        return t0
    aot, h2o = T.aot_ref, T.h2o
    parts = []
    for w, (a, b) in model.get("windows", WINDOWS).items():
        if model["forms"][(w, term)] == "eta":
            c = coefficient(model, w, term, m, aot, h2o)
            t = tg_of_m(tg.sel(wl=slice(a, b)), c * m)
        else:
            t = t0.sel(wl=slice(a, b))
            c = coefficient(model, w, term, m, aot, h2o, tau=-np.log(t))
            t = t ** (1 - c) * single_scattering(t) ** c
        parts.append(t)
    # same dimensions for all the windows (the eta form does not depend on the azimuth)
    ref = xr.broadcast(*[p.isel(wl=0, drop=True) for p in parts])[0]
    out = xr.concat([p.broadcast_like(ref) for p in parts], "wl")
    return out.where(np.isfinite(t0)) if term != "Ed" else out


def toa(eff, Tatm, Tw, rho):
    """Model R = Tatm R0_noabs + Tw rho A / (1 - S rho) and coupled simulation R_abs(rho)."""
    N, A = eff["dec"]["noabs"], eff["dec"]["abs"]
    model = Tatm * N.R0 + Tw * rho * N.A / (1 - N.S * rho)
    truth = A.R0 + rho * A.A / (1 - A.S * rho)
    return model, truth


def table_frame(model):
    """Table of radcalnet_oc (kernel.GasCoupling) of a model, pandas.DataFrame."""
    import pandas as pd
    rows = []
    for (w, k), c in model["coefs"].items():
        a, b = model.get("windows", WINDOWS)[w]
        rows.append({"window": w, "wl_min": a, "wl_max": b, "term": k, "form": model["forms"][(w, k)],
                     "c0": c[0], "c1": c[1], "c2": c[2], "c3": c[3], "c4": c[4] if len(c) > 4 else 0.,
                     "rms": model["rms"].get((w, k), np.nan),
                     **model["ranges"][(w, k)]})
    return pd.DataFrame(rows)


def write_table(model, path=TABLE, source="v2"):
    """Write the table of radcalnet_oc (kernel.GasCoupling)."""
    table = table_frame(model)
    with open(path, "w") as f:
        f.write("# Gaseous transmittance of the scattered light (radcalnet_oc.kernel.GasCoupling), per absorption\n"
                "# window and term (atm: path reflectance, w: water-leaving term, m = 1/mu0 + 1/muv; Ed: downwelling\n"
                "# irradiance, m = 1/mu0):\n"
                "#   c = c0 + c1 (m - 2) + c2 (aot550 - 0.1) + c3 ln(U / 2 g cm-2) + c4 tau, U water vapour column\n"
                "#   (g cm-2), tau = -ln Tg(m) at the 5 cm-1 resolution of the fit (blend form only),\n"
                "#   form eta  : T = Tg(c m)\n"
                "#   form blend: T = Tg(m)^(1-c) Tss^c, Tss = (1 - Tg(m)) / (-ln Tg(m))\n"
                "# m, aot550, U and tau clipped to [m_min, m_max], [aot_min, aot_max], [h2o_min, h2o_max], [0, tau_max];\n"
                "# transmittances smoothed to 5 cm-1 for the correction (GasCoupling.correction); rms of the fit\n"
                "# residual (eta, or ln T for blend).\n"
                f"# SOS-ABS coupled simulations, set {source} of scripts/sos_abs_coupling/make_luts.py (CKD 5 cm-1,\n"
                "# US62, WMO maritime aerosols, Cox-Munk 5 m/s, 1013 hPa), fitted by scripts/sos_abs_coupling/coupling.py\n")
        table.to_csv(f, index=False, float_format="%.5g")
    return table


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "fit":
        sys.exit(__doc__)
    d = sys.argv[2] if len(sys.argv) > 2 else DATA_V2
    out = sys.argv[3] if len(sys.argv) > 3 else TABLE
    eff = effective(load(d))
    table = write_table(fit_all(eff, FORMS_ADOPTED, NO_TAU), path=out, source=os.path.basename(os.path.normpath(d)))
    print(table.round(4).to_string())
    print("written:", os.path.abspath(out))
