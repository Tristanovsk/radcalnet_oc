"""
Gaseous absorption-scattering coupling over 364-2500 nm: SOS-ABS simulations v2 (684-990 nm) and v3 (the rest
of 364-2500 nm, see make_luts.py), analysed with the tools of coupling.py (decomposition, effective
transmittances) and ppld.py (photon path-length model).

The windows of the 684-990 nm parameterization of radcalnet_oc are kept; new windows are defined from the
dominant absorber of the sampled CKD intervals. The study is in
notebook/research/radcalnet_oc_gaseous_coupling_350_2500nm.ipynb.

Usage: python fullrange.py fit        # fit and write the candidate table gas_coupling_ppld_full.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupling as cp  # noqa: E402
import ppld  # noqa: E402

DATA_V3 = os.environ.get("SOS_ABS_V3", os.path.join(cp.DATA, "v3"))
TABLE_FULL = os.path.join(HERE, "gas_coupling_ppld_full.csv")

# windows of the 684-990 nm parameterization (coupling.WINDOWS) and new windows, by dominant absorber
WINDOWS_684_990 = dict(cp.WINDOWS)
WINDOWS_NEW = {
    "O3-VIS": (364, 683),          # O3 Chappuis (with NO2, weak H2O 590 / 650 nm and O2-gamma 628 nm lines)
    "H2O-1000": (990, 1047),       # long-wave wing of the 940 nm band
    "H2O-1130": (1047, 1252),      # H2O 1130 nm (O4 1060 nm: not in the CKD of SOS-ABS)
    "O2-1270": (1252, 1280),       # O2 a-band 1270 nm
    "H2O-1380": (1280, 1568),      # H2O 1380 nm
    "CO2-1600": (1568, 1634),      # CO2 1575 / 1605 nm
    "CH4-1660": (1634, 1700),      # CH4 2nu3 1665 nm
    "H2O-1880": (1700, 1998),      # H2O 1880 nm (with CO2 1960 nm lines)
    "CO2-2010": (1998, 2091),      # CO2 2010 / 2060 nm
    "H2O-2150": (2091, 2228),      # H2O (wing of the 1880 / 2700 nm bands)
    "CH4-2300": (2228, 2380),      # CH4 2300 nm
    "H2O-2450": (2380, 2500),      # H2O (wing of the 2700 nm band)
}
WINDOWS_FULL = {**WINDOWS_NEW, **WINDOWS_684_990}
WINDOWS_FULL = dict(sorted(WINDOWS_FULL.items(), key=lambda kv: kv[1][0]))
# dominant absorber class of each window (vertical distribution relative to the scatterers)
CLASS = {w: ("O3 (stratospheric)" if w.startswith("O3") else
             "H2O (boundary layer)" if w.startswith("H2O") else "well mixed (O2, CO2, CH4)")
         for w in WINDOWS_FULL}
# windows with a water vapour term in the model (all but the visible O3 window)
H2O_WINDOWS_FULL = tuple(w for w in WINDOWS_FULL if w != "O3-VIS")
GASES = ["h2o", "o3", "no2", "ch4", "co2", "o2", "co", "n2o"]


def use_full_windows():
    """Point coupling.py / ppld.py at the full-range windows."""
    cp.WINDOWS.clear()
    cp.WINDOWS.update(WINDOWS_FULL)
    cp.H2O_WINDOWS = H2O_WINDOWS_FULL
    cp.RANGES = [WINDOWS_FULL[w] for w in WINDOWS_FULL]


def window_of(wl):
    """Name of the window of each wavelength (nm), '' outside."""
    out = np.full(len(wl), "", dtype=object)
    for w, (a, b) in WINDOWS_FULL.items():
        out[(wl >= a) & (wl < b)] = w
    return out


# ---------------------------------------------------------------------------
# simulations
# ---------------------------------------------------------------------------
def _concat(a, b):
    return xr.concat([a, b], "wl").sortby("wl")


def load_full(tg_min=0.0, tg_max=0.999):
    """Effective transmittances of v2 and v3 merged along wl (dict as coupling.effective), restricted to the
    channels whose CKD direct transmittance at m = 2, U = 2 g cm-2 is in [tg_min, tg_max] (absorbing
    channels; the O4 intervals of the selection are transparent in SOS-ABS)."""
    luts = {s: cp.load(d) for s, d in (("v2", cp.DATA_V2), ("v3", DATA_V3))}
    # same grids up to the float precision of the Gauss viewing angles (< 0.01 deg) and of the air masses:
    # v3 is put on the coordinates of v2
    for k, c, tol in (("abs", "vza", 0.01), ("noabs", "vza", 0.01), ("tg", "m", 1e-6)):
        ref, x = luts["v2"][k][c].values, luts["v3"][k][c].values
        if len(ref) != len(x) or np.abs(ref - x).max() > tol:
            raise ValueError(f"v2 and v3 differ in {k}.{c}")
        luts["v3"][k] = luts["v3"][k].assign_coords({c: ref})
    eff = {s: cp.effective(lut) for s, lut in luts.items()}
    out = {"geo": eff["v2"]["geo"],
           "T": _concat(eff["v2"]["T"], eff["v3"]["T"]),
           "eta": _concat(eff["v2"]["eta"], eff["v3"]["eta"]),
           "tg": _concat(eff["v2"]["tg"], eff["v3"]["tg"]),
           "dec": {k: _concat(eff["v2"]["dec"][k], eff["v3"]["dec"][k]) for k in ("abs", "noabs")}}
    t2 = cp.tg_of_m(out["tg"], 2.).sel(h2o=2.0)
    keep = ((t2 >= tg_min) & (t2 <= tg_max)).values
    for k in ("T", "eta", "tg"):
        out[k] = out[k].isel(wl=keep)
    out["dec"] = {k: v.isel(wl=keep) for k, v in out["dec"].items()}
    wl = out["T"].wl.values
    out["window"] = xr.DataArray(window_of(wl), coords={"wl": wl}, dims="wl")
    out["set"] = xr.DataArray(np.where((wl >= 684) & (wl <= 990), "v2", "v3"), coords={"wl": wl}, dims="wl")
    return out


def dominant_gas(wl, h2o=20.):
    """Dominant absorber and absorption optical thickness per gas (radcalnet_oc table, 5 cm-1 interval
    around each wavelength, default columns of radcalnet_oc and h2o kg m-2)."""
    import radcalnet_oc as radoc
    g = radoc.LUT().gas_lut
    d = radoc.kernel.Gases()
    cols = {"h2o": h2o, "o3": d.gas_tc["o3"], "no2": d.gas_tc["no2"], "ch4": d.gas_tc["ch4"]}
    nu = 1e7 / g.wl.values
    rows = []
    for w in wl:
        nc = 1e7 / w
        m = (nu >= nc - 2.5) & (nu < nc + 2.5)
        r = {k: float(g[k].values[m].mean()) * cols.get(k, 1.) for k in GASES if k in g}
        rows.append(r)
    tau = pd.DataFrame(rows, index=pd.Index(wl, name="wl"))
    return tau.idxmax(axis=1), tau


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
TRUE = {"atm": "Tatm", "w": "Tw", "Ed": "TEd"}


def decoupled(eff, term):
    """Transmittance of the decoupled model (direct beam of the term)."""
    return eff["T"]["Tg_m0" if term == "Ed" else "Tg_m"]


def channel_error(eff, model_t, term):
    """Absolute transmittance error T_model - T_true (fraction of the signal without absorption)."""
    err = model_t - eff["T"][TRUE[term]]
    return err if term == "Ed" else err.where(eff["geo"].valid)


def gauss_band(center, fwhm, wl):
    return np.exp(-4 * np.log(2) * ((wl - center) / fwhm) ** 2)


# strides of make_luts.py (V3_REGIONS) and steps of v2, to weight each sample by the spectral interval
# (cm-1) it represents
V3_STRIDES = [((364, 684), 11), ((990, 1300), 4), ((1300, 1500), 2), ((1500, 1800), 3), ((1800, 2000), 1),
              ((2000, 2500), 2)]
V2_STEPS = [((684, 698), 0.5), ((712, 736), 1.0), ((757, 772), 0.5), ((808, 836), 1.0), ((890, 990), 1.0)]


def sample_width(wl):
    """Spectral interval (cm-1) represented by each sample: stride x 5 cm-1 (v3), step converted to cm-1
    (v2)."""
    out = np.full(len(wl), np.nan)
    for (a, b), s in V3_STRIDES:
        out[(wl >= a) & (wl < b)] = 5. * s
    for (a, b), s in V2_STEPS:
        sel = (wl >= a) & (wl <= b)
        out[sel] = s * 1e7 / wl[sel] ** 2
    return out


def band_weights(eff, center, fwhm):
    """Weights of a Gaussian band on the samples: response x solar irradiance (TSIS) x sample width."""
    from radcalnet_oc import SolarIrradiance
    wl = eff["T"].wl.values
    f0 = SolarIrradiance().tsis.interp(wl=wl).values
    return xr.DataArray(gauss_band(center, fwhm, wl) * f0 * sample_width(wl), coords={"wl": wl}, dims="wl")


def depth_error(eff, model_t, term, center, fwhm, base=None):
    """Error of the band absorption depth, in % of the absorption: sum w R (T_model - T_true) / sum w R (1 -
    T_true) over the sampled channels of a Gaussian band (the sampled CKD intervals are a uniform subset of
    the absorbing ones, the transparent intervals add neither absorption nor error). w: band_weights."""
    w = band_weights(eff, center, fwhm)
    t = eff["T"][TRUE[term]]
    if base is None:
        base = (eff["dec"]["noabs"].R0 if term == "atm" else eff["dec"]["noabs"].A if term == "w"
                else eff["T"].Ed_noabs)
    num = xr.dot((model_t - t).fillna(0), w * base, dim="wl")
    den = xr.dot((1 - t).fillna(0), w * base, dim="wl")
    out = num / den * 100
    return out.where(eff["geo"].valid) if term != "Ed" else out


def band_depth(center, fwhm, air_mass=2., h2o=20.):
    """Absorption depth 1 - T_band of a Gaussian band for the direct beam (radcalnet_oc table, REPTRAN O2,
    default columns, h2o kg m-2), solar weighting: the band TOA error is about depth_error x band_depth."""
    import radcalnet_oc as radoc
    gt = radoc.GaseousTransmittance(radoc.LUT().gas_lut, o2_model="reptran")
    gt.pressure, gt.air_mass = 1013., air_mass
    gt.gas_tc["h2o"] = h2o
    tg = gt.get_gaseous_transmittance()
    wl = tg.wl.values
    w = gauss_band(center, fwhm, wl) * radoc.SolarIrradiance().tsis.interp(wl=wl).values
    return 1 - float((w * tg.values).sum() / w.sum())


# ---------------------------------------------------------------------------
# photon path-length model on the full range
# ---------------------------------------------------------------------------
def fit_full(eff, a, windows=None, exclude=None, n=60000):
    """PPLD coefficients per (window, term), coupling.py / ppld.py forms, on the given windows."""
    use_full_windows()
    windows = windows or [w for w in WINDOWS_FULL if (eff["window"] == w).any()]
    model = dict(coefs={}, rms={}, ranges={}, windows={w: WINDOWS_FULL[w] for w in windows})
    sel = lambda x: cp.subset(x, exclude)
    geo, eta = eff["geo"], eff["eta"]
    for w in windows:
        for k in cp.TERMS:
            try:
                ce, cs, rms = ppld.fit_term(eff, a, w, k, n=n, exclude=exclude)
            except Exception as e:                      # too few absorbing channels
                print(f"{w} {k}: not fitted ({e})")
                continue
            model["coefs"][(w, k)] = (ce, cs)
            model["rms"][(w, k)] = rms
            m = sel(geo.m0 if k == "Ed" else geo.m.where(geo.valid))
            e = sel(eta)
            model["ranges"][(w, k)] = dict(m_min=float(m.min()), m_max=float(m.max()),
                                           aot_min=float(e.aot_ref.min()), aot_max=float(e.aot_ref.max()),
                                           h2o_min=float(e.h2o.min()), h2o_max=float(e.h2o.max()))
    return model


def fit_class(eff, a, n=60000, exclude=None):
    """One PPLD model per absorber class (windows of a class pooled): test of a compact parameterization."""
    use_full_windows()
    out = {}
    for c in sorted(set(CLASS.values())):
        ws = [w for w in WINDOWS_FULL if CLASS[w] == c and (eff["window"] == w).any()]
        if not ws:
            continue
        lo = min(WINDOWS_FULL[w][0] for w in ws)
        name = "class:" + c
        # pooled samples: fit on the union of the windows of the class (temporary window covering them,
        # restricted to the channels of the class)
        mask = eff["window"].isin(ws)
        sub = {**eff, "T": eff["T"].where(mask), "eta": eff["eta"].where(mask)}
        cp.WINDOWS[name] = (lo, max(WINDOWS_FULL[w][1] for w in ws))
        cp.H2O_WINDOWS = tuple(cp.H2O_WINDOWS) + ((name,) if c != "O3 (stratospheric)" else ())
        for k in cp.TERMS:
            ce, cs, rms = ppld.fit_term(sub, a, name, k, n=n, exclude=exclude)
            for w in ws:
                out[(w, k)] = (ce, cs)
    use_full_windows()
    return out


def apply_full(eff, model, a, term, coefs=None):
    """Transmittance of a term with the PPLD model, on the channels of the fitted windows (nan elsewhere)."""
    use_full_windows()
    coefs = coefs or model["coefs"]
    T, geo = eff["T"], eff["geo"]
    m = geo.m0 if term == "Ed" else geo.m
    parts = []
    for w in model["windows"]:
        if (w, term) not in coefs:
            continue
        lo, hi = WINDOWS_FULL[w]
        ce, cs = coefs[(w, term)]
        rng = model["ranges"].get((w, term)) or next(iter(model["ranges"].values()))
        mm, aa, uu = ppld._clip(rng, m, T.aot_ref, T.h2o)
        eta = ppld.linear(ce, mm, aa, uu, w)
        s2 = np.maximum(ppld.linear(cs, mm, aa, uu, w), ppld.S2_MIN)
        aw = a.sel(wl=slice(lo, hi - 1e-9))
        if aw.sizes["wl"]:
            parts.append(ppld.transmittance(aw, m, eta, s2))
    out = xr.concat(parts, "wl")
    t0 = T.Tg_m0 if term == "Ed" else T.Tg_m
    out = out.broadcast_like(t0).reindex(wl=t0.wl)
    return out.where(np.isfinite(t0)) if term != "Ed" else out


def table_frame(model):
    """Candidate table of radcalnet_oc (GasCoupling form 'ppld') for the full range."""
    use_full_windows()
    return ppld.table_frame(model)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "fit":
        sys.exit(__doc__)
    eff = load_full()
    a, rms = ppld.exponential_sum(eff["tg"])
    model = fit_full(eff, a)
    t = table_frame(model)
    t.to_csv(TABLE_FULL, index=False, float_format="%.6g")
    print(t.round(4).to_string())
    print("written", TABLE_FULL)
