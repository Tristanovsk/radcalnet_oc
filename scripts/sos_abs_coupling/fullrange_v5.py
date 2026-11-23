"""
Saturated cores of the H2O 1380 and 1880 nm bands in the full-range coupling (ROADMAP task 14).

Set v3 of SOS-ABS skipped the CKD intervals with Tg < 0.02 (air mass 2, 2 g cm-2); set v5 runs them (one interval out
of two, same grid as v3, make_luts.py v5). The photon path-length models of the windows H2O-1380 (1280-1568 nm) and
H2O-1880 (1700-1998 nm) were fitted on the wings only (fullrange.py): in the cores, the path reflectance is the
scattering above the water vapour, and the model recovers only 40 % of the clear-sky floor of MSI B10 (0.20 against
0.50e-3 observed, check_fullrange_s2.py).

This script merges v3 and v5 in the two windows, refits the windows on v3 alone (as the candidate table) and on
v3 + v5, and compares both with SOS-ABS:

- channel error of the path transmittance T_atm in the cores (v5 channels) and in the wings (v3 channels);
- band path reflectance of dark water in MSI B10 (S2A response): SOS-ABS, decoupled model (direct transmittance of
  the whole path), PPLD v3 and PPLD v3 + v5, per AOT, air mass and water vapour.

Writes gas_coupling_ppld_full_v5.csv: the candidate table with the two windows refitted on v3 + v5.

Usage: python fullrange_v5.py [--n 60000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coupling as cp  # noqa: E402
import fullrange as fr  # noqa: E402
import ppld  # noqa: E402

DATA_V5 = os.environ.get("SOS_ABS_V5", "/data/sos_abs_coupling/v5")
TABLE_V5 = os.path.join(HERE, "gas_coupling_ppld_full_v5.csv")
WINDOWS = ["H2O-1380", "H2O-1880"]
# MSI B10 (S2A): central wavelength and FWHM of a Gaussian approximation of its spectral response
B10 = (1373.5, 31.)


def load_set(d, ref=None):
    """Effective transmittances of a set restricted to the two windows, on the coordinates of ``ref``."""
    lut = cp.load(d)
    if ref is not None:
        for k, c, tol in (("abs", "vza", 0.01), ("noabs", "vza", 0.01), ("tg", "m", 1e-6)):
            a, b = ref[k][c].values, lut[k][c].values
            if len(a) != len(b) or np.abs(a - b).max() > tol:
                raise ValueError(f"{d} differs from the reference in {k}.{c}")
            lut[k] = lut[k].assign_coords({c: a})
    lo, hi = min(fr.WINDOWS_FULL[w][0] for w in WINDOWS), max(fr.WINDOWS_FULL[w][1] for w in WINDOWS)
    lut = {k: (v.sel(wl=slice(lo, hi)) if "wl" in v.dims else v) for k, v in lut.items()}
    return lut, cp.effective(lut)


def merge(effs, names):
    """Effective transmittances of several sets merged along wl, with the window and the set of each channel."""
    out = {"geo": effs[0]["geo"],
           "T": xr.concat([e["T"] for e in effs], "wl"),
           "eta": xr.concat([e["eta"] for e in effs], "wl"),
           "tg": xr.concat([e["tg"] for e in effs], "wl"),
           "dec": {k: xr.concat([e["dec"][k] for e in effs], "wl") for k in ("abs", "noabs")}}
    tags = np.concatenate([np.full(e["T"].sizes["wl"], n) for e, n in zip(effs, names)])
    order = np.argsort(out["T"].wl.values, kind="stable")
    for k in ("T", "eta", "tg"):
        out[k] = out[k].isel(wl=order)
    out["dec"] = {k: v.isel(wl=order) for k, v in out["dec"].items()}
    wl = out["T"].wl.values
    out["window"] = xr.DataArray(fr.window_of(wl), coords={"wl": wl}, dims="wl")
    out["set"] = xr.DataArray(tags[order], coords={"wl": wl}, dims="wl")
    return out


def subset(eff, keep):
    out = dict(eff)
    for k in ("T", "eta", "tg", "window", "set"):
        out[k] = eff[k].isel(wl=keep)
    out["dec"] = {k: v.isel(wl=keep) for k, v in eff["dec"].items()}
    return out


def widths(eff):
    """Spectral interval (cm-1) of each sample: stride x 5 cm-1 of its set (v3 strides, v5 one out of two)."""
    wl = eff["T"].wl.values
    w = fr.sample_width(wl)
    w[eff["set"].values == "v5"] = 10.
    return w


def band_path_reflectance(eff, t, center, fwhm):
    """Band path reflectance of dark water: sum w R0_noabs T / sum w (w: response x TSIS x sample width), the band
    integral over all the CKD intervals (transparent intervals, absent from the sets, are not in the band cores)."""
    from radcalnet_oc import SolarIrradiance
    wl = eff["T"].wl.values
    f0 = SolarIrradiance().tsis.interp(wl=wl).values
    w = xr.DataArray(fr.gauss_band(center, fwhm, wl) * f0 * widths(eff), coords={"wl": wl}, dims="wl")
    r0 = eff["dec"]["noabs"].R0
    return xr.dot((r0 * t).fillna(0), w, dim="wl") / float(w.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60000, help="samples of each fit")
    args = p.parse_args()

    lut3, eff3 = load_set(fr.DATA_V3)
    _, eff5 = load_set(DATA_V5, ref=lut3)
    # v3 keeps the absorbing channels as fullrange.load_full (Tg <= 0.999 at m = 2, 2 g cm-2)
    t2 = cp.tg_of_m(eff3["tg"], 2.).sel(h2o=2.0)
    eff3 = subset(merge([eff3], ["v3"]), (t2 <= 0.999).values)
    eff5 = merge([eff5], ["v5"])
    both = merge([{k: eff3[k] for k in ("geo", "T", "eta", "tg", "dec")},
                  {k: eff5[k] for k in ("geo", "T", "eta", "tg", "dec")}], ["v3", "v5"])
    print(f"channels: v3 {eff3['T'].sizes['wl']}, v5 {eff5['T'].sizes['wl']}; "
          f"per window: {pd.Series(both['window'].values).value_counts().to_dict()}")

    a, rms = ppld.exponential_sum(both["tg"])
    a3 = a.sel(wl=eff3["T"].wl)
    print(f"exponential sums of the direct transmittance: rms {float(np.nanmax(rms)):.2e} (max)")

    fits = {"PPLD v3": fr.fit_full(eff3, a3, windows=WINDOWS, n=args.n),
            "PPLD v3+v5": fr.fit_full(both, a, windows=WINDOWS, n=args.n)}

    # channel errors of T_atm on all the channels (cores = v5)
    rows = []
    T = both["T"]
    models = {"decoupled": fr.decoupled(both, "atm")}
    models.update({k: fr.apply_full(both, m, a, "atm") for k, m in fits.items()})
    for name, t in models.items():
        err = fr.channel_error(both, t, "atm")
        for s in ("v3", "v5"):
            for w in WINDOWS:
                sel = (both["set"] == s) & (both["window"] == w)
                e = err.isel(wl=np.flatnonzero(sel.values))
                rows.append({"model": name, "window": w, "channels": "cores (v5)" if s == "v5" else "wings (v3)",
                             "bias": float(e.mean()), "rms": float(np.sqrt((e ** 2).mean())),
                             "T_true": float(T.Tatm.isel(wl=np.flatnonzero(sel.values)).mean())})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nchannel error of T_atm (model - SOS-ABS), mean over the geometries, AOT and H2O:")
    print(tab.pivot_table(index=["window", "channels"], columns="model", values=["bias", "rms"]).round(4)
          .to_string())

    # band path reflectance of dark water in MSI B10
    true = band_path_reflectance(both, T.Tatm, *B10)
    band = {"SOS-ABS": true}
    band.update({k: band_path_reflectance(both, t, *B10) for k, t in models.items()})
    geo = both["geo"]
    df = xr.Dataset({k: v.where(geo.valid) for k, v in band.items()}).to_dataframe().reset_index()
    df = df.dropna(subset=["SOS-ABS"])
    df["m"] = np.round(df.m, 1) if "m" in df else np.nan
    keys = [k for k in ("aot_ref", "h2o") if k in df]
    print("\nMSI B10 path reflectance of dark water (1e-3), median over the geometries:")
    print((df.groupby(keys)[list(band)].median() * 1e3).round(3).to_string())
    print("\nratio model / SOS-ABS (median over all):")
    print({k: round(float((df[k] / df["SOS-ABS"]).median()), 3) for k in band if k != "SOS-ABS"})

    # candidate table with the two windows refitted on v3 + v5
    full = pd.read_csv(fr.TABLE_FULL)
    new = fr.table_frame(fits["PPLD v3+v5"])
    out = pd.concat([full[~full.window.isin(WINDOWS)], new[new.window.isin(WINDOWS)]])
    out = out.sort_values("wl_min", kind="stable").reset_index(drop=True)
    out.to_csv(TABLE_V5, index=False, float_format="%.6g")
    print("\nrefitted rows:")
    print(new[new.window.isin(WINDOWS)].round(4).to_string())
    print("written", TABLE_V5)


if __name__ == "__main__":
    main()
