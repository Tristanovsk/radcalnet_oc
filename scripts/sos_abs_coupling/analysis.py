"""Decompose the coupled SOS-ABS TOA signal and derive effective gaseous transmittances / air masses."""
import os
import sys
import numpy as np
import xarray as xr

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WINDOWS = {"O2-B": (684, 698), "H2O-720": (712, 736), "O2-A": (757, 772), "H2O-820": (808, 836), "H2O-940": (890, 990)}


def tg_dir_table(d=D):
    """CKD direct transmittance Tg(wl, m) from the flux ratio (m = 1/mu0)."""
    a = xr.open_dataset(f"{d}/tgdir_abs.nc").flux_dir_down
    n = xr.open_dataset(f"{d}/tgdir_noabs.nc").flux_dir_down
    tg = (a / n).squeeze(drop=True)
    tg = tg.assign_coords(m=("sza", np.round(1 / np.cos(np.radians(tg.sza.values)), 3))).swap_dims(sza="m").drop_vars("sza")
    tg["wl"] = np.round(tg.wl * 1e3, 2)
    return tg.sortby("m")


def tg_of_m(tg, m):
    """Tg_dir at air mass m (DataArray), log-interpolation in m."""
    return np.exp(np.log(tg).interp(m=m))


def m_of_tg(tg, t):
    """Effective air mass m* such that Tg_dir(m*) = t (per wl); nan if out of the table."""
    lt = np.log(tg.transpose("wl", "m").values)
    mm = tg.m.values
    lt_t = np.log(t.transpose("wl", ...).values)
    out = np.full(lt_t.shape, np.nan)
    for i in range(lt.shape[0]):
        x = lt[i][::-1]  # increasing
        if np.all(np.diff(x) > 0):
            v = lt_t[i]
            out[i] = np.interp(v, x, mm[::-1], left=np.nan, right=np.nan)
    return t.transpose("wl", ...).copy(data=out)


def decompose(d=D):
    """R(rho) = R0 + rho A / (1 - S rho) for the abs and noabs LUTs (TOA reflectance R = I / mu0)."""
    res = {}
    for k in ("abs", "noabs"):
        ds = xr.open_dataset(f"{d}/{k}.nc")
        ds["wl"] = np.round(ds.wl * 1e3, 2)
        R = ds.I_up / np.cos(np.radians(ds.sza))
        rho = R.albedo
        R0 = R.sel(albedo=0)
        dR = (R - R0).sel(albedo=rho[rho > 0])
        y = dR.albedo / dR                           # = 1/A - (S/A) rho
        x = y.albedo
        xm, ym = x.mean(), y.mean("albedo")
        b = ((x - xm) * (y - ym)).sum("albedo") / ((x - xm) ** 2).sum()
        a = ym - b * xm
        A = 1 / a
        S = -b * A
        res[k] = xr.Dataset({"R": R, "R0": R0, "A": A, "S": S})
    return res


def geometry_airmass(ds):
    return 1 / np.cos(np.radians(ds.sza)) + 1 / np.cos(np.radians(ds.vza))


if __name__ == "__main__":
    tg = tg_dir_table()
    r = decompose()
    A, N = r["abs"], r["noabs"]
    m = geometry_airmass(A)
    Tw = A.A / N.A
    T_m = tg_of_m(tg, m).broadcast_like(Tw)         # decoupled model (radcalnet_oc)
    Tatm = A.R0 / N.R0
    sel = dict(vza=slice(0, 61))
    off = lambda x: x.sel(**sel).where((x.azi >= 60) & (x.azi <= 300))   # off-glint geometries
    print("check fit: S noabs range", float(N.S.sel(**sel).min()), float(N.S.sel(**sel).max()))
    # fit residual at rho=0.1 (exact? reconstruct)
    for k in ("abs", "noabs"):
        Rf = r[k].R0 + 0.1 * r[k].A / (1 - r[k].S * 0.1)
        print(k, "max rel. err of 3-param fit at rho=0.1:", float(abs(Rf / r[k].R.sel(albedo=0.1) - 1).sel(**sel).max()))
    print("\nwindow      min Tg_dir(m) | Tw/Tdir(m) mean[min,max] | Tatm/Tdir(m) mean[min,max]  (off-glint, vza<=60)")
    for w, (a, b) in WINDOWS.items():
        s = slice(a, b)
        tdir = off(T_m.sel(wl=s))
        mask = tdir < 0.95
        rw = (off(Tw.sel(wl=s)) / tdir).where(mask)
        ra = (off(Tatm.sel(wl=s)) / tdir).where(mask)
        print(f"{w:9s} {float(tdir.min()):.3f} | {float(rw.mean()):.3f} [{float(rw.min()):.3f},{float(rw.max()):.3f}]"
              f" | {float(ra.mean()):.3f} [{float(ra.min()):.3f},{float(ra.max()):.3f}]")
