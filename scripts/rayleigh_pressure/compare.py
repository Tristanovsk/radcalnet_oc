"""OSOAA pressure runs against the Rayleigh pressure scaling of radcalnet_oc (Process.rayleigh_pressure_correction)."""
import os, sys
import numpy as np, pandas as pd, xarray as xr
HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))  # folder of run_pressure.py outputs
LUT = '/data/vrtc/lut/atmo_full'
WL_NODES = [0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1., 1.6, 2.2, 2.5]
P0 = 1015.2
SZAS, VZAS, AZIS = [20., 40., 60.], [0., 10., 20., 30., 40.], [0., 45., 90., 135., 180.]


def toa(f, sza=None):
    """TOA path reflectance I / mu0 (sza, vza, azi) of an OSOAA file."""
    s = xr.open_dataset(f).sza.values
    I = xr.open_dataset(f, group='stokes/up').I.isel(z=-1).assign_coords(sza=s)
    I = I.sel(sza=SZAS).interp(vza=VZAS).sel(azi=AZIS)
    return I / np.cos(np.radians(I.sza))


# D from the LUT nodes (aot 0, wind 2), as in Process: gradient of ln R_ray against ln tau_r along the nodes
R, tau = [], []
for wl in WL_NODES:
    f = f'{LUT}/osoaa_atmo_full_aot0.0000_aerCOAV_rh70_HR8.0_HA2.0_ws2_wl{wl:0.3f}.nc'
    R.append(toa(f)); tau.append(float(xr.open_dataset(f).attrs['AP.MOT']))
R = xr.concat(R, 'wl').assign_coords(wl=WL_NODES)
D = R.copy(data=np.gradient(np.log(R.values), np.log(tau), axis=0))

rows = []
for aot in ['0.0000', '0.1']:
    for wl in [0.4, 0.5, 0.8]:
        name = lambda P: f'{HERE}/out/osoaa_atmo_full_aot{aot}_aerCOAV_rh70_HR8.0_HA2.0_ws2_wl{wl:0.3f}_P{P:.1f}.nc'
        if not all(os.path.exists(name(P)) for P in [995.2, 1015.2, 1035.2]):
            continue
        ref = toa(name(1015.2))
        for P in [995.2, 1035.2]:
            osoaa = toa(name(P)) - ref
            pred = R.sel(wl=wl) * ((P / P0) ** D.sel(wl=wl) - 1)
            err = (pred - osoaa) / ref           # error of the scaling relative to the path reflectance
            rows.append({'aot': float(aot), 'wl nm': int(wl * 1e3), 'P': P, 'D median': float(D.sel(wl=wl).median()),
                         'OSOAA dR/R %': 100 * float((osoaa / ref).median()),
                         'scaling dR/R %': 100 * float((pred / ref).median()),
                         'err median % of R': 100 * float(err.median()),
                         'err max % of R': 100 * float(np.abs(err).max()),
                         'err rel. to dR %': 100 * float((np.abs(pred - osoaa) / np.abs(osoaa)).median()),
                         'single-scat. tau ratio %': 100 * (P / P0 - 1)})
pd.set_option('display.width', 220)
print(pd.DataFrame(rows).round(3).to_string(index=False))
