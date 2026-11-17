"""Wavelength interpolation of the TOA LUT against the OSOAA runs between its nodes.

The TOA LUT of radcalnet_oc keeps 10 wavelength nodes (0.35-2.5 um) of the OSOAA runs of /data/vrtc/lut/atmo_full;
the runs at 0.45, 0.55, 0.65, 0.75, 0.85, 0.9 and 2.4 um are left out of it, and give the true TOA path reflectance
between the nodes; the runs of run_offnode.py (1.15, 1.25, 1.4 um, OFFNODE) check the gap between 1.0 and 1.6 um.
The geometries without a valid run (NaN, e.g. some solar angles at 0.9 um) are left out. For each interpolation scheme, the TOA path reflectance (I / mu0, TOA level of the runs) is
interpolated from the 10 nodes and compared with the OSOAA run at the left-out wavelength, wind 2 m s-1, all the
aerosol models and AOTs available, solar zenith angles 20, 40, 60 deg, VZA 0-40 deg, azimuths 0-180 deg. The
spectral AOT of the runs (optical_properties/aerosol: aot) is compared in the same way.

Usage: python compare_schemes.py [output csv]
Writes the errors per case (csv) and prints the summary per wavelength and scheme.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline

LUT = '/data/vrtc/lut/atmo_full'
WL_NODES = np.array([0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1., 1.6, 2.2, 2.5])
OFFNODE = '/data/vrtc/lut/osoaa_offnode/out'
WL_TEST = [0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 1.15, 1.25, 1.4, 2.4]
WIND = '2'
SZAS, VZAS, AZIS = [20., 40., 60.], [0., 10., 20., 30., 40.], [0., 45., 90., 135., 180.]
HERE = os.path.dirname(os.path.abspath(__file__))


def scheme(kind, space):
    '''Interpolator of the values y (nodes on axis 0) at x, in linear or log-log space.'''
    def f(x_nodes, y, x):
        if space == 'loglog':
            x_nodes, x, y = np.log(x_nodes), np.log(x), np.log(y)
        if kind == 'pchip':
            v = PchipInterpolator(x_nodes, y, axis=0)(x)
        elif kind == 'spline':
            v = CubicSpline(x_nodes, y, axis=0, bc_type='natural')(x)
        else:
            v = interp1d(x_nodes, y, kind=kind, axis=0)(x)
        return np.exp(v) if space == 'loglog' else v
    return f


SCHEMES = {'quadratic_linear (current)': scheme('quadratic', 'linear'),
           'linear_linear': scheme('linear', 'linear'),
           'linear_loglog': scheme('linear', 'loglog'),
           'quadratic_loglog': scheme('quadratic', 'loglog'),
           'pchip_loglog': scheme('pchip', 'loglog'),
           'spline_loglog': scheme('spline', 'loglog')}


def name(aot, model, wl):
    base = f'osoaa_atmo_full_aot{aot}_aer{model}_HR8.0_HA2.0_ws{WIND}_wl{wl:0.3f}.nc'
    return f'{LUT}/{base}' if os.path.exists(f'{LUT}/{base}') else f'{OFFNODE}/{base}'


def toa(f):
    '''TOA path reflectance I / mu0 (sza, vza, azi) and AOT of an OSOAA file.'''
    s = xr.open_dataset(f).sza.values
    with xr.open_dataset(f, group='stokes/up') as up:
        I = up.I.isel(z=-1).assign_coords(sza=s).sel(sza=SZAS).interp(vza=VZAS).sel(azi=AZIS).load()
    try:
        with xr.open_dataset(f, group='optical_properties/aerosol') as aer:
            aot = float(aer.aot)
    except (OSError, KeyError, ValueError, AttributeError):  # metadata not written by the converter
        aot = np.nan
    return (I / np.cos(np.radians(I.sza))).values, aot


def cases():
    '''(aot, model) with the runs at all the nodes and at least one test wavelength.'''
    out = set()
    for f in glob.glob(f'{LUT}/*_ws{WIND}_wl*.nc'):
        m = re.search(r'_aot([0-9.]+)_aer(.+)_HR8.0_HA2.0_ws[0-9.]+_wl([0-9.]+)\.nc', f)
        out.add((m.group(1), m.group(2)))
    return sorted(c for c in out if all(os.path.exists(name(*c, w)) for w in WL_NODES)
                  and any(os.path.exists(name(*c, w)) for w in WL_TEST))


def main(ofile=f'{HERE}/compare_schemes.csv'):
    rows = []
    for aot, model in cases():
        nodes = [toa(name(aot, model, w)) for w in WL_NODES]
        R = np.stack([n[0] for n in nodes])
        tau = np.array([n[1] for n in nodes])
        for wl in WL_TEST:
            f = name(aot, model, wl)
            if not os.path.exists(f):
                continue
            truth, tau_true = toa(f)
            for label, interp in SCHEMES.items():
                est = interp(WL_NODES, R, wl)
                rel = (est / truth - 1).ravel()
                ok = np.isfinite(rel)
                if not ok.any():
                    continue
                rel, diff = rel[ok], (est - truth).ravel()[ok]
                row = {'aot': float(aot), 'model': model, 'wl': wl, 'scheme': label,
                       'R_median': float(np.nanmedian(truth)), 'n_geometries': int(ok.sum()),
                       'rel_median': float(np.median(rel)), 'abs_rel_median': float(np.median(np.abs(rel))),
                       'abs_rel_max': float(np.abs(rel).max()),
                       'abs_median': float(np.median(diff))}
                if float(aot) > 0:
                    row['aot_rel'] = float(interp(WL_NODES, tau, wl) / tau_true - 1)
                rows.append(row)
        print(f'{model} aot {aot} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ofile, index=False)

    pd.set_option('display.width', 200)
    print('\nTOA path reflectance: median over the cases of the median |relative error| (%), [max over cases]')
    g = df.groupby(['scheme', 'wl'])
    tab = (100 * g.abs_rel_median.median()).unstack('wl').round(2).astype(str) + ' [' + \
          (100 * g.abs_rel_max.max()).unstack('wl').round(1).astype(str) + ']'
    print(tab.to_string())
    print('\nTOA path reflectance: median signed relative error (%) per wavelength, AOT <= 0.1 / > 0.1')
    for low in (True, False):
        sub = df[(df.aot <= 0.1) == low]
        print(('AOT <= 0.1' if low else 'AOT > 0.1'))
        print((100 * sub.groupby(['scheme', 'wl']).rel_median.median()).unstack('wl').round(2).to_string())
    print('\nSpectral AOT: median |relative error| (%)')
    print((100 * df.dropna(subset=['aot_rel']).assign(a=lambda d: d.aot_rel.abs())
           .groupby(['scheme', 'wl']).a.median()).unstack('wl').round(2).to_string())
    print(f'\n{df[["aot", "model"]].drop_duplicates().shape[0]} cases; written {ofile}')


if __name__ == '__main__':
    main(*sys.argv[1:2])
