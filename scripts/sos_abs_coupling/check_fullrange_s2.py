'''
End-to-end check of the full-range coupling table (gas_coupling_ppld_full.csv) on the Sentinel-2 matchups.

The package couples the absorption with the scattering in the windows of 684-990 nm only; beyond, the path
reflectance gets the gaseous transmittance of the direct beam over the whole Sun-surface-sensor path, as if all the
scattering occurred below the absorbers. In the saturated water-vapour band of MSI B10 (1375 nm) the simulation is then
~0, while the clear-sky L1C B10 has a floor of ~0.5e-3, the single scattering of the molecules above the water vapour
(``scripts/s2_toa_matchup/residual_budget.py``). The SOS-ABS study recommended the full-range photon path-length
models beyond 990 nm after an end-to-end check, first in the H2O 1380 and 1880 nm windows
(``conclusions_fullrange.md``). This script makes that check with the observations.

Candidate table: the package table (684-990 nm) plus the windows of the full-range fit recommended by the study,
H2O 1000, 1130, 1380, 1880, 2150, 2450 nm, O2 1270 and CO2 2010 nm (the O3 and CO2 1600 nm windows keep the decoupled
model; the CH4 windows are left out, CH4 not being a coupled gas of the package).

For the matchups of the standard selection (VEIT L2.0, heavy-tailed glint file), the nearest AERONET-OC measurement
is simulated with the package table and with the candidate (geometry, spectral responses and band integration of
``s2_toa_matchup.py``), and the band TOA reflectances are compared with the L1C: B10 against its clear-sky floor
(scenes without cirrus: B10 excess over the Rayleigh above the water vapour below 1e-4) and the change of the other
bands.

Usage: python check_fullrange_s2.py [--n N] [--n-jobs 4] [--out csv]
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 's2_toa_matchup'))

import aeronet_oc_db  # noqa: E402
import radcalnet_oc  # noqa: E402
from radcalnet_oc.kernel import GasCoupling, GAS_COUPLING_PPLD_FILE  # noqa: E402

import s2_toa_matchup as m  # noqa: E402
from residual_budget import load, rayleigh_above  # noqa: E402

FULL_TABLE = os.path.join(HERE, 'gas_coupling_ppld_full.csv')
NEW_WINDOWS = ['H2O-1000', 'H2O-1130', 'O2-1270', 'H2O-1380', 'H2O-1880', 'CO2-2010', 'H2O-2150', 'H2O-2450']
SITE, LEVEL, CODE = 'AAOT', 'lev20', 'AAOT_lev20'
BANDS = [865, 945, 1375, 1610, 2190]


def candidate_table():
    '''Package table (684-990 nm) and the recommended full-range windows beyond 990 nm.'''
    base = pd.read_csv(GAS_COUPLING_PPLD_FILE, comment='#')
    full = pd.read_csv(FULL_TABLE, comment='#')
    return pd.concat([base, full[full.window.isin(NEW_WINDOWS)]], ignore_index=True)


class _Process(radcalnet_oc.Process):
    '''Process whose path-length coupling table is replaced by ``TABLE`` (None: package table).'''
    TABLE = None

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.TABLE is not None:
            self.coupling = GasCoupling(self.TABLE)


def run_scene(product, path, lon, lat, records, table):
    sat, srf = m.s2_extract(path, lon, lat)
    _Process.TABLE = table
    m.radoc.Process = _Process
    try:
        rec = m.simulate(records, sat, srf)
    finally:
        m.radoc.Process = radcalnet_oc.Process
    return None if rec is None else rec.Rtoa.isel(time=0).sel(wl=BANDS).values


def task(product, path, lon, lat, records):
    out = {'product': product}
    for name, table in (('ref', None), ('new', candidate_table())):
        r = run_scene(product, path, lon, lat, records, table)
        if r is None:   # Rrs reconstruction failed for the measurement
            return None
        out.update({f'{name}_{b}': v for b, v in zip(BANDS, r)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n', type=int, default=60, help='number of matchups (half without cirrus, half at random)')
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--odir', default='/data/radcalnet/matchups/sentinel2/v1.2.0')
    p.add_argument('--out', default=None, help='CSV of the band reflectances per scene')
    args = p.parse_args()

    _, ds = load(args.odir, CODE)
    r_above, _ = rayleigh_above(ds)
    excess = ds.sat_median.sel(wl=1375).values - r_above
    df = pd.DataFrame({'time': ds.time.values, 'product': ds['product'].astype(str).values, 'excess': excess,
                       'R_above': r_above, 'sza': ds.sza.values, 'tcwv': ds.tcwv.values,
                       **{f'sat_{b}': ds.sat_median.sel(wl=b).values for b in BANDS}})
    clear = df[df.excess < 1e-4]
    rng = np.random.default_rng(0)
    pick = pd.concat([clear.sample(min(len(clear), args.n // 2), random_state=0),
                      df.drop(clear.index).sample(args.n - min(len(clear), args.n // 2), random_state=1)])
    print(f'{len(pick)} matchups ({min(len(clear), args.n // 2)} without cirrus of {len(clear)})')

    info = aeronet_oc_db.site_info(level=LEVEL).loc[SITE]
    lon, lat = float(info.longitude), float(info.latitude)
    insitu = m.aeronet_inputs(SITE, LEVEL).load()
    inv = m.s2_inventory(info.s2_tile)
    tasks = []
    for _, row in pick.iterrows():
        path = inv[inv['product'] == row['product']].path
        rec = insitu.sel(time=slice(row.time - pd.Timedelta(hours=2), row.time + pd.Timedelta(hours=2)))
        rec = rec.where(rec.aot550.notnull() & (rec.Rrs_Lwn.count('wl') >= 5), drop=True)
        if len(path) == 0 or rec.sizes['time'] == 0:
            continue
        nearest = int(np.argmin(np.abs(rec.time.values - np.datetime64(row.time))))
        tasks.append((row['product'], path.iloc[0], lon, lat, rec.isel(time=[nearest])))

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=args.n_jobs)(delayed(task)(*t) for t in tasks)
    res = [r for r in res if r is not None]
    out = pick.merge(pd.DataFrame(res), on='product')
    if args.out:
        out.to_csv(args.out, index=False)

    k = 1e3
    print('\n--- band TOA reflectance, medians (x1e-3): L1C, package table, candidate table')
    for b in BANDS:
        for name, sub in (('without cirrus', out[out.excess < 1e-4]), ('all', out)):
            print(f'{b:5d} {name:15s} N={len(sub):3d}  sat {k * sub[f"sat_{b}"].median():7.3f}  '
                  f'ref {k * sub[f"ref_{b}"].median():7.3f}  new {k * sub[f"new_{b}"].median():7.3f}  '
                  f'sat-ref {k * (sub[f"sat_{b}"] - sub[f"ref_{b}"]).median():+7.3f}  '
                  f'sat-new {k * (sub[f"sat_{b}"] - sub[f"new_{b}"]).median():+7.3f}')
    c = out[out.excess < 1e-4]
    print('\n--- B10 without cirrus (x1e-3): L1C, single-scattering Rayleigh above the water vapour, candidate')
    print(pd.DataFrame({'sat': k * c.sat_1375, 'R_above': k * c.R_above, 'new': k * c.new_1375,
                        'sza': c.sza, 'tcwv': c.tcwv}).describe(percentiles=[.1, .5, .9]).round(3).to_string())
    print('corr(new B10, sat B10) without cirrus:', round(float(np.corrcoef(c.new_1375, c.sat_1375)[0, 1]), 2))


if __name__ == '__main__':
    main()
