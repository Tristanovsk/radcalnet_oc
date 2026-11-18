"""Change of the Sentinel-2 matchup residuals between two runs of the simulation (e.g. before and after a change of
the radiative transfer model).

For each site of both folders, the scenes of the standard selection of matchup_analysis.py (valid fraction > 0.5,
>= 50 pixels, nearest measurement within 60 min, AOT(550) < 0.3, glint angle > 30 deg; simulation without glint)
present in both runs are compared: median residual sim - sat (1e-3 and %) of the old and new runs and median
relative change of the simulation, for all the scenes and per AOT class, at the bands given (443 and 865 nm by
default). The change expected from the Rayleigh pressure scaling (version 1.2: about +0.06 % of the TOA
reflectance per hPa above 1015.2 hPa in the blue, CHANGELOG) is given at 443 nm from the pressure of each scene, to
separate it from the other changes.

Usage: python residual_change.py <old odir> <new odir> [bands]
    python residual_change.py /data/radcalnet/matchup/sentinel2 /data/radcalnet/matchup/sentinel2_dev 443,865
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matchup_analysis import load, selection, screen  # noqa: E402

ARGS = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=30., flags_exclude=None,
                       glint_stats='cm_iso', glint_pixels=False)
AOT_CLASSES = [(0, 0.08), (0.08, 0.15), (0.15, 0.3)]
P_REF, DP_443 = 1015.2, 0.06e-2   # LUT pressure (hPa) and relative TOA change per hPa at 443 nm


def sites(odir):
    return {f.replace('_s2_toa_matchup.nc', '') for f in os.listdir(odir) if f.endswith('_s2_toa_matchup.nc')}


def main(old_dir, new_dir, bands=(443, 865)):
    rows = []
    for site in sorted(sites(old_dir) & sites(new_dir)):
        old, new = load(old_dir, site, ARGS), load(new_dir, site, ARGS)
        keep = (selection(new, ARGS) & screen(new, ARGS)).to_pandas()
        prods = set(new['product'].values[keep.values].astype(str)) & set(old['product'].values.astype(str))
        if not prods:
            continue
        o = old.isel(time=np.isin(old['product'].values.astype(str), list(prods))).sortby('product')
        n = new.isel(time=np.isin(new['product'].values.astype(str), list(prods))).sortby('product')
        assert (o['product'].values == n['product'].values).all()
        p = n.pressure.values / (100. if np.nanmedian(n.pressure) > 2000 else 1.)   # Pa or hPa
        for lo, hi in [(0, 0.3)] + AOT_CLASSES:
            m = (n.aot550.values >= lo) & (n.aot550.values < hi)
            if m.sum() < 3:
                continue
            for b in bands:
                obs, so, sn = (x.sel(wl=b).values[m] for x in (n.sat_median, o.sim_median, n.sim_median))
                row = {'site': site, 'aot': f'{lo:g}-{hi:g}', 'band': b, 'N': int(m.sum()),
                       'old_1e3': 1e3 * np.nanmedian(so - obs), 'new_1e3': 1e3 * np.nanmedian(sn - obs),
                       'old_%': 100 * np.nanmedian(so / obs - 1), 'new_%': 100 * np.nanmedian(sn / obs - 1),
                       'sim_change_%': 100 * np.nanmedian(sn / so - 1),
                       'sim_change_1e3': 1e3 * np.nanmedian(sn - so)}
                if b == 443:
                    row['pressure_part_%'] = 100 * np.nanmedian(DP_443 * (p[m] - P_REF))
                rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_rows', 200)
    print(df.round(3).to_string(index=False))
    return df


if __name__ == '__main__':
    bands = tuple(int(b) for b in sys.argv[3].split(',')) if len(sys.argv) > 3 else (443, 865)
    main(sys.argv[1], sys.argv[2], bands)
