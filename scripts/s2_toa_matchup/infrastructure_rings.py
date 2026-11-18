"""Footprint of the platform infrastructure (tower, shadow, moored ships) in the Sentinel-2 L1C pixels around an
AERONET-OC site, to set the exclusion radius and the region of interest (ROI) of the TOA comparisons.

For the scenes of the standard selection of matchup_analysis.py (valid fraction > 0.5, >= 50 pixels, nearest
measurement within 60 min, AOT(550) < 0.3, glint angle > 30 deg), the valid pixels (finite, B11 < 0.03) are binned
by distance to the site; per ring and scene: median TOA reflectance minus the median of the 600 m reference box
(1e-3, at 490, 665, 865 and 1610 nm), fraction of the pixels brighter than the box median + 4 robust standard
deviations at 865 nm (outliers: structures, ships, wakes), and fraction of the pixels with B11 >= 0.03.
Printed: median over the scenes, and 90th percentile of the outlier fractions. Also the change of the box
median with the exclusion radius (rings removed from the centre) and with the ROI half-size.

Usage: python infrastructure_rings.py <site_level> [odir] [csv]
    python infrastructure_rings.py AAOT_lev20
    python infrastructure_rings.py Section-7_Platform_lev15
"""
import glob
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matchup_analysis import load, selection, screen  # noqa: E402

ARGS = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=30., flags_exclude=None,
                       glint_stats='cm_iso', glint_pixels=False)
S2_ROOT = '/data/satellite/Sentinel-2/subset/aeronet/L1C'
SITES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'radcalnet_oc', 'data', 'aeronet',
                     'info_aeronet_oc_sentinel2_location.csv')
ALIASES = {'AAOT': 'Venise'}   # same tower: AERONET-OC site renamed in 2017
EDGES = [0, 30, 60, 100, 150, 200, 300, 450, 600, 900, 1200]
WLS = [490, 665, 865, 1610]
EXCLUSIONS = [0, 30, 60, 100, 150, 200]
HALF_SIZES = [150, 300, 600, 900, 1200]


def site_location(site):
    info = pd.read_csv(SITES).set_index('site')
    row = info.loc[ALIASES.get(site, site)]
    return float(row.longitude), float(row.latitude), row.s2_tile


def scene_stats(path, lon, lat):
    with xr.open_dataset(path) as s:
        epsg = int(re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]$', s.spatial_ref.attrs['crs_wkt']).group(1))
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        b = s.bands.sel(x=slice(x0 - 1200, x0 + 1200), y=slice(y0 + 1200, y0 - 1200), wl=WLS).load()
    X, Y = np.meshgrid(b.x.values - x0, b.y.values - y0)
    r = np.hypot(X, Y)
    v = {w: b.sel(wl=w).values for w in WLS}
    finite = b.notnull().all('wl').values
    ok = finite & (v[1610] < 0.03)
    box = ok & (np.abs(X) <= 600) & (np.abs(Y) <= 600)
    if box.sum() < 50:
        return [], []
    ref = {w: np.median(v[w][box]) for w in WLS}
    mad = max(1.4826 * np.median(np.abs(v[865][box] - ref[865])), 2e-4)
    rings = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = ok & (r >= lo) & (r < hi)
        if m.sum() == 0:
            continue
        row = {'r': f'{lo}-{hi}', 'n': int(m.sum())}
        for w in WLS:
            row[f'd{w}_1e3'] = 1e3 * (np.median(v[w][m]) - ref[w])
        row['outlier865_%'] = 100 * np.mean(v[865][m] > ref[865] + 4 * mad)
        ring = finite & (r >= lo) & (r < hi)
        row['swir_bright_%'] = 100 * np.mean(v[1610][ring] >= 0.03)
        rings.append(row)
    rois = []
    for h in HALF_SIZES:
        for e in EXCLUSIONS:
            m = ok & (np.abs(X) <= h) & (np.abs(Y) <= h) & (r >= e)
            if m.sum() < 10:
                continue
            row = {'half_size': h, 'exclusion': e, 'n': int(m.sum())}
            for w in WLS[:3]:
                row[f'd{w}_1e3'] = 1e3 * (np.median(v[w][m]) - ref[w])
                row[f'iqr{w}_1e3'] = 1e3 * np.subtract(*np.percentile(v[w][m], [75, 25]))
            rois.append(row)
    return rings, rois


def main(site_level, odir='/data/radcalnet/matchup/sentinel2_dev', ofile=None):
    site = re.sub(r'_lev(15|20)$', '', site_level)
    lon, lat, tile = site_location(site)
    ds = load(odir, site_level, ARGS)
    keep = (selection(ds, ARGS) & screen(ds, ARGS)).values
    prods = sorted(set(ds['product'].values[keep].astype(str)))
    paths = {os.path.basename(p)[:-3]: p for p in glob.glob(f'{S2_ROOT}/{tile}/*/*.nc')}
    rings, rois = [], []
    for prod in prods:
        if prod not in paths:
            continue
        a, b = scene_stats(paths[prod], lon, lat)
        rings += [{'product': prod, **x} for x in a]
        rois += [{'product': prod, **x} for x in b]
    rings, rois = pd.DataFrame(rings), pd.DataFrame(rois)
    pd.set_option('display.width', 200)
    print(f'{site_level} ({lon:.4f}, {lat:.4f}, tile {tile}): {rings["product"].nunique()} scenes')
    print('\nMedian over the scenes, per ring of distance to the site (m): difference with the 600 m box median')
    print(rings.groupby('r', sort=False).median(numeric_only=True).round(3).to_string())
    print('\n90th percentile over the scenes of the outlier fractions (%)')
    print(rings.groupby('r', sort=False)[['outlier865_%', 'swir_bright_%']].quantile(0.9).round(2).to_string())
    print('\nROI median minus the 600 m box median, median over the scenes (1e-3), and median IQR of the pixels')
    print(rois.groupby(['half_size', 'exclusion']).median(numeric_only=True).round(3).to_string())
    if ofile:
        rings.to_csv(ofile, index=False)
    return rings, rois


if __name__ == '__main__':
    main(*sys.argv[1:])
