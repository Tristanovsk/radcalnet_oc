"""Region of interest (ROI) of the Sentinel-2 TOA comparisons at the RadCalNet water sites: size, shape and
exclusion of the platform (and of the other fixed structures), from the Sentinel-2 L1C subsets of the site tile.
Functions used by the notebook ``roi_selection.ipynb`` (same folder).

Two steps:

1. ``extract`` (command line, a few minutes per site): for every L1C subset of the tile, the window of
   +/- ``HALF`` m around the site (20 m pixels, bands ``BANDS``), with the mean geometry of the window; scenes with
   less than ``FRAC_KEEP`` of clear pixels (finite, cirrus 1375 nm < 0.01, SWIR 1610 nm < 0.03) are skipped.
   Written to ``/data/radcalnet/studies/roi_selection/<code>_s2_l1c_window.nc`` (scene, wl, y, x).
2. analysis in the notebook: fixed structures (frequency of the SWIR-bright pixels over the clear scenes),
   profiles around the site and around the structures, ROI candidates (square or disc, half-size, exclusion of a
   disc or of the dilated structure mask) scored by their contamination (ROI median minus the far-field median),
   their spatial heterogeneity, the scenes kept, and the dispersion of the matchup residual against the
   RadCalNet-OC simulation of the matchup files (``matchup_analysis.py``).

Usage: python roi_selection.py extract VEIT
       python roi_selection.py extract CORO
"""
import glob
import os
import re
import warnings
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
S2_ROOT = '/data/satellite/Sentinel-2/subset/aeronet/L1C'
ODIR = '/data/radcalnet/studies/roi_selection'
MATCHUPS = '/data/radcalnet/matchups/sentinel2/v1.2.0'
LOCATIONS = os.path.join(HERE, '..', '..', 'radcalnet_oc', 'data', 'aeronet', 'info_aeronet_oc_sentinel2_location.csv')
#: RadCalNet code -> AERONET-OC location name (info csv) and matchup files (<site>_<level>, glint statistics)
SITES = {'VEIT': {'location': 'Venise', 'name': 'Venise (AAOT, Acqua Alta tower)',
                  'matchups': [('AAOT_lev20', 'cm_iso_mix')]},
         'CORO': {'location': 'Section-7_Platform', 'name': 'Section-7 Platform (Black Sea)',
                  'matchups': [('Section-7_Platform_lev20', 'cm_iso'), ('Section-7_Platform_lev15', 'cm_iso')]}}
BANDS = [490, 560, 665, 865, 1375, 1610, 2190]
HALF = 1500.        # half size of the extracted window (m)
FRAC_KEEP = 0.3     # clear fraction of the window below which a scene is not kept
SWIR_MAX, CIRRUS_MAX = 0.03, 0.01


def site_location(code):
    info = pd.read_csv(LOCATIONS).set_index('site').loc[SITES[code]['location']]
    return float(info.longitude), float(info.latitude), info.s2_tile


def window(path, lon, lat, n):
    """Window of 2n+1 pixels centred on the pixel of (lon, lat), bands BANDS (float32), and its mean geometry."""
    with xr.open_dataset(path) as s:
        epsg = int(re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]$', s.spatial_ref.attrs['crs_wkt']).group(1))
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        ix, iy = int(np.abs(s.x.values - x0).argmin()), int(np.abs(s.y.values - y0).argmin())
        if ix < n or iy < n or ix + n >= s.sizes['x'] or iy + n >= s.sizes['y']:
            return None
        sl = dict(x=slice(ix - n, ix + n + 1), y=slice(iy - n, iy + n + 1))
        b = s.bands.isel(**sl).sel(wl=BANDS).astype('float32').load()
        geo = {'sza': float(s.sza.isel(**sl).mean()), 'vza': float(s.vza.isel(**sl).sel(wl=865).mean()),
               'raa': float(np.degrees(np.angle(np.exp(1j * np.radians(s.raa.isel(**sl).sel(wl=865))).mean())) % 360),
               'time': s.time.values}
        dx, dy = s.x.values[sl['x']] - x0, s.y.values[sl['y']] - y0
    return b.values, dx, dy, geo


def glint_angle(sza, vza, raa):
    """Angle between the viewing direction and the specular direction (deg), raa = viewing - solar azimuth."""
    azi = (180. - raa) % 360
    azi = np.where(azi <= 180, azi, 360 - azi)
    s0, sv, a = np.radians(sza), np.radians(vza), np.radians(azi)
    return np.degrees(np.arccos(np.cos(s0) * np.cos(sv) + np.sin(s0) * np.sin(sv) * np.cos(a)))


def extract(code):
    lon, lat, tile = site_location(code)
    n = int(round(HALF / 20.))
    paths = sorted(glob.glob(f'{S2_ROOT}/{tile}/*/*.nc'))
    cubes, rows, grid = [], [], None
    for i, p in enumerate(paths):
        w = window(p, lon, lat, n)
        if w is None:
            continue
        b, dx, dy, geo = w
        clear = np.isfinite(b).all(0) & (b[BANDS.index(1375)] < CIRRUS_MAX) & (b[BANDS.index(1610)] < SWIR_MAX)
        if clear.mean() < FRAC_KEEP:
            continue
        grid = grid or (dx, dy)
        cubes.append(b)
        rows.append({'product': os.path.basename(p)[:-3], **geo, 'clear_frac': float(clear.mean())})
        if i % 100 == 0:
            print(f'{code}: {i}/{len(paths)} scenes read, {len(cubes)} kept', flush=True)
    meta = pd.DataFrame(rows)
    ds = xr.Dataset({'bands': (('scene', 'wl', 'y', 'x'), np.stack(cubes))},
                    coords={'wl': BANDS, 'y': grid[1], 'x': grid[0], 'scene': np.arange(len(meta))})
    for k in ['product', 'sza', 'vza', 'raa', 'clear_frac']:
        ds[k] = ('scene', meta[k].values)
    ds['time'] = ('scene', pd.to_datetime(meta['time']).values)
    ds['glint_angle'] = ('scene', glint_angle(meta.sza.values, meta.vza.values, meta.raa.values))
    ds.x.attrs = ds.y.attrs = {'units': 'm', 'description': 'distance to the site (pixel centres), UTM'}
    ds.attrs = {'site': code, 'site_name': SITES[code]['name'], 'longitude': lon, 'latitude': lat, 'tile': tile,
                'source': S2_ROOT, 'description': 'Sentinel-2 L1C TOA reflectance around the site, clear scenes '
                f'(clear fraction of the window >= {FRAC_KEEP}; clear: finite, 1375 nm < {CIRRUS_MAX}, '
                f'1610 nm < {SWIR_MAX})', 'script': 'scripts/s2_toa_matchup/roi_selection.py'}
    os.makedirs(ODIR, exist_ok=True)
    out = os.path.join(ODIR, f'{code}_s2_l1c_window.nc')
    ds.to_netcdf(out, encoding={'bands': {'zlib': True, 'complevel': 4}})
    print(f'{code}: {len(meta)} scenes of {len(paths)} -> {out}')


# ---------------------------------------------------------------------------------------------------------------
# analysis (notebook)
# ---------------------------------------------------------------------------------------------------------------
def load_window(code):
    ds = xr.open_dataset(os.path.join(ODIR, f'{code}_s2_l1c_window.nc')).load()
    b = ds.bands
    ds['clear'] = (b.notnull().all('wl') & (b.sel(wl=1375) < CIRRUS_MAX) & (b.sel(wl=1610) < SWIR_MAX))
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    ds['r'] = (('y', 'x'), np.hypot(X, Y))
    return ds


def far_field(ds, structures=None, r_min=600., r_max=1500.):
    """Mask (y, x) of the far field: annulus r_min-r_max around the site, without the structures."""
    m = (ds.r >= r_min) & (ds.r <= r_max)
    return m if structures is None else m & ~structures


def structure_frequency(ds, delta=0.005, fmin=0.5, swir_clean=0.01):
    """
    Frequency, over the clear scenes, of the SWIR-bright pixels: 1610 nm above the far-field median by more than
    `delta` (the water is black at 1610 nm: solid structures, also when smaller than a 20 m pixel as the AAOT
    tower, ships, wakes), in the scenes whose far field is clean (median 1610 nm < swir_clean) and mostly clear
    (clear fraction of the far field >= fmin; clouds and strong glint excluded).
    """
    ff = far_field(ds)
    swir = ds.bands.sel(wl=1610)
    ok_scene = (swir.where(ff).median(['y', 'x']) < swir_clean) & (ds.clear.where(ff).mean(['y', 'x']) >= fmin)
    s = swir.sel(scene=ok_scene)
    finite = s.notnull()
    bright = (s - s.where(ff).median(['y', 'x'])) > delta
    freq = (bright & finite).sum('scene') / finite.sum('scene')
    freq.attrs.update(n_scenes=int(ok_scene.sum()), delta=delta)
    return freq


def structures_from_frequency(freq, threshold=0.5):
    """Fixed structures: pixels SWIR-bright in more than `threshold` of the scenes (transient objects as ships
    are below); connected components (8-connectivity)."""
    mask = (freq > threshold).values
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    rows = []
    X, Y = np.meshgrid(freq.x.values, freq.y.values)
    for k in range(1, n + 1):
        m = lab == k
        rows.append({'id': k, 'n_pix': int(m.sum()), 'area_m2': 400 * int(m.sum()),
                     'x_m': float(X[m].mean()), 'y_m': float(Y[m].mean()),
                     'dist_site_m': float(np.hypot(X[m].mean(), Y[m].mean())),
                     'freq_max': float(freq.values[m].max())})
    return xr.DataArray(mask, coords=freq.coords, dims=freq.dims), pd.DataFrame(rows)


def distance_to(mask, dx=20.):
    """Distance (m) of each pixel to the nearest pixel of `mask` (DataArray y, x)."""
    if not mask.values.any():
        return xr.full_like(mask, np.inf, dtype=float)
    return xr.DataArray(ndimage.distance_transform_edt(~mask.values) * dx, coords=mask.coords, dims=mask.dims)


def anomaly(ds, ref_mask, wl=865):
    """Per scene: band minus the median of the reference mask (clear pixels), NaN where not clear."""
    b = ds.bands.sel(wl=wl).where(ds.clear)
    return b - b.where(ref_mask).median(['y', 'x'])


def profile(an, dist, edges):
    """Median over the scenes of the per-scene median anomaly in bins of `dist` (m); also the 90th percentile."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if not m.any():
            continue
        per_scene = an.where(m).median(['y', 'x'])
        rows.append({'bin': f'{lo:g}-{hi:g}', 'mid': 0.5 * (lo + hi), 'n_pix': int(m.sum()),
                     'median_1e3': 1e3 * float(per_scene.median()), 'p90_1e3': 1e3 * float(per_scene.quantile(0.9))})
    return pd.DataFrame(rows)


def roi_mask(ds, shape, half, exclusion=0., structures=None, buffer=None):
    """ROI (y, x): square or disc of half-size `half` (m), minus a disc of radius `exclusion` around the site
    and, with `buffer`, minus the structures dilated by `buffer` m."""
    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    m = (np.maximum(np.abs(X), np.abs(Y)) <= half) if shape == 'square' else (np.hypot(X, Y) <= half)
    m &= np.hypot(X, Y) >= exclusion
    if buffer is not None and structures is not None:
        m &= (distance_to(structures).values > buffer)
    return xr.DataArray(m, coords={'y': ds.y, 'x': ds.x}, dims=('y', 'x'))


def roi_stats(ds, roi, wls=(490, 665, 865)):
    """Per scene: clear fraction of the ROI, number of clear pixels, median and relative IQR per band."""
    idx = np.asarray(roi.values, dtype=bool)
    if not idx.any():
        raise ValueError('empty ROI')
    flat =ds.bands.sel(wl=list(wls)).where(ds.clear).values[:, :, idx]        # (scene, wl, pixels of the ROI)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                        # scenes without clear pixels
        q25, med, q75 = np.nanpercentile(flat, [25, 50, 75], axis=-1)
    n = xr.DataArray(ds.clear.values[:, idx].sum(-1), dims='scene')
    return xr.Dataset({'median': (('scene', 'wl'), med), 'riqr': (('scene', 'wl'), (q75 - q25) / med),
                       'n_clear': n, 'frac_clear': n / int(roi.sum())},
                      coords={'scene': ds.scene, 'wl': list(wls)})


def matchups(code, glint_min=30.):
    """Matchups of the site kept by the standard selection (matchup_analysis.py): product, simulated TOA with the
    direct glint (simg_median), per band; the scenes of several levels are concatenated (one row per product)."""
    sys.path.insert(0, HERE)
    from types import SimpleNamespace
    from matchup_analysis import load, selection, screen
    out = []
    for site, gstats in SITES[code]['matchups']:
        a = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=glint_min,
                            flags_exclude=None, glint_stats=gstats, glint_pixels=False)
        m = load(MATCHUPS, site, a)
        keep = (selection(m, a) & screen(m, a)).values
        d = m.simg_median.isel(time=keep).to_pandas()
        d.index = m['product'].values[keep].astype(str)
        d['level'] = site.rsplit('_', 1)[1]
        d['glint_angle'] = m.glint_angle.values[keep]
        out.append(d)
    out = pd.concat(out)
    return out[~out.index.duplicated()]


def score(ds, candidates, structures, sim, wls=(490, 665, 865), frac_min=0.5, ff_buffer=100.):
    """
    Metrics of the ROI candidates (list of dicts: shape, half, exclusion, buffer) over the clear scenes of the
    window with a glint angle > 30 deg:

    - ``contam<wl>_1e3``: median over the scenes of the ROI median minus the far-field median (600-1500 m, outside
      the structures dilated by ff_buffer m), 1e-3 of reflectance: contamination by the structures, and spatial
      representativeness;
    - ``riqr<wl>_%``: median relative IQR of the clear pixels of the ROI (heterogeneity: waves, glint, ships);
    - ``kept_%``: scenes with a clear fraction of the ROI >= frac_min (and >= 9 pixels);
    - over the matchup scenes (``sim``): ``bias<wl>_%`` median of (ROI - sim) / sim and ``rsd<wl>_%`` its robust
      standard deviation (1.4826 MAD), ``n_mu`` the matchups kept.
    """
    ff = far_field(ds, distance_to(structures) <= ff_buffer if structures is not None else None)
    glint_ok = ds.glint_angle > 30
    ffstat = roi_stats(ds, ff, wls)
    prods = pd.Index(ds['product'].values.astype(str))
    in_mu = prods.isin(sim.index)
    sim_ds = sim.reindex(prods[in_mu])[list(wls)].values
    rows = []
    warnings.simplefilter('ignore', RuntimeWarning)
    for c in candidates:
        roi = roi_mask(ds, c['shape'], c['half'], c.get('exclusion', 0.), structures, c.get('buffer'))
        if roi.sum() < 9:     # e.g. a disc smaller than the buffer around the structure
            continue
        st = roi_stats(ds, roi, wls)
        ok = (st.frac_clear >= frac_min) & (st.n_clear >= 9)
        sel = (ok & glint_ok).values
        row = {**c, 'n_pix': int(roi.sum()), 'kept_%': 100 * float(ok.sel(scene=glint_ok).mean())}
        for j, w in enumerate(wls):
            row[f'contam{w}_1e3'] = 1e3 * float(np.nanmedian(st['median'].values[sel, j] - ffstat['median'].values[sel, j]))
            row[f'riqr{w}_%'] = 100 * float(np.nanmedian(st['riqr'].values[sel, j]))
        mu = sel[in_mu]
        rel = 100 * (st['median'].values[in_mu][mu] - sim_ds[mu]) / sim_ds[mu]
        row['n_mu'] = int(np.isfinite(rel).all(1).sum())
        for j, w in enumerate(wls):
            r = rel[:, j][np.isfinite(rel[:, j])]
            row[f'bias{w}_%'] = float(np.median(r)) if r.size else np.nan
            row[f'rsd{w}_%'] = float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def residuals(ds, c, structures, sim, wls=(490, 665, 865), frac_min=0.5):
    """Matchup residual (ROI - sim) / sim (%) of the candidate `c`, per matchup scene (DataFrame product x wl)."""
    roi = roi_mask(ds, c['shape'], c['half'], c.get('exclusion', 0.), structures, c.get('buffer'))
    st = roi_stats(ds, roi, wls)
    ok = ((st.frac_clear >= frac_min) & (st.n_clear >= 9) & (ds.glint_angle > 30)).values
    prods = pd.Index(ds['product'].values.astype(str))
    keep = ok & prods.isin(sim.index)
    rel = 100 * (st['median'].values[keep] - sim.reindex(prods[keep])[list(wls)].values) / \
        sim.reindex(prods[keep])[list(wls)].values
    return pd.DataFrame(rel, index=prods[keep], columns=list(wls)).dropna()


def robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def bootstrap_rsd(res_a, res_b, wl, n=2000, seed=0):
    """Paired bootstrap over the common matchups of the difference of robust SD (a - b) at `wl`: median and
    5-95 % interval; an interval containing 0 means that the two ROIs cannot be told apart by the matchups."""
    common = res_a.index.intersection(res_b.index)
    a, b = res_a.loc[common, wl].values, res_b.loc[common, wl].values
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(common), len(common))
        d.append(robust_sd(a[i]) - robust_sd(b[i]))
    return np.percentile(d, [5, 50, 95])


def recommend(S, tol865=0.1, tol665=0.05, kept_margin=3., wls=(665, 865)):
    """
    Candidates without contamination (|ROI - far field| <= tol865 at 865 nm and <= tol665 at 665 nm, 1e-3; the
    L1C quantisation is 1e-4) and keeping nearly all the scenes (kept_% >= max - kept_margin), ranked by the
    dispersion of the matchup residual (sum of the ranks of rsd at `wls`), then by the heterogeneity.
    """
    ok = (S['contam865_1e3'].abs() <= tol865) & (S['contam665_1e3'].abs() <= tol665) & \
        (S['kept_%'] >= S['kept_%'].max() - kept_margin)
    c = S[ok].copy()
    c['rank'] = sum(c[f'rsd{w}_%'].rank() for w in wls)
    return c.sort_values(['rank', 'riqr865_%'])


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != 'extract' or sys.argv[2] not in SITES:
        sys.exit(__doc__)
    extract(sys.argv[2])
