'''
Sentinel-2 true-colour images of the RadCalNet water sites at 10 m (figures of the site documents): the site and
its surroundings, and a zoom on the platform with the region of interest (ROI) of the Sentinel-2 comparisons.

The L1C bands B04, B03, B02 and B08 (native 10 m) are read from the local copy of the L1C products
(``/data/satellite/s3/grstbx-s3/sentinel-2/l1c/<tile>/<yyyy>/<mm>/<dd>/*.SAFE``), window of the site only.
Contrast as in aqua3cs (``notebook/hypernets/s2_site_rgb.py``): TOA reflectance (radiometric offset of the
processing baselines >= 04.00 removed) minus a dark value per band (0.2 percentile over the water, first-order removal
of the path radiance), water (NDWI of B03 and B08 > 0) and land stretched separately with a white point common to the
three bands and a gamma.

The zoom shows the ROI of ``scripts/s2_toa_matchup/s2_toa_matchup.py --roi`` (square of half-size 200 m at VEIT,
300 m at CORO, minus the fixed structure + 100 m; structure pixels detected on the 20 m L1C bands,
``/data/radcalnet/studies/roi_selection/<code>_structures.csv``) and the 1.2 km box of the first revisions. GABU
(Galata) has no structure detection yet: its zoom shows the platform and the 1.2 km box only.

    python site_rgb.py --fig-dir <document>/media/v1.3 VEIT
    python site_rgb.py --preview <dir> CORO      # quick looks of the candidate dates
'''
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from matplotlib import patheffects
from rasterio.windows import from_bounds

L1C = Path('/data/satellite/s3/grstbx-s3/sentinel-2/l1c')
ROI_DIR = Path('/data/radcalnet/studies/roi_selection')

SITES = {
    # relative orbit 22 covers the whole subset (orbit 122: western edge of the swath)
    'VEIT': dict(lon=12.5083, lat=45.3139, tile='32TQR', orbit='R022', label='VEIT (AAOT tower)',
                 bbox=(12.18, 45.19, 12.64, 45.49), date='20240402', roi=(200., 100.), zoom=700., scalebar_km=5,
                 candidates=['20240402', '20260926', '20230410'],
                 places={'Venice': (12.335, 45.437), 'Chioggia': (12.279, 45.219), 'Lido': (12.37, 45.40)}),
    # only relative orbit 7 covers the platform; the eastern edge of the tile 35TPK is at 29.62-29.67 E
    'CORO': dict(lon=29.447, lat=44.546, tile='35TPK', orbit='R007', label='CORO (Section-7 platform)',
                 bbox=(28.90, 44.36, 29.61, 44.92), date='20260307', roi=(300., 100.), zoom=800., scalebar_km=5,
                 candidates=['20260307', '20251102', '20260312', '20250920'],
                 places={'Sfântu Gheorghe': (29.594, 44.898), 'Lake Razim': (28.98, 44.83)}),
    # relative orbit 7 covers the whole tile 35TNH (orbit 50: western part only); the eastern edge of the tile is at
    # about 28.36 E. No fixed-structure detection at Galata yet: zoom without ROI (roi=None)
    'GABU': dict(lon=28.19319, lat=43.044624, tile='35TNH', orbit='R007', label='GABU (Galata platform)',
                 bbox=(27.84, 42.93, 28.34, 43.26), date='20240411', roi=None, zoom=800., scalebar_km=5,
                 candidates=['20240411', '20230502'],
                 places={'Varna': (27.915, 43.208), 'Cape Galata': (27.945, 43.165), 'Kamchia mouth': (27.895, 43.03)}),
}


def find_safe(tile, date, orbit=None):
    d = L1C / tile / date[:4] / date[4:6] / date[6:]
    safes = sorted(p for p in d.glob('*.SAFE') if orbit is None or f'_{orbit}_' in p.name)
    if not safes:
        raise FileNotFoundError(f'no L1C product for {tile} {date}')
    return safes[-1]


def read_bands(safe, bbox):
    '''TOA reflectance of B04, B03, B02, B08 in the bounding box (lon/lat), transform and CRS.'''
    baseline = float(re.search(r'_N(\d{4})_', safe.name).group(1)) / 100.
    offset = -1000. if baseline >= 4. else 0.
    out = []
    for b in ('B04', 'B03', 'B02', 'B08'):
        path = next(safe.glob(f'GRANULE/*/IMG_DATA/*_{b}.jp2'))
        with rasterio.open(path) as src:
            tr = pyproj.Transformer.from_crs(4326, src.crs, always_xy=True)
            xs, ys = tr.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
            dn = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
            transform, crs = src.window_transform(win), src.crs
        rho = (dn + offset) / 10000.
        rho[dn == 0] = np.nan
        out.append(rho)
    return np.stack(out, -1), transform, crs


def stretch(bands, gamma_water=0.75, gamma_land=0.6, white_water=99.8, white_land=99.5):
    rgb, nir = bands[..., :3], bands[..., 3]
    water = (rgb[..., 1] - nir) / (rgb[..., 1] + nir) > 0.
    x = rgb - np.array([np.nanpercentile(rgb[..., k][water], 0.2) for k in range(3)])
    img = np.ones_like(x)
    for m, wp, g in ((water, white_water, gamma_water), (~water, white_land, gamma_land)):
        if m.any():
            w = np.nanpercentile(x[m], wp)
            img[m] = np.clip(x[m] / w, 0, 1) ** g
    return np.where(np.isfinite(img), img, 1.)


def _halo(w=2):
    return [patheffects.withStroke(linewidth=w, foreground='k', alpha=0.7)]


def plot_site(code, date, out, dpi=250):
    s = SITES[code]
    safe = find_safe(s['tile'], date, s.get('orbit'))
    bands, tr, crs = read_bands(safe, s['bbox'])
    img = stretch(bands)
    ny, nx = img.shape[:2]
    ext = (tr.c, tr.c + nx * tr.a, tr.f + ny * tr.e, tr.f)
    to = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    px, py = to.transform(s['lon'], s['lat'])

    fig = plt.figure(figsize=(12, 6.2))
    ax = fig.add_axes([0.0, 0.0, 0.6, 1.0])
    az = fig.add_axes([0.665, 0.08, 0.32, 0.8])
    ax.imshow(img, extent=ext, interpolation='none')
    ax.plot(px, py, marker='s', ms=10, mfc='none', mec='#ff3b30', mew=1.8)
    ax.annotate(s['label'], (px, py), xytext=(10, 8), textcoords='offset points', color='w', fontsize=10,
                weight='bold', path_effects=_halo())
    for name, (lo, la) in s['places'].items():
        qx, qy = to.transform(lo, la)
        if ext[0] < qx < ext[1] and ext[2] < qy < ext[3]:
            ax.annotate(name, (qx, qy), color='w', fontsize=8, ha='center', path_effects=_halo())
    L = s['scalebar_km'] * 1000.
    bx, by = ext[0] + 0.04 * (ext[1] - ext[0]), ext[2] + 0.05 * (ext[3] - ext[2])
    ax.plot([bx, bx + L], [by, by], c='w', lw=3, solid_capstyle='butt', path_effects=_halo(3))
    ax.annotate(f'{s["scalebar_km"]} km', (bx + L / 2, by), xytext=(0, 5), textcoords='offset points', ha='center',
                color='w', fontsize=9, path_effects=_halo())
    ax.annotate(f'{safe.name[:3]} MSI L1C, {date[:4]}-{date[4:6]}-{date[6:]}, B4-B3-B2 at 10 m', (0.99, 0.01),
                xycoords='axes fraction', ha='right', va='bottom', color='w', fontsize=8, path_effects=_halo())
    z = s['zoom']
    ax.add_patch(plt.Rectangle((px - z, py - z), 2 * z, 2 * z, fill=False, ec='yellow', lw=1.2))
    ax.set_axis_off()
    ax.set(xlim=ext[:2], ylim=ext[2:])

    # zoom on the platform (contrast of its own pixels: the open water is dark in the stretch of the whole scene);
    # distances to the AERONET-OC coordinates (m)
    i0, i1 = int((tr.f - (py + z)) / -tr.e), int(np.ceil((tr.f - (py - z)) / -tr.e))
    j0, j1 = int((px - z - tr.c) / tr.a), int(np.ceil((px + z - tr.c) / tr.a))
    # the image of the scene re-stretched with a low and a high value common to the three bands (hue kept; a
    # stretch per band turns the L1C quantisation of the dark water into colour noise)
    sub = img[i0:i1, j0:j1]
    lo, hi = np.nanpercentile(sub, [1, 99.5])
    zimg = np.clip((sub - lo) / (hi - lo), 0, 1)
    zext = (tr.c + j0 * tr.a - px, tr.c + j1 * tr.a - px, tr.f + i1 * tr.e - py, tr.f + i0 * tr.e - py)
    az.imshow(np.where(np.isfinite(zimg), zimg, 1.), extent=zext, interpolation='none')
    if s['roi'] is not None:
        half, buffer = s['roi']
        xy = pd.read_csv(ROI_DIR / f'{code}_structures.csv', comment='#')[['x_m', 'y_m']].values
        g = np.arange(-z, z + 1, 5.)
        X, Y = np.meshgrid(g, g)
        d = np.hypot(X[..., None] - xy[:, 0], Y[..., None] - xy[:, 1]).min(-1)
        roi = (np.maximum(np.abs(X), np.abs(Y)) <= half) & (d > buffer)
        az.contour(g, g, roi.astype(float), [0.5], colors='#ff2bd6', linewidths=2.0)
        az.contour(g, g, (d <= 10.).astype(float), [0.5], colors='#ff3b30', linewidths=1.2)
    az.add_patch(plt.Rectangle((-600, -600), 1200, 1200, fill=False, ec='w', ls='--', lw=1.2))
    az.plot(0, 0, '+', color='#ff3b30', ms=12, mew=1.8)
    az.set(xlim=(-z, z), ylim=(-z, z), xlabel='x (m, east of the AERONET-OC coordinates)', ylabel='y (m, north)')
    az.tick_params(labelsize=8)
    if s['roi'] is not None:
        az.set_title(f'{code}: ROI (magenta), square of half-size {half:g} m\nminus the structure (red) + {buffer:g} m; '
                     'dashed: 1.2 km box', fontsize=9)
    else:
        az.set_title(f'{code}: platform at the AERONET-OC coordinates (+)\ndashed: 1.2 km box', fontsize=9)
    fig.savefig(out, dpi=dpi, **({'pil_kwargs': {'quality': 92}} if str(out).endswith('.jpg') else {}))
    plt.close(fig)
    print(out, safe.name, img.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sites', nargs='+', choices=list(SITES))
    ap.add_argument('--fig-dir', help='folder of the figures (s2_<code>_rgb.jpg)')
    ap.add_argument('--preview', metavar='DIR', help='quick looks of the candidate dates in DIR')
    ap.add_argument('--date', help='date (yyyymmdd) instead of the default of the site')
    a = ap.parse_args()
    for code in a.sites:
        if a.preview:
            Path(a.preview).mkdir(parents=True, exist_ok=True)
            for d in SITES[code]['candidates']:
                try:
                    plot_site(code, d, Path(a.preview) / f'{code}_{d}.jpg', dpi=60)
                except Exception as e:
                    print(code, d, 'skip:', e)
        else:
            Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
            plot_site(code, a.date or SITES[code]['date'], Path(a.fig_dir) / f's2_{code.lower()}_rgb.jpg')


if __name__ == '__main__':
    main()
