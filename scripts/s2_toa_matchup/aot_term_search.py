'''
Search for the term correlated with the AOT in the NIR-SWIR deficit of the Sentinel-2 TOA matchups.

The deficit (sim - sat) increases with AOT(550) with a spectrally flat slope (about -12, -11, -9 x1e-3 per unit AOT
at 865, 1610 and 2190 nm), which the aerosol models of the LUT cannot produce for the measured AOT spectrum
(``aerosol_diagnosis.py``). Candidate non-aerosol terms, each tested on its signature:

1. thin cirrus (or haze above the boundary layer): spectrally flat, it raises the AERONET-OC AOT when not
   screened; tested with the reflectance observed in the cirrus band B10 (1375 nm);
2. AERONET-OC AOT not representative of the satellite box (time and space): an error proportional to the AOT
   follows the spectral shape of dR_atm/dAOT of the aerosol model, steep for a fine mode; tested by comparing that
   shape with the observed slopes (LUT at the median geometry) and the slope against the time lag and the
   variability of the AOT in the window;
3. coupling of the aerosols with the glint (forward scattering of the glint): tested with the slope per glint-angle
   bin;
4. a term driven by the wind (sea-salt AOT, foam, roughness): tested with the AOT-wind correlation and the slope at
   fixed wind;
5. the seasonal confound with the glint angle (high AOT in summer, when the glint angle is small): tested with the
   joint regression on AOT, glint angle, SZA and B10, and with the simulated direct glint (isotropic Cox-Munk at the
   AERONET-OC wind) per glint-angle bin.

Example::

    python aot_term_search.py
'''
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import xarray as xr

ODIR = '/data/radcalnet/matchup/sentinel2'
CACHE = '/data/radcalnet/matchup/sentinel2/aerosol_diagnosis'
SITES = ['AAOT_lev20', 'Galata_Platform_lev20', 'AAOT_lev15']
BANDS = [865, 1610, 2190]
FLAGS = {'glint_low': 1, 'glint_moderate': 2, 'glint_high': 4, 'glint_near_specular': 8}


def load(odir, site, dt_max=60, aot_max=0.3, glint_min=30, glint_stats='cm_iso', with_glint=False,
         flags_exclude=None):
    '''Selected matchups: deficit (x1e-3), AOT, B10 reflectance, glint angle, wind, AOT variability in the window.'''
    sfx = '' if glint_stats == 'cm_iso' else f'_{glint_stats}'
    ds = xr.open_dataset(f'{odir}/{site}_s2_toa_matchup{sfx}.nc')
    if flags_exclude:  # glint screening by the sunglint flags instead of the glint angle
        glint_min = -1.
        ok = (ds.glint_flags & np.uint8(sum(FLAGS[f] for f in flags_exclude))) == 0
    else:
        ok = True
    sel = (ok & (ds.frac_valid > 0.5) & (ds.n_pix >= 50) & (np.abs(ds.dt_nearest) <= dt_max)
           & (ds.aot550 < aot_max) & (ds.glint_angle > glint_min))
    d = ds.sel(time=sel)
    sim = d.simg_median if with_glint else d.sim_median
    df = pd.DataFrame({f'D{b}': 1e3 * (sim - d.sat_median).sel(wl=b).values for b in BANDS},
                      index=pd.DatetimeIndex(d.time.values))
    for b in BANDS:
        df[f'G{b}'] = 1e3 * d.glint_median.sel(wl=b).values  # simulated direct glint (not in sim_median)
    df['B10'] = 1e3 * d.sat_median.sel(wl=1375).values
    for v in ['aot550', 'ang_exp', 'wind', 'sza', 'vza', 'glint_angle', 'dt_nearest']:
        df[v] = d[v].values
    df['month'] = df.index.month
    df['abs_dt'] = np.abs(df.dt_nearest)
    # variability of the AOT of the AERONET-OC measurements within +/- 60 min of the overpass
    rec = pd.DataFrame({'scene': ds.rec_scene.values, 'aot': ds.rec_aot550.values, 'dt': ds.rec_dt.values})
    rec = rec[np.abs(rec.dt) <= 60]
    g = rec.groupby('scene').aot
    df['aot_cv'] = (g.std() / g.mean()).reindex(df.index).values
    df['n_rec'] = g.size().reindex(df.index).values
    df['site'] = site
    return df.dropna(subset=['D865', 'aot550'])


def fit(df, cols, y):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    Y = df[y].values
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    se = np.sqrt(np.diag(res @ res / (len(Y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return beta, beta / se, 1 - res @ res / ((Y - Y.mean()) @ (Y - Y.mean())), len(Y)


def show(df, cols, label=''):
    for b in BANDS:
        beta, t, r2, n = fit(df, cols, f'D{b}')
        print(f'  {label:14s} D{b:<4d} ~ 1 + {" + ".join(cols):16s} n={n:3d} R2={r2:.2f}  ' +
              '  '.join(f'{c}={v:+.2f}(t={tt:+.1f})' for c, v, tt in zip(['1'] + cols, beta, t)))


def slopes(df):
    '''Offset and slope per unit AOT for the three bands.'''
    out = {}
    for b in BANDS:
        beta, t, _, n = fit(df, ['aot550'], f'D{b}')
        out[f'slope {b}'], out[f't {b}'] = beta[1], t[1]
    out['N'] = len(df)
    return out


def aot_derivative(df, cache):
    '''dR_atm/dAOT (x1e-3) of the mean retrieved mixture at the median geometry (LUT), at 865, 1610, 2190 nm.'''
    import radcalnet_oc as radoc
    logging.getLogger().setLevel(logging.ERROR)
    f3 = xr.concat([xr.open_dataset(f'{cache}/{s}.nc').f3 for s in SITES], 'time').mean('time')
    lut = radoc.LUT()
    s0, sv = df.sza.median(), df.vza.median()
    # azimuth of the LUT from the median glint angle: cos(glint) = mu0 muv + sin0 sinv cos(azi)
    ca = (np.cos(np.radians(df.glint_angle.median())) - np.cos(np.radians(s0)) * np.cos(np.radians(sv))) / (
        np.sin(np.radians(s0)) * np.sin(np.radians(sv)))
    azi = float(np.degrees(np.arccos(np.clip(ca, -1, 1))))
    lut.lut_preparation(wind=float(df.wind.median()), sza=[s0], vza=[sv], azi=[azi], aot_refs=[0.05, 0.15],
                        aerosol_combination=f3)
    R = lut.Rdiff_lut.squeeze(drop=True).interp(wl=BANDS, method='quadratic')
    return 1e3 * (R.sel(aot_ref=0.15) - R.sel(aot_ref=0.05)) / 0.1, (s0, sv, azi), f3


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--odir', default=ODIR, help='directory of the matchup files')
    p.add_argument('--cache', default=CACHE, help='directory of the results of aerosol_diagnosis.py')
    p.add_argument('--glint-stats', default='cm_iso', help='glint statistics of the matchup files (suffix)')
    p.add_argument('--with-glint', action='store_true',
                   help='deficit with the simulated direct glint (simg - sat) instead of sim - sat')
    p.add_argument('--flags-exclude', nargs='+', default=None, choices=list(FLAGS),
                   help='glint screening by the sunglint flags instead of the glint angle > 30 deg')
    args = p.parse_args()
    warnings.simplefilter('ignore')
    pd.set_option('display.width', 200)
    df = pd.concat([load(args.odir, s, glint_stats=args.glint_stats, with_glint=args.with_glint,
                         flags_exclude=args.flags_exclude) for s in SITES])
    print(f'    glint statistics {args.glint_stats}, deficit {"with" if args.with_glint else "without"} the simulated '
          f'glint, screening ' + (f'without {"/".join(args.flags_exclude)}' if args.flags_exclude else
                                  'glint angle > 30 deg'))
    print(f'=== {len(df)} matchups ({", ".join(f"{s}: {(df.site == s).sum()}" for s in SITES)})')
    print('    deficit = offset + slope x AOT(550):')
    show(df, ['aot550'], 'all')

    print('\n=== 1. thin cirrus: reflectance observed in B10 (1375 nm, x1e-3)')
    print(f'    B10 p10/50/90: {np.nanpercentile(df.B10, [10, 50, 90]).round(2)}; '
          f'correlation with AOT(550): {df[["B10", "aot550"]].corr().iloc[0, 1]:.2f}, '
          f'with the Angstrom exponent: {df[["B10", "ang_exp"]].corr().iloc[0, 1]:.2f}')
    show(df, ['B10'], 'cirrus')
    show(df, ['aot550', 'B10'], 'cirrus+AOT')
    q = pd.qcut(df.B10, 3)
    print('    median deficit and AOT per B10 tercile')
    print(df.groupby(q, observed=True)[[f'D{b}' for b in BANDS] + ['aot550', 'ang_exp']].median()
          .assign(N=df.groupby(q, observed=True).size()).round(3).to_string())
    print('    slope with AOT per B10 tercile')
    print(pd.DataFrame({str(k): slopes(g) for k, g in df.groupby(q, observed=True)}).T.round(2).to_string())

    print('\n=== 2. representativeness of the AERONET-OC AOT')
    dR, geom, f3 = aot_derivative(df, args.cache)
    obs = {b: fit(df, ['aot550'], f'D{b}')[0][1] for b in BANDS}
    print(f'    dR_atm/dAOT of the mean retrieved mixture at the median geometry (sza {geom[0]:.0f}, vza {geom[1]:.0f}, '
          f'azi {geom[2]:.0f} deg), x1e-3 per unit AOT: ' +
          ', '.join(f'{b}: {float(dR.sel(wl=b)):.1f}' for b in BANDS))
    print('    ratio to 2190 nm, model: ' + ', '.join(f'{b}: {float(dR.sel(wl=b) / dR.sel(wl=2190)):.2f}' for b in BANDS) +
          ';  observed slope: ' + ', '.join(f'{b}: {obs[b] / obs[2190]:.2f}' for b in BANDS))
    print('    AOT error needed to explain the slope at each band (fraction of the AOT): ' +
          ', '.join(f'{b}: {-obs[b] / float(dR.sel(wl=b)):.2f}' for b in BANDS))
    print('    slope per bin of time lag of the nearest measurement and of AOT variability in the window (cv)')
    rows = {}
    for name, mask in [('|dt| < 10 min', df.abs_dt < 10), ('|dt| >= 10 min', df.abs_dt >= 10),
                       ('AOT cv < median', df.aot_cv < df.aot_cv.median()),
                       ('AOT cv >= median', df.aot_cv >= df.aot_cv.median())]:
        rows[name] = slopes(df[mask])
    print(pd.DataFrame(rows).T.round(2).to_string())

    print('\n=== 3. coupling with the glint: slope per glint-angle bin')
    g = pd.cut(df.glint_angle, [30, 40, 50, 60, 90])
    print(pd.DataFrame({str(k): slopes(v) for k, v in df.groupby(g, observed=True)}).T.round(2).to_string())
    show(df, ['aot550', 'glint_angle'], 'AOT+glint')

    print('\n=== 4. wind')
    print(f'    correlation AOT(550)-wind: {df[["aot550", "wind"]].corr().iloc[0, 1]:.2f}')
    show(df, ['aot550', 'wind'], 'AOT+wind')
    w = pd.cut(df.wind, [0, 2, 4, 15])
    print(pd.DataFrame({str(k): slopes(v) for k, v in df.groupby(w, observed=True)}).T.round(2).to_string())


    if args.with_glint:
        return
    print('\n=== 5. seasonal confound with the glint angle')
    print(f'    correlation of AOT(550) with the glint angle: {df[["aot550", "glint_angle"]].corr().iloc[0, 1]:.2f}, '
          f'with SZA: {df[["aot550", "sza"]].corr().iloc[0, 1]:.2f}')
    season = df.month.map(lambda m: ['DJF', 'MAM', 'JJA', 'SON'][(m % 12) // 3])
    print(df.groupby(season)[['aot550', 'glint_angle', 'sza'] + [f'D{b}' for b in BANDS]].median()
          .assign(N=df.groupby(season).size()).round(3).to_string())
    show(df, ['glint_angle', 'sza'], 'glint+SZA')
    show(df, ['glint_angle', 'B10'], 'glint+B10')
    show(df, ['aot550', 'glint_angle', 'B10'], 'AOT+glint+B10')
    print('    simulated direct glint G (isotropic Cox-Munk) and deficit with it, Dg = D + G, per glint-angle bin')
    cols = [f'G{b}' for b in BANDS] + [f'D{b}' for b in BANDS]
    for b in BANDS:
        df[f'Dg{b}'] = df[f'D{b}'] + df[f'G{b}']
    print(df.groupby(g, observed=True)[cols + [f'Dg{b}' for b in BANDS] + ['aot550', 'wind']].median()
          .assign(N=df.groupby(g, observed=True).size()).round(2).to_string())
    for b in BANDS:
        beta, t, r2, n = fit(df, [f'G{b}', 'B10'], f'D{b}')
        print(f'  D{b} ~ 1 + G + B10: n={n} R2={r2:.2f}  ' +
              '  '.join(f'{c}={v:+.2f}(t={tt:+.1f})' for c, v, tt in zip(['1', 'G', 'B10'], beta, t)))
    for b in BANDS:
        beta, t, r2, n = fit(df, ['glint_angle', 'B10'], f'Dg{b}')
        print(f'  Dg{b} ~ 1 + glint_angle + B10: n={n} R2={r2:.2f}  ' +
              '  '.join(f'{c}={v:+.3f}(t={tt:+.1f})' for c, v, tt in zip(['1', 'glint', 'B10'], beta, t)))


if __name__ == '__main__':
    main()
