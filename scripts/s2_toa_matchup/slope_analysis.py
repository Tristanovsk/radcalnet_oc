'''
Why the regression slope of the simulation against MSI is below 1 in the NIR-SWIR (uncertainty statements, figure
"matchup_bands": e.g. VEIT 0.71 at 865 nm, 0.36 at 1610 nm, 0.27 at 2190 nm with r = 0.6-0.8).

With MSI on the x axis, the least-squares slope of the simulation y = x - E against x is

    b = 1 - cov(E, x) / var(x),          E = sat - sim (residual),

so a slope below 1 is not a random error of the simulation (independent of the observed signal, it leaves b at 1):
it is a part of the observed signal that the simulation does not follow, cov(E, x) > 0, i.e. a term missing from the
simulation, or under-predicted, that grows with the signal. Noise of the observation (spatial spread in the ROI,
L1C noise) also lowers b by var(noise) / var(x) (regression dilution); it is estimated here from the spread of the ROI
and is small.

The deficit 1 - b is split into the covariances of candidate terms X with the observed signal, cov(X, x) / var(x):

- the predicted missing terms of the physical residual budget (``residual_budget.py``): thin cirrus from the B10
  excess, whitecaps, land environment (each predicted from independent inputs, no coefficient fitted);
- the simulated components themselves: the path reflectance without the glint (``sim_median``, mostly aerosol in
  the SWIR) and the direct glint (``simg_median - sim_median``): if the real component is (1 + k) times the simulated
  one, E contains k X.

A multiple regression E = sum k_X X + c on these terms gives the missing fraction k_X of each one, and the part of
1 - b it explains (k_X cov(X, x) / var(x)); the rest of the deficit is in the intercept-free remainder.

Usage:
    python slope_analysis.py [--site VEIT|CORO] [--selection noglint|nohigh|angle30] [--no-roi] [--csv out.csv]
'''
import argparse
import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'radcalnet_products'))
import radcalnet_oc as radoc  # noqa: E402
import residual_budget as rb  # noqa: E402
from matchup_analysis import load, selection, screen  # noqa: E402

SITES = {'VEIT': (['AAOT_lev20'], (45.3139, 12.5083)),
         'CORO': (['Section-7_Platform_lev20', 'Section-7_Platform_lev15'], (44.5458, 29.4466))}
SELECTIONS = {'angle30': dict(glint_min=30., flags_exclude=None),
              'nohigh': dict(glint_min=0., flags_exclude=['glint_high', 'glint_near_specular']),
              'noglint': dict(glint_min=0., flags_exclude=['glint_moderate', 'glint_high', 'glint_near_specular'])}
# environment left out of the regression: nearly constant at a site, degenerate with the intercept
TERMS = ['path', 'glint', 'cirrus', 'whitecaps']


def matchups(levels, sel, roi):
    '''Matchups of the statement figures (statement_figures.matchups) with all the variables of the files, and the
    full file of the first level (for the AOT sensitivity of the budget).'''
    s = SELECTIONS[sel]
    args = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=s['glint_min'],
                           flags_exclude=s['flags_exclude'], glint_stats='cm_iso_mix', glint_pixels=False, roi=roi)
    out, seen, full = [], set(), None
    for sl in levels:
        ds = load(f'/data/radcalnet/matchups/sentinel2/v{radoc.__version__}', sl, args)
        full = ds if full is None else full
        keep = (selection(ds, args) & screen(ds, args)).values
        keep &= ~np.isin(ds['product'].values.astype(str), list(seen))
        ds = ds.isel(time=keep)
        seen |= set(ds['product'].values.astype(str))
        # variables of the simulations of the window (dimension rec, length different per file) left out
        ds = ds.drop_vars([v for v in rb.DROP if v in ds])
        out.append(ds.drop_dims('rec') if 'rec' in ds.dims else ds)
    import xarray as xr
    if len(out) == 1:
        return full, out[0]
    return full, xr.concat(out, 'time', data_vars='all', coords='minimal', compat='override',
                           join='outer').sortby('time')


def ols(x, y):
    a, c = np.polyfit(x, y, 1)
    return a, c, np.corrcoef(x, y)[0, 1]


def analyse(df, ds):
    rows = []
    for band, g in df.groupby('band'):
        x, y = g.sat.values, g.sim.values
        E = x - y
        var = np.var(x, ddof=1)
        b, c, r = ols(x, y)
        # dilution: spread of the observation in the ROI, as the standard error of the ROI median (n pixels)
        noise = np.nan
        if 'sat_std' in ds and 'n_pix' in ds:
            # variance of the ROI median (1.2533 sigma / sqrt(n) for a normal spread); an upper bound of the
            # observation noise: the spatial spread includes real structures of the water
            sd = ds.sat_std.sel(wl=band).values
            noise = np.nanmean((1.2533 * sd) ** 2 / np.maximum(ds.n_pix.values, 1)) / var
        X = pd.DataFrame({'path': g.sim_path.values, 'glint': g.sim_glint.values, 'cirrus': g.cirrus.values,
                          'whitecaps': g.whitecaps.values, 'environment': g.environment.values})
        share = {t: np.cov(X[t], x)[0, 1] / var for t in TERMS}
        # multiple regression of the residual on the terms (with an intercept)
        A = np.column_stack([X[t].values for t in TERMS] + [np.ones(len(x))])
        coef, *_ = np.linalg.lstsq(A, E, rcond=None)
        k = dict(zip(TERMS, coef[:-1]))
        expl = {t: k[t] * share[t] for t in TERMS}
        fit = A @ coef
        rows.append({'band': band, 'N': len(x), 'slope': b, 'intercept': c, 'r': r, '1-slope': 1 - b,
                     'cov(E,x)/var(x)': np.cov(E, x)[0, 1] / var, 'dilution': noise,
                     'slope with predicted terms': ols(x, y + g.pred.values)[0],
                     **{f'share {t}': share[t] for t in TERMS},
                     **{f'k {t}': k[t] for t in TERMS}, 'k const': coef[-1],
                     **{f'explained {t}': expl[t] for t in TERMS},
                     'R2 of E': 1 - np.var(E - fit) / np.var(E)})
    return pd.DataFrame(rows).set_index('band')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='VEIT', choices=list(SITES))
    p.add_argument('--selection', default='noglint', choices=list(SELECTIONS))
    p.add_argument('--no-roi', action='store_true')
    p.add_argument('--csv', default=None, help='per-band table')
    args = p.parse_args()
    warnings.filterwarnings('ignore')
    radoc.set_logging('WARNING')
    levels, (lat, lon) = SITES[args.site]
    full, ds = matchups(levels, args.selection, not args.no_roi)
    import cartopy.io.shapereader as shpreader
    import shapely
    land = shapely.unary_union(list(shpreader.Reader(
        shpreader.natural_earth('10m', 'physical', 'land')).geometries()))
    df = rb.budget(full, ds, rb.environment_weights(lat, lon, land))
    comp = []
    for band in rb.BANDS:
        comp.append(pd.DataFrame({'band': band, 'sim_path': ds.sim_median.sel(wl=band).values,
                                  'sim_glint': (ds.simg_median - ds.sim_median).sel(wl=band).values}))
    comp = pd.concat(comp, ignore_index=True)
    df[['sim_path', 'sim_glint']] = comp[['sim_path', 'sim_glint']].values
    tab = analyse(df, ds)
    pd.set_option('display.width', 220)
    print(f'{args.site}, selection {args.selection}, ROI {not args.no_roi}: {ds.sizes["time"]} matchups')
    print('\nslope of the simulation against MSI and its deficit:')
    print(tab[['N', 'slope', 'r', '1-slope', 'dilution', 'slope with predicted terms', 'R2 of E']].round(3).to_string())
    print('\ncov(term, MSI) / var(MSI): part of the observed variance carried by each term')
    print(tab[[f'share {t}' for t in TERMS]].round(3).to_string())
    print('\nresidual regressed on the terms: missing fraction k of each term (E = sum k X + c)')
    print(tab[[f'k {t}' for t in TERMS] + ['k const']].round(3).to_string())
    print('\npart of 1 - slope explained by each term (k x share):')
    print(tab[[f'explained {t}' for t in TERMS] + ['1-slope']].round(3).to_string())
    # what drives the residual left after the cirrus (the largest predicted term): Spearman correlations
    import scipy.stats as st
    drivers = {'aot550': ds.aot550.values, 'ang_exp': ds.ang_exp.values, 'wind': ds.wind.values,
               'glint_angle': ds.glint_angle.values, 'sza': ds.sza.values, 'vza': ds.vza.values,
               'B10': ds.sat_median.sel(wl=1375).values,
               'sim glint 865': (ds.simg_median - ds.sim_median).sel(wl=865).values}
    rows = {}
    for band in rb.BANDS:
        g = df[df.band == band]
        resid = g.sat.values - g.sim.values - g.cirrus.values
        rows[band] = {k: st.spearmanr(resid, v, nan_policy='omit').correlation for k, v in drivers.items()}
    print('\nSpearman correlation of the residual after the cirrus (MSI - sim - cirrus) with:')
    print(pd.DataFrame(rows).T.round(2).to_string())
    if args.csv:
        tab.to_csv(args.csv)


if __name__ == '__main__':
    main()
