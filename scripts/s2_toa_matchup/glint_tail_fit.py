'''
Heavy-tailed slope statistics of radcalnet_oc.Sunglint fitted on the SWIR excess of the Sentinel-2 TOA matchups.

The SWIR excess E = sat - sim (simulation without the direct glint, corrected for the cirrus term) requires a slope
distribution with a much heavier tail than the Gaussian of Cox and Munk at the AERONET-OC wind
(``glint_slope_inversion.py``). Two heavy-tailed isotropic statistics of ``Sunglint`` are fitted here, at the wind
of the station and with the atmospheric factor of the matchup script:

- ``cm_iso_t``: 2-D Student t with the Cox-Munk mean square slope, parameter nu (degrees of freedom);
- ``cm_iso_mix``: Cox-Munk Gaussian plus a fraction f_tail of wide slopes of mean square slope sigma2_tail.

The parameters minimise the median absolute residual at 1610 and 2190 nm (sum of both). The fit is then checked by
inverting the wind of each scene with the fitted statistics: if the tail is right, the equivalent wind no longer
depends on the glint angle and follows the station wind.

Example::

    python glint_tail_fit.py
'''
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

import radcalnet_oc as radoc
from glint_slope_inversion import ODIR, SITES, BANDS, load

FIT_BANDS = [1610, 2190]
GRID = {'cm_iso': [{}],
        'cm_iso_t': [{'nu': nu} for nu in [2.1, 2.2, 2.3, 2.5, 2.75, 3., 3.5, 4., 5., 7., 10., 20.]],
        'cm_iso_mix': [{'f_tail': f, 'sigma2_tail': s} for f, s in
                       itertools.product([0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15],
                                         [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3])]}
WINDS = np.concatenate([np.arange(0., 10., 0.25), np.arange(10., 40.1, 1.)])  # inversion grid (m/s)


def surface(df, wind, stats, kw):
    '''Surface glint (shadowing included) of each scene for the wind ``wind`` (array or scalar).'''
    wind = np.broadcast_to(wind, len(df))
    return np.array([max(radoc.Sunglint(sza, vza, 180. - azi).sunglint(w, stats=stats, **kw)[0], 0.)
                     for (sza, vza, azi), w in zip(df[['sza', 'vza', 'azi']].values, wind)])


def invert_wind(df, E, T, stats, kw):
    '''Smallest wind whose TOA glint reaches the excess E (x1e-3) for each scene; NaN if E <= 0, inf if never.'''
    G = np.column_stack([1e3 * T * surface(df, w, stats, kw) for w in WINDS])
    out = np.full(len(df), np.nan)
    for i, (e, g) in enumerate(zip(E, G)):
        if not np.isfinite(e) or e <= 0:
            continue
        k = np.argmax(g >= e)
        if g[k] < e:
            out[i] = np.inf
        elif k == 0:
            out[i] = WINDS[0]
        else:
            out[i] = WINDS[k - 1] + (e - g[k - 1]) / (g[k] - g[k - 1]) * (WINDS[k] - WINDS[k - 1])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--odir', default=ODIR, help='directory of the matchup files')
    args = p.parse_args()
    warnings.simplefilter('ignore')
    pd.set_option('display.width', 200)
    df = pd.concat([load(args.odir, s) for s in SITES])
    # cirrus-corrected excess (coefficient of B10 in D ~ 1 + glint angle + B10)
    E = {}
    for b in BANDS:
        X = np.column_stack([np.ones(len(df)), df.glint_angle, df.B10])
        e = np.linalg.lstsq(X, df[f'D{b}'].values, rcond=None)[0][2]
        E[b] = -(df[f'D{b}'] - e * df.B10).values
    gbin = pd.cut(df.glint_angle.values, [30, 40, 50, 60, 90])
    print(f'=== {len(df)} matchups; cirrus-corrected excess (x1e-3), median per glint-angle bin: ' +
          '; '.join(f'{b}: {pd.Series(E[b]).groupby(gbin, observed=True).median().round(2).values}' for b in BANDS))

    rows, best = [], {}
    for stats, grid in GRID.items():
        for kw in grid:
            Rs = surface(df, df.wind.values, stats, kw)
            r = {'stats': stats, **kw}
            for b in BANDS:
                res = E[b] - 1e3 * df[f'T{b}'].values * Rs
                r[f'MAD {b}'] = np.median(np.abs(res))
                r[f'bias {b}'] = pd.Series(res).groupby(gbin, observed=True).median().round(2).values
            r['score'] = sum(r[f'MAD {b}'] for b in FIT_BANDS)
            rows.append(r)
            if stats not in best or r['score'] < best[stats]['score']:
                best[stats] = {**r, 'kw': kw}
    res = pd.DataFrame(rows)
    print('\n=== median absolute residual (x1e-3) at the station wind, best parameters per statistics '
          '(bias: median residual E - G per glint-angle bin 30-40, 40-50, 50-60, >60 deg)')
    for stats, r in best.items():
        print(f'  {stats:10s} {r["kw"]}: ' + ', '.join(f'MAD {b} {r[f"MAD {b}"]:.2f}' for b in BANDS) +
              '\n             ' + '; '.join(f'bias {b} {r[f"bias {b}"]}' for b in BANDS))
    print('\n    cm_iso_t, score (MAD 1610 + 2190) per nu:')
    t = res[res.stats == 'cm_iso_t']
    print('    ' + ', '.join(f'{nu:g}: {s:.2f}' for nu, s in zip(t.nu, t.score)))
    print('    cm_iso_mix, score per f_tail (rows) and sigma2_tail (columns):')
    m = res[res.stats == 'cm_iso_mix'].pivot(index='f_tail', columns='sigma2_tail', values='score')
    print(m.round(2).to_string())

    print('\n=== wind inverted per scene from the excess at 2190 nm with each statistics (median per glint-angle bin)')
    for stats, r in best.items():
        u = pd.Series(invert_wind(df, E[2190], df.T2190.values, stats, r['kw']))
        ok = np.isfinite(u)
        tab = pd.DataFrame({'N': u.groupby(gbin).size(), 'no excess': u.isna().groupby(gbin).mean(),
                            'U_eq': u[ok].groupby(gbin[ok.values]).median(),
                            'U station': df.wind.groupby(gbin).median().values})
        print(f'  {stats} {r["kw"]}: Spearman U_eq vs glint angle '
              f'{u[ok].corr(pd.Series(df.glint_angle.values)[ok], method="spearman"):.2f}, vs station wind '
              f'{u[ok].corr(pd.Series(df.wind.values)[ok], method="spearman"):.2f}; median U_eq - U '
              f'{(u[ok] - df.wind.values[ok]).median():.1f} m/s')
        print(tab.round(2).to_string())


if __name__ == '__main__':
    main()
