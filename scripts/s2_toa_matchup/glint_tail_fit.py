'''
Heavy-tailed slope statistics of radcalnet_oc.Sunglint fitted on the SWIR excess of the Sentinel-2 TOA matchups.

The SWIR excess E = sat - sim (simulation without the direct glint, corrected for the cirrus term) requires a slope
distribution with a much heavier tail than the Gaussian of Cox and Munk at the AERONET-OC wind
(``glint_slope_inversion.py``). Two heavy-tailed isotropic statistics of ``Sunglint`` are fitted here, at the wind
of the station and with the atmospheric factor of the matchup script:

- ``cm_iso_t``: 2-D Student t with the Cox-Munk mean square slope, parameter nu (degrees of freedom);
- ``cm_iso_mix``: Cox-Munk Gaussian plus a fraction f_tail of wide slopes of mean square slope sigma2_tail.

The mean square slope of the core can be scaled by a factor a (sigma2 = a (0.003 + 5.12e-3 W), through an
effective wind), since near the glint (glint angle < 30 deg) Cox-Munk at the station wind is too peaked. The
parameters minimise the median absolute residual at 1610 and 2190 nm (sum of both), over the glint angles larger
than ``--glint-min`` (0 by default: all the glint angles; 30 for the fit of the far tail only). The fit is then
checked by inverting the wind of each scene with the fitted statistics: if the distribution is right, the
equivalent wind no longer depends on the glint angle and follows the station wind.

Example::

    python glint_tail_fit.py                  # all glint angles, core scaled
    python glint_tail_fit.py --glint-min 30   # far tail only
'''
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

import radcalnet_oc as radoc
from glint_slope_inversion import ODIR, SITES, BANDS, load

FIT_BANDS = [1610, 2190]
A_CORE = [1., 1.5, 2., 3., 4.]  # scale factor of the mean square slope of the core
GRID = {'cm_iso': [{'a': a} for a in A_CORE + [6., 8.]],
        'cm_iso_t': [{'a': a, 'nu': nu} for a, nu in itertools.product(A_CORE, [2.5, 3., 4., 5., 7., 10., 20.])],
        'cm_iso_mix': [{'a': a, 'f_tail': f, 'sigma2_tail': s} for a, f, s in
                       itertools.product(A_CORE, [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
                                         [0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3])]}
BINS = [0, 20, 25, 30, 40, 50, 60, 90]
WINDS = np.concatenate([np.arange(0., 10., 0.25), np.arange(10., 40.1, 1.)])  # inversion grid (m/s)


def effective_wind(wind, a):
    '''Wind giving the Cox-Munk mean square slope a (0.003 + 5.12e-3 W).'''
    return (a * (3e-3 + 512e-5 * np.asarray(wind)) - 3e-3) / 512e-5


def surface(df, wind, stats, kw):
    '''Surface glint (shadowing included) of each scene for the wind ``wind`` (array or scalar); ``kw['a']`` scales
    the mean square slope of the core.'''
    kw = dict(kw)
    wind = effective_wind(np.broadcast_to(wind, len(df)), kw.pop('a', 1.))
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
    p.add_argument('--glint-min', type=float, default=0., help='minimum glint angle of the fit (deg)')
    args = p.parse_args()
    warnings.simplefilter('ignore')
    pd.set_option('display.width', 200)
    # statistics without the SWIR threshold (which removes the most glinted scenes near the specular direction)
    df = pd.concat([load(args.odir, s, glint_min=args.glint_min, glint_pixels=True) for s in SITES])
    # cirrus-corrected excess (coefficient of B10 in D ~ 1 + glint angle + B10, fitted beyond 30 deg, where the
    # glint-angle dependence is close to linear)
    E = {}
    far = df.glint_angle > 30
    for b in BANDS:
        X = np.column_stack([np.ones(far.sum()), df.glint_angle[far], df.B10[far]])
        e = np.linalg.lstsq(X, df[f'D{b}'][far].values, rcond=None)[0][2]
        E[b] = -(df[f'D{b}'] - e * df.B10).values
    gbin = pd.cut(df.glint_angle.values, [b for b in BINS if b >= args.glint_min or b == BINS[-1]])
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
    print(f'\n=== median absolute residual (x1e-3) at the station wind, best parameters per statistics '
          f'(bias: median residual E - G per glint-angle bin '
          f'{list(pd.Series(gbin).value_counts(sort=False).loc[lambda c: c > 0].index.astype(str))})')
    for stats, r in best.items():
        print(f'  {stats:10s} {r["kw"]}: ' + ', '.join(f'MAD {b} {r[f"MAD {b}"]:.2f}' for b in BANDS) +
              '\n             ' + '; '.join(f'bias {b} {r[f"bias {b}"]}' for b in BANDS))
    print('\n    cm_iso, score (MAD 1610 + 2190) per a: ' +
          ', '.join(f'{a:g}: {s:.2f}' for a, s in res[res.stats == 'cm_iso'][['a', 'score']].values))
    print('    cm_iso_t, score per a (rows) and nu (columns):')
    print(res[res.stats == 'cm_iso_t'].pivot(index='a', columns='nu', values='score').round(2).to_string())
    m = res[res.stats == 'cm_iso_mix']
    a_best = best['cm_iso_mix']['kw']['a']
    print(f'    cm_iso_mix, score per f_tail (rows) and sigma2_tail (columns) at a = {a_best:g}:')
    print(m[m.a == a_best].pivot(index='f_tail', columns='sigma2_tail', values='score').round(2).to_string())

    print('\n=== wind inverted per scene from the excess at 2190 nm with each statistics (median per glint-angle bin)')
    for stats, r in best.items():
        u = pd.Series(invert_wind(df, E[2190], df.T2190.values, stats, r['kw']))
        ok = np.isfinite(u)
        tab = pd.DataFrame({'N': u.groupby(gbin, observed=True).size(),
                            'no excess': u.isna().groupby(gbin, observed=True).mean(),
                            'U_eq': u[ok].groupby(gbin[ok.values], observed=True).median(),
                            'U station': df.wind.groupby(gbin, observed=True).median().values})
        print(f'  {stats} {r["kw"]}: Spearman U_eq vs glint angle '
              f'{u[ok].corr(pd.Series(df.glint_angle.values)[ok], method="spearman"):.2f}, vs station wind '
              f'{u[ok].corr(pd.Series(df.wind.values)[ok], method="spearman"):.2f}; median U_eq - U '
              f'{(u[ok] - df.wind.values[ok]).median():.1f} m/s')
        print(tab.round(2).to_string())


if __name__ == '__main__':
    main()
