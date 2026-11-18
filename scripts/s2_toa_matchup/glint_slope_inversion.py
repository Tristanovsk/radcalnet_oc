'''
Slope variance of the sea surface inverted per scene from the SWIR deficit of the Sentinel-2 TOA matchups.

The deficit (sim - sat, simulation without the direct glint) decreases with the glint angle, also from 40 to 60 deg
where the isotropic Cox-Munk glint at the AERONET-OC wind is zero (``aot_term_search.py``). If it is direct glint,
a slope variance sigma2 exists for each scene such that the TOA glint matches the excess E = sat - sim:

    E(wl) = T(wl) R_g(sigma2),    R_g: isotropic Cox-Munk (with shadowing) of radcalnet_oc.Sunglint,

with T = Tg Tdir the atmospheric factor of the matchup script (glint_median / Rg_surface). sigma2 is solved on
the rising branch of R_g(sigma2) (sigma2 < tan^2 theta_n), expressed as the equivalent Cox-Munk wind
U = (sigma2 - 0.003) / 0.00512 and compared with the AERONET-OC wind. Two versions of the excess are inverted:

- raw: E = -D;
- cirrus-corrected: E = -(D - e B10), with e the coefficient of the B10 reflectance in D ~ 1 + glint angle + B10
  (``aot_term_search.py``), to remove the high-cloud contribution.

The inversion is done at 1610 and 2190 nm separately (the glint is spectrally flat apart from T, so both bands must
give the same sigma2 if the excess is glint) and the residual at 865 nm of the 2190-nm solution is checked.

A single slope variance per scene should not depend on the geometry. Since it does (it grows with the glint angle),
the shape of the slope distribution is examined directly:

- empirical slope pdf p = E / (T K) per bin of facet tilt theta_n, with K = R_g / p_CM (Fresnel, shadowing and
  geometry), compared with the Gaussian of the station wind and of 5 and 10 m/s;
- the directional statistics (Cox-Munk Gram-Charlier, Breon and Henriot 2006, heavier tails) at the station wind,
  averaged over the wind directions (not measured);
- a two-component distribution (1 - f) p_CM(sigma2_station) + f p_CM(sigma2_wide) (+ offset c0), fitted by
  least absolute deviations over all the scenes.

Example::

    python glint_slope_inversion.py
'''
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import radcalnet_oc as radoc

ODIR = '/data/radcalnet/matchup/sentinel2'
SITES = ['AAOT_lev20', 'Galata_Platform_lev20', 'AAOT_lev15']
BANDS = [865, 1610, 2190]
IQR_GLINT = 0.15  # relative IQR at 2190 nm below which a scene rejected by the SWIR threshold is glint, not cloud
SIGMA2 = np.geomspace(0.003, 0.5, 300)  # slope variance grid (cm_iso: 0.003 at zero wind, 0.157 at 30 m/s)


def wind_eq(sigma2):
    '''Equivalent wind speed of the isotropic Cox-Munk slope variance.'''
    return (sigma2 - 3e-3) / 512e-5


def load(odir, site, dt_max=60, aot_max=0.3, glint_min=30, glint_pixels=False):
    '''
    Selected matchups (same selection as aot_term_search.py), deficit and glint x1e-3.

    :param glint_pixels: if True, satellite statistics without the SWIR threshold of the clear pixels
        (``satg_*``): near the specular direction, the threshold removes the most glinted scenes and biases the
        glint low. The scenes rejected by the threshold are kept when they are homogeneous in the box (relative
        interquartile range at 2190 nm < ``IQR_GLINT``), which separates the glint (< 0.1) from the clouds (> 0.25)
    '''
    ds = xr.open_dataset(f'{odir}/{site}_s2_toa_matchup.nc')
    clear = (ds.frac_valid > 0.5) & (ds.n_pix >= 50)
    if glint_pixels:
        iqr = ((ds.satg_q75 - ds.satg_q25) / ds.satg_median).sel(wl=2190, drop=True)
        valid = clear | ((ds.frac_valid_glint > 0.5) & (ds.n_pix_glint >= 50) & (iqr < IQR_GLINT))
        sat = ds.satg_median
    else:
        valid, sat = clear, ds.sat_median
    sel = valid & (np.abs(ds.dt_nearest) <= dt_max) & (ds.aot550 < aot_max) & (ds.glint_angle > glint_min)
    d = ds.sel(time=sel)
    sat = sat.sel(time=sel)
    df = pd.DataFrame({f'D{b}': 1e3 * (d.sim_median - sat).sel(wl=b).values for b in BANDS},
                      index=pd.DatetimeIndex(d.time.values))
    for b in BANDS:
        df[f'G{b}'] = 1e3 * d.glint_median.sel(wl=b).values
        # atmospheric factor Tg Tdir of the glint (ratio of the medians over the measurements of the window)
        T = (d.glint_median.sel(wl=b) / d.Rg_surface).values
        df[f'T{b}'] = np.where((d.Rg_surface.values > 1e-10) & (T > 0.3) & (T <= 1), T, np.nan)
    df['B10'] = 1e3 * sat.sel(wl=1375).values
    for v in ['aot550', 'wind', 'sza', 'vza', 'azi', 'glint_angle']:
        df[v] = d[v].values
    df['site'] = site
    df = df.dropna(subset=['D2190'])
    # missing atmospheric factor (glint underflow): median of the site
    for b in BANDS:
        df[f'T{b}'] = df[f'T{b}'].fillna(df[f'T{b}'].median())
    return df


def surface_glint(df):
    '''Isotropic Cox-Munk surface glint R_g(sigma2) for each scene (rows) on the SIGMA2 grid, and tan^2 theta_n.'''
    R = np.empty((len(df), len(SIGMA2)))
    tan2 = np.empty(len(df))
    for i, (sza, vza, azi) in enumerate(df[['sza', 'vza', 'azi']].values):
        g = radoc.Sunglint(sza, vza, 180. - azi)  # same convention as s2_toa_matchup.sunglint_toa
        R[i] = [max(g.sunglint(wind_eq(s2), stats='cm_iso')[0], 0.) for s2 in SIGMA2]
        zx, zy, _, _ = g.sunglint(0., stats='cm_iso', slope=True)
        tan2[i] = zx ** 2 + zy ** 2
    return R, tan2


def invert(E, R, tan2):
    '''
    sigma2 such that R(sigma2) = E on the rising branch (sigma2 < tan2), per scene.

    :return: sigma2 (NaN when E <= 0: no excess; inf when E is larger than the maximum of the rising branch)
    '''
    out = np.full(len(E), np.nan)
    for i, (e, r, t2) in enumerate(zip(E, R, tan2)):
        if not np.isfinite(e) or e <= 0:
            continue
        rising = SIGMA2 <= t2
        x, y = SIGMA2[rising], r[rising]
        if y.size < 2 or e > y.max():
            out[i] = np.inf
            continue
        k = np.argmax(y >= e)
        if k == 0:
            out[i] = x[0]
            continue
        # log-linear interpolation (R grows as exp(-tan2/sigma2))
        ly0, ly1 = np.log(max(y[k - 1], 1e-300)), np.log(y[k])
        out[i] = x[k - 1] + (np.log(e) - ly0) / (ly1 - ly0) * (x[k] - x[k - 1])
    return out


def summary(df, key, by):
    '''Median equivalent wind and fractions of scenes without excess / out of range, per group.'''
    def f(g):
        s = g[f's2_{key}']
        ok = np.isfinite(s)
        return pd.Series({'N': len(g), 'no excess': (s.isna()).mean(), 'too large': np.isinf(s).mean(),
                          'sigma2 med': s[ok].median(), 'U_eq med': wind_eq(s[ok]).median(),
                          'U_eq q25': wind_eq(s[ok]).quantile(0.25), 'U_eq q75': wind_eq(s[ok]).quantile(0.75),
                          'U station': g.wind.median()})
    return df.groupby(by, observed=True).apply(f, include_groups=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--odir', default=ODIR, help='directory of the matchup files')
    args = p.parse_args()
    warnings.simplefilter('ignore')
    pd.set_option('display.width', 200)
    df = pd.concat([load(args.odir, s) for s in SITES])
    print(f'=== {len(df)} matchups; atmospheric factor T (median): ' +
          ', '.join(f'{b}: {df[f"T{b}"].median():.2f}' for b in BANDS))

    # cirrus coefficient of D ~ 1 + glint angle + B10
    e = {}
    for b in BANDS:
        X = np.column_stack([np.ones(len(df)), df.glint_angle, df.B10])
        e[b] = np.linalg.lstsq(X, df[f'D{b}'].values, rcond=None)[0][2]
    print('    cirrus coefficient e (D ~ 1 + glint angle + B10): ' + ', '.join(f'{b}: {e[b]:+.2f}' for b in BANDS))

    R, tan2 = surface_glint(df)
    for b in BANDS:
        Rb = 1e3 * df[f'T{b}'].values[:, None] * R
        df[f's2_raw{b}'] = invert(-df[f'D{b}'].values, Rb, tan2)
        df[f's2_cir{b}'] = invert(-(df[f'D{b}'] - e[b] * df.B10).values, Rb, tan2)
    # check: station wind reproduces the simulated glint
    s2_st = 3e-3 + 512e-5 * df.wind.values
    i = np.array([np.argmin(np.abs(SIGMA2 - s)) for s in s2_st])
    G_chk = 1e3 * df.T2190.values * R[np.arange(len(df)), i]
    big = df.G2190 > 0.1
    print(f'    check, glint at the station wind vs glint_median (2190 nm, G > 0.1e-3, n={big.sum()}): '
          f'median ratio {np.median(G_chk[big] / df.G2190[big]):.3f}')

    gbin = pd.cut(df.glint_angle, [30, 40, 50, 60, 90])
    for key in ['raw', 'cir']:
        label = {'raw': 'raw excess E = -D', 'cir': 'cirrus-corrected excess E = -(D - e B10)'}[key]
        for b in [1610, 2190]:
            print(f'\n=== {label}, {b} nm: equivalent Cox-Munk wind per glint-angle bin')
            print(summary(df, f'{key}{b}', gbin).round(3).to_string())

    # consistency of the two SWIR bands and the 865 nm residual of the 2190 nm solution (cirrus-corrected)
    ok = np.isfinite(df.s2_cir1610) & np.isfinite(df.s2_cir2190)
    print(f'\n=== consistency (cirrus-corrected, n={ok.sum()}): U_eq(1610) vs U_eq(2190): median '
          f'{wind_eq(df.s2_cir1610[ok]).median():.1f} / {wind_eq(df.s2_cir2190[ok]).median():.1f} m/s, '
          f'Spearman {df.s2_cir1610[ok].corr(df.s2_cir2190[ok], method="spearman"):.2f}')
    ok = np.isfinite(df.s2_cir2190)
    i = np.array([np.argmin(np.abs(SIGMA2 - s)) for s in df.s2_cir2190[ok]])
    G865 = 1e3 * df.T865[ok].values * R[np.flatnonzero(ok), i]
    res865 = -(df.D865[ok] - e[865] * df.B10[ok]) - G865
    print('    865 nm excess not explained by the 2190 nm glint, median per glint-angle bin (x1e-3):')
    print(res865.groupby(gbin[ok], observed=True).median().round(2).to_string())

    # equivalent wind vs station wind
    for key in ['raw2190', 'cir2190']:
        ok = np.isfinite(df[f's2_{key}'])
        u = wind_eq(df[f's2_{key}'][ok])
        print(f'\n=== {key}: U_eq vs AERONET-OC wind (n={ok.sum()}): Spearman '
              f'{u.corr(df.wind[ok], method="spearman"):.2f}; median U_eq - U = {(u - df.wind[ok]).median():.1f} m/s; '
              f'U_eq vs glint angle: Spearman {u.corr(df.glint_angle[ok], method="spearman"):.2f}')
        w = pd.cut(df.wind[ok], [0, 2, 4, 6, 15])
        print(pd.DataFrame({'N': u.groupby(w, observed=True).size(), 'U_eq med': u.groupby(w, observed=True).median(),
                            'U med': df.wind[ok].groupby(w, observed=True).median()}).round(2).to_string())
        print('    per site: ' + ', '.join(f'{s}: {u[df.site[ok] == s].median():.1f} (station '
                                           f'{df.wind[ok][df.site[ok] == s].median():.1f})' for s in SITES))

    pdf_shape(df, R, tan2, e)


def pdf_shape(df, R, tan2, e):
    '''Empirical slope pdf, directional statistics and two-component fit (cirrus-corrected excess).'''
    s2 = 3e-3 + 512e-5 * df.wind.values
    i = np.array([np.argmin(np.abs(SIGMA2 - s)) for s in s2])
    s2g = SIGMA2[i]
    p_st = np.exp(-tan2 / s2g) / (np.pi * s2g)
    K = R[np.arange(len(df)), i] / p_st  # Fresnel x shadowing x pi / (4 mu0 muv cos^4 theta_n)
    E = {b: -(df[f'D{b}'] - e[b] * df.B10).values for b in BANDS}
    gbin = pd.cut(df.glint_angle.values, [30, 40, 50, 60, 90])

    print('\n=== empirical slope pdf p = E / (T K) per bin of facet tilt theta_n (median), Gaussian pdfs for comparison')
    t = pd.DataFrame({'theta_n': np.degrees(np.arctan(np.sqrt(tan2))), 'tan2': tan2, 'glint': df.glint_angle.values,
                      'wind': df.wind.values})
    for b in BANDS:
        t[f'p{b}'] = E[b] / (1e3 * df[f'T{b}'].values * K)
    out = t.groupby(pd.cut(t.theta_n, [15, 20, 22.5, 25, 27.5, 30, 35, 45]), observed=True).median()
    out.insert(0, 'N', t.groupby(pd.cut(t.theta_n, [15, 20, 22.5, 25, 27.5, 30, 35, 45]), observed=True).size())
    for name, s in [('p_station', 3e-3 + 512e-5 * out.wind), ('p_5ms', 3e-3 + 512e-5 * 5), ('p_10ms', 3e-3 + 512e-5 * 10)]:
        out[name] = np.exp(-out.tan2 / s) / (np.pi * s)
    print(out.drop(columns='theta_n').to_string(float_format=lambda v: f'{v:.3g}'))

    print('\n=== directional statistics at the station wind, averaged over 12 wind directions (2190 nm, x1e-3)')
    rows = []
    for sza, vza, azi, w in df[['sza', 'vza', 'azi', 'wind']].values:
        g = radoc.Sunglint(sza, vza, 180. - azi)
        rows.append([max(g.sunglint(w, stats='cm_iso')[0], 0.)] +
                    [np.mean([max(g.sunglint(w, wazi=a, stats=st)[0], 0.) for a in range(0, 360, 30)])
                     for st in ['cm_dir', 'bh2006']])
    G = pd.DataFrame(1e3 * np.array(rows) * df.T2190.values[:, None], columns=['G cm_iso', 'G cm_dir', 'G bh2006'])
    G['excess E'] = E[2190]
    print(G.groupby(gbin, observed=True).median().round(3).to_string())

    print('\n=== two-component slope distribution (1 - f) p(sigma2_station) + f p(sigma2_wide) + c0, '
          'least absolute deviations')
    for b in BANDS:
        A = 1e3 * df[f'T{b}'].values * K
        best = {}
        for sw in np.geomspace(0.02, 0.5, 40):
            pw = np.exp(-tan2 / sw) / (np.pi * sw)
            for f in np.linspace(0, 0.5, 51):
                M = A * ((1 - f) * p_st + f * pw)
                for c0 in np.linspace(-1, 2, 31):
                    mad = np.median(np.abs(E[b] - M - c0))
                    for key in (['free c0', 'c0 = 0'] if abs(c0) < 1e-9 else ['free c0']):
                        if key not in best or mad < best[key][0]:
                            best[key] = (mad, sw, f, c0, M + c0)
        for key, (mad, sw, f, c0, M) in best.items():
            print(f'  {b} nm, {key:7s}: sigma2_wide {sw:.3f} (U_eq {wind_eq(sw):.0f} m/s), f {f:.2f}, c0 {c0:+.1f}, '
                  f'MAD {mad:.2f} (of E: {np.median(np.abs(E[b] - np.median(E[b]))):.2f}); per glint bin, model '
                  f'{pd.Series(M).groupby(gbin, observed=True).median().round(2).values}, observed '
                  f'{pd.Series(E[b]).groupby(gbin, observed=True).median().round(2).values}')


if __name__ == '__main__':
    main()
