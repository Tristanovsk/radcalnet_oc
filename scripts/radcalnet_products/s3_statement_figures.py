'''
Sentinel-3 OLCI figures and tables of the RadCalNet site uncertainty statements, from the matchup file of the
Cal/Val monitoring notebook (``notebook/sentinel3/sentinel3_radcalnet_oc_toa_calval_monitoring.ipynb``): ratio
r = L_OLCI / L_sim of the OLCI L1 radiance (box around the station) to the RadCalNet-OC simulation of the AERONET-OC
measurements within +/- 1 h, at the geometry of each scene, with the direct glint.

Selection: scenes without high or near-specular glint (glint flags of the simulation), as the Sentinel-2 comparisons.

Outputs (``--fig-dir``):

- ``s3_ratio.png``: (left) median and interquartile range of r per band and platform, with the standard uncertainty of
  the simulation (k = 1, median) around 1, absorption bands open symbols; (right) r against time at 443 and 865 nm;
- ``s3_ratio.csv`` and ``s3_ratio.tex``: per band and platform N, median r, robust standard deviation, u_sim, Theil-Sen
  drift (%/yr, 90 % interval), and S3A - S3B (%, 90 % bootstrap interval);
- ``s3_glint_slopes.csv``: slope of L_sim against L_OLCI per glint-angle class (|dt| < 15 min) at 665, 865, 1020 nm.

Usage:
    python s3_statement_figures.py --site CORO --fig-dir <document>/media/v1.3
'''
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr

MATCHUP = '/data/radcalnet/matchups/sentinel3/v1.2.0/olci_radcalnet-oc_v1.2.0_VEIT-GABU-CORO_20160401-20261231_matchup.nc'
ABSORPTION = ['Oa13', 'Oa14', 'Oa15', 'Oa19', 'Oa20']
TABLE_BANDS = ['Oa02', 'Oa03', 'Oa04', 'Oa06', 'Oa08', 'Oa11', 'Oa12', 'Oa16', 'Oa17', 'Oa18', 'Oa21']
GLINT_CLASSES = [(0., 35.), (35., 45.), (45., 55.), (55., 90.)]
COLORS = {'S3A': '#1f77b4', 'S3B': '#d62728'}
STYLE = {'font.family': 'serif', 'mathtext.fontset': 'stix',
         'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']}


def rsd(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def select(ds, site):
    keep = (ds.site.values.astype(str) == site) & ~np.isin(ds.glint_level.values.astype(str), ['high', 'near_specular'])
    return ds.isel(time=np.flatnonzero(keep))


def decimal_year(t):
    t = pd.DatetimeIndex(t)
    return t.year + (t.dayofyear - 0.5) / 365.25


def table(ds, rng=np.random.default_rng(0), n_boot=2000):
    rows = []
    wl = ds.lambda0.median('time').values
    for i, b in enumerate(ds.band.values.astype(str)):
        row = {'band': b, 'wl': wl[i]}
        r = {}
        for p in ('S3A', 'S3B'):
            k = ds.platform.values.astype(str) == p
            x = ds.ratio.values[k, i]
            ok = np.isfinite(x)
            r[p] = x[ok]
            row.update({f'N {p}': int(ok.sum()), f'r {p}': np.median(x[ok]), f'rsd {p} %': 100 * rsd(x[ok]),
                        f'u_sim {p} %': 100 * np.nanmedian(ds.u_sim_rel.values[k, i])})
            if ok.sum() >= 10:
                yr = decimal_year(ds.time.values[k][ok])
                s, _, lo, hi = st.theilslopes(x[ok], yr, 0.90)
                m = np.median(x[ok])
                row.update({f'drift {p} %/yr': 100 * s / m, f'drift {p} lo': 100 * lo / m, f'drift {p} hi': 100 * hi / m})
        if len(r['S3A']) >= 10 and len(r['S3B']) >= 10:
            d = [np.median(rng.choice(r['S3A'], len(r['S3A']))) - np.median(rng.choice(r['S3B'], len(r['S3B'])))
                 for _ in range(n_boot)]
            row.update({'S3A-S3B %': 100 * (np.median(r['S3A']) - np.median(r['S3B'])),
                        'S3A-S3B lo': 100 * np.percentile(d, 5), 'S3A-S3B hi': 100 * np.percentile(d, 95)})
        rows.append(row)
    return pd.DataFrame(rows).set_index('band')


def glint_slopes(ds, bands=('Oa08', 'Oa17', 'Oa21')):
    names = list(ds.band.values.astype(str))
    rows = []
    ga = ds.glint_angle.values
    short = np.abs(ds.dt_min.values) < 15
    for g0, g1 in GLINT_CLASSES:
        k = short & (ga >= g0) & (ga < g1)
        row = {'glint angle': f'{g0:g}-{g1:g}', 'N': int(k.sum())}
        for b in bands:
            i = names.index(b)
            x, y = ds.L_olci.values[k, i], ds.Lsim.values[k, i]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 10:
                row[f'{b} slope'] = np.polyfit(x[ok], y[ok], 1)[0]
                row[f'{b} r'] = np.corrcoef(x[ok], y[ok])[0, 1]
        rows.append(row)
    return pd.DataFrame(rows)


def figure(ds, tab, site, ofile):
    wl = tab.wl.values
    absb = tab.index.isin(ABSORPTION)
    with plt.rc_context({**STYLE, 'font.size': 11}):
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.05, 1]})
        ax = axs[0]
        u = np.nanmean([tab['u_sim S3A %'].values, tab['u_sim S3B %'].values], 0) / 100
        ax.fill_between(wl[~absb], 1 - u[~absb], 1 + u[~absb], color='0.85', label='$u_{sim}$ (k = 1)')
        for dx, p in ((-1.5, 'S3A'), (1.5, 'S3B')):
            k = ds.platform.values.astype(str) == p
            q1, q3 = np.nanpercentile(ds.ratio.values[k], [25, 75], axis=0)
            med = tab[f'r {p}'].values
            ax.errorbar(wl[~absb] + dx, med[~absb], yerr=[med[~absb] - q1[~absb], q3[~absb] - med[~absb]], fmt='o',
                        ms=4, color=COLORS[p], capsize=2, label=f'{p} (N = {int(k.sum())})')
            ax.plot(wl[absb] + dx, med[absb], 'o', ms=4, mfc='none', color=COLORS[p])
        ax.axhline(1, color='k', lw=0.8)
        ax.set(xlabel='OLCI band central wavelength (nm)', ylabel='$L_{OLCI} / L_{sim}$', ylim=(0.8, 1.45))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_title('median and interquartile range (open: absorption bands)', fontsize=10)
        ax = axs[1]
        names = list(ds.band.values.astype(str))
        for b, mk in (('Oa03', 'o'), ('Oa17', 's')):
            i = names.index(b)
            for p in ('S3A', 'S3B'):
                k = ds.platform.values.astype(str) == p
                s = pd.Series(ds.ratio.values[k, i], index=pd.DatetimeIndex(ds.time.values[k])).dropna().sort_index()
                ax.plot(s.index, s.values, mk, ms=2.5, alpha=0.35, color=COLORS[p], mfc='none' if b == 'Oa17' else None)
                roll = s.rolling('180D', min_periods=10).median()
                ax.plot(roll.index, roll.values, color=COLORS[p], lw=1.6 if b == 'Oa03' else 1.0,
                        ls='-' if b == 'Oa03' else '--', label=f'{p} {b} ({tab.wl[b]:.0f} nm), 180-day median')
        ax.axhline(1, color='k', lw=0.8)
        ax.set(ylabel='$L_{OLCI} / L_{sim}$', ylim=(0.85, 1.45))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', ncol=2)
        ax.set_title('ratio against time', fontsize=10)
        fig.suptitle(f'{site}: Sentinel-3 OLCI against RadCalNet-OC ({pd.Timestamp(ds.time.values.min()):%Y-%m} to '
                     f'{pd.Timestamp(ds.time.values.max()):%Y-%m}; no high or near-specular glint)', fontsize=12, y=0.99)
        fig.tight_layout()
        fig.savefig(ofile, dpi=200)
        plt.close(fig)


def latex(tab, ofile):
    lines = []
    for b, r in tab.loc[[b for b in TABLE_BANDS if b in tab.index]].iterrows():
        def drift(p):
            return (f"{r[f'drift {p} %/yr']:+.2f} [{r[f'drift {p} lo']:+.2f}, {r[f'drift {p} hi']:+.2f}]"
                    if np.isfinite(r.get(f'drift {p} %/yr', np.nan)) else '--')
        lines.append(f"{b} & {r.wl:.0f} & {r['r S3A']:.3f} & {r['r S3B']:.3f} & {r['rsd S3A %']:.1f} / {r['rsd S3B %']:.1f} & "
                     f"{r['u_sim S3A %']:.1f} & {r['S3A-S3B %']:+.1f} [{r['S3A-S3B lo']:+.1f}, {r['S3A-S3B hi']:+.1f}] & "
                     f"{drift('S3A')} & {drift('S3B')} \\\\")
    with open(ofile, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', required=True, choices=['VEIT', 'GABU', 'CORO'])
    p.add_argument('--matchup', default=MATCHUP)
    p.add_argument('--fig-dir', required=True)
    a = p.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    ds = select(xr.open_dataset(a.matchup).load(), a.site)
    tab = table(ds)
    tab.round(4).to_csv(os.path.join(a.fig_dir, 's3_ratio.csv'))
    latex(tab, os.path.join(a.fig_dir, 's3_ratio.tex'))
    figure(ds, tab, a.site, os.path.join(a.fig_dir, 's3_ratio.png'))
    gs = glint_slopes(ds)
    gs.round(3).to_csv(os.path.join(a.fig_dir, 's3_glint_slopes.csv'), index=False)
    pd.set_option('display.width', 220)
    print(f'{a.site}: {ds.sizes["time"]} matchups ({pd.Series(ds.platform.values.astype(str)).value_counts().to_dict()})')
    cols = ['wl', 'r S3A', 'r S3B', 'rsd S3A %', 'u_sim S3A %', 'S3A-S3B %', 'S3A-S3B lo', 'S3A-S3B hi',
            'drift S3A %/yr', 'drift S3A lo', 'drift S3A hi', 'drift S3B %/yr']
    print(tab.loc[[b for b in TABLE_BANDS if b in tab.index], cols].round(3).to_string())
    print('\nslope per glint-angle class (|dt| < 15 min):\n' + gs.round(2).to_string(index=False))
    ratio = ds.ratio.median('time')
    names = list(ds.band.values.astype(str))
    for num, den in (('Oa13', 'Oa12'), ('Oa14', 'Oa12'), ('Oa20', 'Oa18')):
        print(f'{num}/{den} depth ratio (median r ratio): '
              f'{float(ratio.isel(band=names.index(num)) / ratio.isel(band=names.index(den))):.3f}')


if __name__ == '__main__':
    main()
