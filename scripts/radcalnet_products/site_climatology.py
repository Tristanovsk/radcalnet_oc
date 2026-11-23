'''Atmospheric climatology of an AERONET-OC site for the RadCalNet site questionnaire (e.g.
QA4EO-WGCV-RadCalNet-CORO-Q), from the AERONET-OC database (``aeronet_oc_db``).

Variables: aerosol optical thickness at 500 nm (``aot500``, log-log interpolation of the AERONET-OC bands),
Angstrom exponent 440-870 (``ang_440_870``, from the 443 and 865 nm bands of the SeaPRISM), precipitable water
(``pw``, cm), total ozone (``ozone``, DU), surface pressure (``pressure``, hPa) and wind speed (``wind``, m s-1), as
given in the AERONET-OC files. Runs of identical pressure or wind values lasting more than 2 days (fill values,
e.g. the monthly constant wind of the level 1.5 data of Section-7_Platform from April 2026) are masked, unless
``--keep-constant``.

The levels of ``--levels`` are merged by priority: the first level is used over its whole record, each next level
only after the end of the previous ones (e.g. ``lev20 lev15``: level 2.0, completed by level 1.5 after the last
level-2.0 measurement). The statistics are computed on daily medians: for each calendar month, median, 25 and 75 %
quantiles of the days, number of days and number of distinct months (years) with data; the row ``YEAR`` gives the
same statistics over all the days.

Outputs (``--fig-dir``):

- ``climatology_<code>.png``: monthly distribution of the daily medians (box: quartiles, whiskers: 5-95 %, line:
  median) of the six variables, with the number of days;
- ``timeseries_<code>.png``: time series of the six variables (all measurements, colour per level);
- ``climatology_<code>.csv``: monthly table (median, q25, q75 per variable, days, months);
- ``climatology_<code>.tex``: two LaTeX longtables of the monthly statistics (aerosols and water vapour; ozone,
  pressure and wind), ``median (q25-q75)``, to be included with ``\\input``.

Usage:
    python site_climatology.py --site Section-7_Platform --code CORO --levels lev20 lev15 \\
        --fig-dir <latex>/QA4EO-WGCV-RadCalNet-CORO-Q-v1/media
'''
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aeronet_oc_db

plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

VARS = {  # name: (label, unit, format)
    'aot500': (r'AOT at 500 nm', '', '{:.3f}'),
    'ang_440_870': (r'Angstrom exponent 440-870', '', '{:.2f}'),
    'pw': ('Precipitable water', 'cm', '{:.2f}'),
    'ozone': ('Total ozone', 'DU', '{:.0f}'),
    'pressure': ('Surface pressure', 'hPa', '{:.1f}'),
    'wind': ('Wind speed', r'm s$^{-1}$', '{:.1f}'),
}
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
LEVEL_NAMES = {'lev20': 'level 2.0', 'lev15': 'level 1.5', 'lev10': 'level 1.0'}
LEVEL_COLORS = ['#1f4e79', '#d9822b', '#4a9a5b']


def load_level(site, level):
    '''Measurements of one level: DataFrame (index: time) of the climatology variables.'''
    ds = aeronet_oc_db.open_site(site, level, legacy_names=True).sortby('wl')
    aod = ds['Aerosol_Optical_Depth']
    wl = ds.wl.values
    # AOT at 500 nm: log-log interpolation between the bands bracketing 500 nm
    i1 = np.searchsorted(wl, 500.) - 1
    w0, w1 = wl[i1], wl[i1 + 1]
    a0, a1 = aod.sel(wl=w0).values, aod.sel(wl=w1).values
    with np.errstate(invalid='ignore', divide='ignore'):
        alpha01 = -np.log(a1 / a0) / np.log(w1 / w0)
        aot500 = a0 * (500. / w0) ** (-alpha01)
        a440, a870 = aod.sel(wl=443).values, aod.sel(wl=865).values
        ang = -np.log(a440 / a870) / np.log(443. / 865.)
    df = pd.DataFrame({'aot500': aot500, 'ang_440_870': ang,
                       'pw': ds['Total_Precipitable_Water(cm)'].values,
                       'ozone': ds['Total_Ozone(Du)'].values,
                       'pressure': ds['Pressure(hPa)'].values,
                       'wind': ds['Wind_Speed(ms)'].values},
                      index=pd.DatetimeIndex(ds.time.values, name='time'))
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df.aot500 <= 0, ['aot500', 'ang_440_870']] = np.nan
    df['level'] = level
    return df.sort_index()


def mask_constant(df, variables=('pressure', 'wind'), max_days=2.):
    '''Set to NaN the runs of identical values lasting more than ``max_days`` days (fill values of the
    ancillary data, e.g. a monthly constant wind). Returns the masked DataFrame and the number of masked values.'''
    df = df.copy()
    nmask = {}
    for v in variables:
        x = df[v]
        run = (x.ne(x.shift())).cumsum()
        t = df.index.to_series()
        span = t.groupby(run).transform(lambda s: (s.max() - s.min()) / pd.Timedelta('1D'))
        bad = x.notnull() & (span > max_days)
        df.loc[bad, v] = np.nan
        nmask[v] = int(bad.sum())
    return df, nmask


def merge_levels(site, levels):
    '''Levels merged by priority (each next level only after the end of the previous ones).'''
    out, end = [], None
    for level in levels:
        df = load_level(site, level)
        if end is not None:
            df = df[df.index > end]
        if len(df):
            out.append(df)
            end = df.index.max() if end is None else max(end, df.index.max())
    return pd.concat(out).sort_index()


def monthly_table(df):
    '''Monthly statistics of the daily medians: DataFrame indexed by JAN..DEC, YEAR.'''
    daily = df[list(VARS)].resample('D').median().dropna(how='all')
    daily['month'] = daily.index.month
    daily['ym'] = daily.index.to_period('M')
    rows = {}
    groups = [(MONTHS[m - 1], daily[daily.month == m]) for m in range(1, 13)] + [('YEAR', daily)]
    for name, g in groups:
        row = {}
        for v in VARS:
            x = g[v].dropna()
            row[f'{v}_median'] = x.median() if len(x) else np.nan
            row[f'{v}_q25'] = x.quantile(0.25) if len(x) else np.nan
            row[f'{v}_q75'] = x.quantile(0.75) if len(x) else np.nan
        row['days'] = len(g)
        row['months'] = g.ym.nunique()
        rows[name] = row
    return pd.DataFrame(rows).T, daily


def fig_climatology(daily, code, period, ofile):
    fig, axs = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for ax, (v, (label, unit, _)) in zip(axs.flat, VARS.items()):
        data = [daily.loc[daily.month == m, v].dropna().values for m in range(1, 13)]
        pos = [m for m, d in zip(range(1, 13), data) if len(d)]
        data = [d for d in data if len(d)]
        ax.boxplot(data, positions=pos, widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='#9ec3e0', edgecolor='#1f4e79'),
                   medianprops=dict(color='k', lw=1.5), whiskerprops=dict(color='#1f4e79'),
                   capprops=dict(color='#1f4e79'))
        ax.set_ylabel(f'{label} ({unit})' if unit else label)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([m[0] for m in MONTHS])
    ndays = [int((daily.month == m).sum()) for m in range(1, 13)]
    for ax in axs[0]:
        top = ax.secondary_xaxis('top')
        top.set_xticks(range(1, 13))
        top.set_xticklabels([str(n) for n in ndays], fontsize=8)
        top.tick_params(length=0)
    axs[0, 0].set_title('number of days per month', fontsize=8, loc='left', pad=14)
    fig.suptitle(f'{code}: monthly distribution of the daily medians, {period}', fontsize=11)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)


def fig_timeseries(df, code, levels, ofile):
    fig, axs = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    for ax, (v, (label, unit, _)) in zip(axs, VARS.items()):
        for level, c in zip(levels, LEVEL_COLORS):
            d = df[df.level == level]
            ax.plot(d.index, d[v], '.', ms=2, color=c, label=LEVEL_NAMES.get(level, level))
        ax.set_ylabel(f'{label}\n({unit})' if unit else label, fontsize=9)
    axs[0].legend(loc='upper left', markerscale=5, fontsize=8, ncol=len(levels))
    fig.suptitle(f'{code}: AERONET-OC measurements', fontsize=11)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)


def cell(tab, v, name):
    fmt = VARS[v][2]
    m, q1, q3 = tab.loc[name, f'{v}_median'], tab.loc[name, f'{v}_q25'], tab.loc[name, f'{v}_q75']
    if not np.isfinite(m):
        return '--'
    return f'{fmt.format(m)} ({fmt.format(q1)}--{fmt.format(q3)})'


def latex_tables(tab, caption_period):
    '''Two longtables (labels tab:clim_aer and tab:clim_met).'''
    groups = [(['aot500', 'ang_440_870', 'pw'],
               [r'$\tau_{a500}$', r'$\alpha_{440-870}$', 'PW (cm)'],
               'Aerosol optical thickness at 500 nm ($\\tau_{a500}$), Angstrom exponent '
               '($\\alpha_{440-870}$) and precipitable water (PW)', 'tab:clim_aer'),
              (['ozone', 'pressure', 'wind'], ['Ozone (DU)', 'Pressure (hPa)', r'Wind (m s$^{-1}$)'],
               'Total ozone, surface pressure and wind speed',
               'tab:clim_met')]
    out = []
    for vs, heads, what, label in groups:
        out.append(r'\begin{longtable}{@{}lrrrrr@{}}')
        out.append(r'\caption{' + what + ': median (25--75\\,\\% range) of the daily medians, number of days ('
                   r'\emph{N}) and of months with data (Months); ' + caption_period + r'.}'
                   + rf'\label{{{label}}}\\')
        out.append(r'\toprule')
        out.append(r'\textbf{Month} & ' + ' & '.join(rf'\textbf{{{h}}}' for h in heads)
                   + r' & \emph{\textbf{N}} & \textbf{Months} \\')
        out.append(r'\midrule')
        out.append(r'\endhead')
        out.append(r'\bottomrule')
        out.append(r'\endlastfoot')
        for name in tab.index:
            if name == 'YEAR':
                out.append(r'\midrule')
            out.append(rf'\textbf{{{name}}} & ' + ' & '.join(cell(tab, v, name) for v in vs)
                       + f' & {int(tab.loc[name, "days"])} & {int(tab.loc[name, "months"])} \\\\')
        out.append(r'\end{longtable}')
        out.append('')
    return '\n'.join(out)


def period_text(df, levels, dash='--'):
    parts = []
    for level in levels:
        d = df[df.level == level]
        if len(d):
            parts.append(f'{LEVEL_NAMES.get(level, level)} {d.index.min():%B %Y}{dash}{d.index.max():%B %Y}')
    return ', '.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='Section-7_Platform', help='AERONET-OC site name')
    p.add_argument('--code', default=None, help='RadCalNet site code for the file names (default: --site)')
    p.add_argument('--levels', nargs='+', default=['lev20', 'lev15'],
                   help='AERONET-OC levels, by priority (default: lev20 lev15)')
    p.add_argument('--fig-dir', required=True, help='output directory')
    p.add_argument('--keep-constant', action='store_true',
                   help='keep the runs of constant pressure or wind lasting more than 2 days (masked by default)')
    args = p.parse_args()
    code = args.code or args.site
    os.makedirs(args.fig_dir, exist_ok=True)

    df = merge_levels(args.site, args.levels)
    if not args.keep_constant:
        df, nmask = mask_constant(df)
        print('constant runs masked (> 2 days):', nmask)
    tab, daily = monthly_table(df)
    period = period_text(df, args.levels)
    tab.to_csv(os.path.join(args.fig_dir, f'climatology_{code}.csv'), float_format='%.4g')
    with open(os.path.join(args.fig_dir, f'climatology_{code}.tex'), 'w') as f:
        f.write(f'% generated by scripts/radcalnet_products/site_climatology.py --site {args.site} '
                f'--levels {" ".join(args.levels)}\n')
        f.write(latex_tables(tab, f'AERONET-OC {args.site.replace("_", chr(92) + "_")}, {period}'))
    fig_climatology(daily, code, period_text(df, args.levels, '\u2013'),
                    os.path.join(args.fig_dir, f'climatology_{code}.png'))
    fig_timeseries(df, code, args.levels, os.path.join(args.fig_dir, f'timeseries_{code}.png'))
    print(f'{args.site}: {len(df)} measurements, {len(daily)} days; {period}')
    print(tab[[c for c in tab.columns if c.endswith('median')] + ['days', 'months']].to_string(
        float_format=lambda x: f'{x:.3g}'))


if __name__ == '__main__':
    main()
