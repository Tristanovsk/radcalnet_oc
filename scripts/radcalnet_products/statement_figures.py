'''Figures of the RadCalNet site uncertainty statement of a water site (e.g. QA4EO-WGCV-RadCalNet-VEIT-U), from the
RadCalNet-OC products, the uncertainty budget of radcalnet_oc and the Sentinel-2 matchups.

Figures (PNG, ``--fig-dir``):

- ``rrs_products.png`` (``--only rrs``): distribution of the nadir Rrs of the products and of the chlorophyll
  index of the invRrs reconstruction;
- ``toa_envelope.png``: temporal distribution of the TOA reflectance of the daily products at nadir (median,
  quartiles, 5-95 %), with and without the direct glint;
- ``glint_nadir.png``: fraction of the times of the products flagged for sunglint at nadir, per month;
- ``uncertainty_budget.png``: terms of the uncertainty budget (``radcalnet_oc.uncertainty.uncertainty_budget``) for
  one measurement per month of a year of the site, at nadir: median absolute and relative uncertainties;
- ``matchup_bias.png``: median and interquartile range of the relative difference simulation - MSI per band,
  with and without the simulated glint;
- ``matchup_bands.png``: simulation (with glint) against MSI per band;
- ``matchup_flags.png``: effect of the region of interest and of the sunglint flags of the matchups on the
  statistics (1.2 km box against the recommended ROI, glint angle > 30 deg against the flags, bands flagged
  ``glint_uncertain`` masked).

The matchups use by default the recommended ROI (``s2_toa_matchup.py --roi`` files) and the selection without
moderate, high or near-specular glint (``--selection``).

Usage:
    python statement_figures.py --site VEIT --aeronet AAOT --level lev20 --fig-dir <document>/media [--year 2020]
    python statement_figures.py --site CORO --aeronet Section-7_Platform --level lev20 \
        --matchups Section-7_Platform_lev20 Section-7_Platform_lev15 --fig-dir <document>/media
'''
import argparse
import glob
import logging
import os
import sys
import warnings
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import radcalnet_oc as radoc
from radcalnet_oc.coxmunk import GLINT_FLAGS
from radcalnet_oc.uncertainty import uncertainty_budget, INPUT_UNC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 's2_toa_matchup'))
sys.path.insert(0, HERE)
from matchup_analysis import load, selection, screen  # noqa: E402
from make_products import simulate_month  # noqa: E402
from s2_toa_matchup import aeronet_inputs  # noqa: E402
from radcalnet_oc.reconstruction import Reconstruction  # noqa: E402

ROOT = '/data/radcalnet'
BANDS = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]   # MSI bands of the comparison (no 1375)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})


def products(site, level, version):
    files = sorted(glob.glob(f'{ROOT}/products/v{version}/daily/{site}/*/*/*/{site}_radcalnet-oc_{level}_*.nc'))
    keep = ['Rtoa', 'Rtoa_glint', 'glint_flags', 'glint_fraction', 'sza', 'aot550']
    out = []
    for f in files:
        with xr.open_dataset(f) as ds:
            out.append(ds[[v for v in keep if v in ds]].load())
    return xr.concat(out, 'time')


def fig_toa_envelope(db, site, ofile):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, var, title in zip(axs, ['Rtoa', 'Rtoa_glint'], ['without direct glint (Rtoa)', 'with direct glint (Rtoa_glint)']):
        R = db[var]
        q = R.quantile([0.05, 0.25, 0.5, 0.75, 0.95], 'time')
        wl = R.wl
        ax.fill_between(wl, q.sel(quantile=0.05), q.sel(quantile=0.95), color='0.8', label='5-95 %')
        ax.fill_between(wl, q.sel(quantile=0.25), q.sel(quantile=0.75), color='tab:purple', alpha=0.5, label='25-75 %')
        ax.plot(wl, q.sel(quantile=0.5), 'k', lw=1.2, label='median')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 0.4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_title(f'{site}: {title}', fontsize=10)
    axs[0].set_ylabel('TOA reflectance (nadir)')
    axs[0].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)


def fig_rrs(site, level, version, ofile):
    '''Distribution of the nadir remote-sensing reflectance of the products (invRrs reconstruction, 10 nm bands)
    over 400-1000 nm, and of the chlorophyll index of the reconstruction.'''
    files = sorted(glob.glob(f'{ROOT}/products/v{version}/daily/{site}/*/*/*/{site}_radcalnet-oc_{level}_*.nc'))
    out = []
    for f in files:
        with xr.open_dataset(f) as ds:
            out.append(ds[['Rrs', 'chl_invrrs']].load())
    db = xr.concat(out, 'time')
    R = db.Rrs.sel(wl=slice(400, 1000))
    q = R.quantile([0.05, 0.25, 0.5, 0.75, 0.95], 'time')
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [2, 1]})
    ax = axs[0]
    ax.fill_between(R.wl, q.sel(quantile=0.05), q.sel(quantile=0.95), color='0.8', label='5-95 %')
    ax.fill_between(R.wl, q.sel(quantile=0.25), q.sel(quantile=0.75), color='tab:cyan', alpha=0.6, label='25-75 %')
    ax.plot(R.wl, q.sel(quantile=0.5), 'k', lw=1.2, label='median')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('$R_{rs}$ at nadir (sr$^{-1}$)')
    t = pd.to_datetime(db.time.values)
    ax.set_title(f'{site}: nadir $R_{{rs}}$ of the products ({len(t)} times, {t.min():%Y-%m} to {t.max():%Y-%m}, '
                 f'{level})', fontsize=10)
    ax.legend(fontsize=8)
    chl = db.chl_invrrs.values
    chl = chl[np.isfinite(chl) & (chl > 0)]
    axs[1].hist(np.log10(chl), bins=40, color='tab:green', alpha=0.8)
    axs[1].set_xlabel('log$_{10}$ Chl (invRrs, mg m$^{-3}$)')
    axs[1].set_ylabel('Times')
    axs[1].set_title(f'median {np.median(chl):.1f} mg m$^{{-3}}$', fontsize=10)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    return q.sel(wl=[440, 490, 560, 670, 860], method='nearest').to_pandas(), float(np.median(chl))


def fig_glint_nadir(db, site, ofile):
    f = db.glint_flags.astype(int)
    month = db.time.dt.month
    df = pd.DataFrame({'month': month.values,
                       'low': (f & GLINT_FLAGS['glint_low']) > 0,
                       'moderate': (f & GLINT_FLAGS['glint_moderate']) > 0,
                       'high': (f & GLINT_FLAGS['glint_high']) > 0,
                       'near_specular': (f & GLINT_FLAGS['glint_near_specular']) > 0})
    df['removed'] = df.moderate | df.near_specular
    g = 100 * df.groupby('month').mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for col, c in [('low', 'tab:green'), ('moderate', 'tab:orange'), ('high', 'tab:red'),
                   ('near_specular', 'tab:purple'), ('removed', 'k')]:
        ax.plot(g.index, g[col], marker='o', color=c, lw=2 if col == 'removed' else 1,
                label=col if col != 'removed' else 'removed by the filter (moderate or near specular)')
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Month')
    ax.set_ylabel('Times flagged (%)')
    ax.set_title(f'{site}: direct sunglint at nadir view ({len(df)} times)', fontsize=10)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    return g


def budget(aeronet, level, year):
    '''Uncertainty budget at nadir for one measurement per month (closest to 10:00 UTC, near the overpasses).'''
    inputs = aeronet_inputs(aeronet, level).sel(time=str(year))
    inputs = inputs.where(inputs.aot550.notnull() & inputs.naot.notnull().any('wl_aeronet'), drop=True)
    t = pd.to_datetime(inputs.time.values)
    pick = pd.Series(np.abs((t.hour + t.minute / 60) - 10), index=t).groupby(t.month).idxmin().values
    records = inputs.sel(time=pick)
    # the simulation of make_products (invRrs reconstruction, aerosol mixture, nadir, direct glint)
    import make_products as mp
    captured = {}
    orig = radoc.Process.execute

    def execute(self, *a, **k):
        r = orig(self, *a, **k)
        captured['process'] = self
        return r
    radoc.Process.execute = execute
    try:
        mp.simulate_month(records, Reconstruction(), 'cm_iso_mix')
    finally:
        radoc.Process.execute = orig
    process = captured['process']
    return uncertainty_budget(process)


def fig_budget(db, site, ofile, csv):
    spectral = radoc.Spectral(np.arange(400, 2500, 10), 10.)
    names = [v for v in db.data_vars if v.startswith('u_') and v not in ('u_total_rel',)]
    conv = xr.merge([spectral.convolve2(db[n].rename(n), expon=3)[n] for n in names + ['Rtoa_glint']])
    med = conv.median('time')
    rel = {n: 100 * (conv[n] / conv.Rtoa_glint).median('time') for n in names}
    labels = {'u_aot550': 'AOT(550) $\\pm$0.01', 'u_tcwv': 'water vapour $\\pm$10 %', 'u_tco3': 'ozone $\\pm$3 %',
              'u_pressure': 'pressure $\\pm$2 hPa', 'u_wind': 'wind $\\pm$1.5 m/s (sky reflection)',
              'u_Rrs': '$R_{rs}$ $\\pm$5 %', 'u_glint': 'direct glint model',
              'u_model': 'model gap (Sentinel-2 matchups)', 'u_total': 'total'}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in names:
        kw = dict(color='k', lw=2) if n == 'u_total' else dict(lw=1)
        axs[0].plot(med.wl, med[n], label=labels.get(n, n), **kw)
        axs[1].plot(med.wl, rel[n], label=labels.get(n, n), **kw)
    axs[0].set_yscale('log')
    axs[0].set_ylim(1e-6, 1e-2)
    axs[0].set_ylabel('Standard uncertainty of $R^{TOA}$')
    axs[1].set_yscale('log')
    axs[1].set_ylim(0.01, 100)
    axs[1].set_ylabel('Relative standard uncertainty (%)')
    for ax in axs:
        ax.set_xlabel('Wavelength (nm)')
    axs[1].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{site}: uncertainty budget of the simulated TOA reflectance at nadir '
                 f'(median of {db.sizes["time"]} measurements, one per month)', fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    tab = pd.DataFrame({n: rel[n].sel(wl=[440, 490, 560, 660, 710, 740, 780, 840, 860, 1610, 2190],
                                       method='nearest').values for n in names},
                       index=[443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]).round(2)
    tab.to_csv(csv)
    return tab


#: scene selections compared in the statements (sunglint flags of the matchups, radcalnet_oc.GLINT_FLAGS)
SELECTIONS = {'angle30': dict(glint_min=30., flags_exclude=None, label='glint angle > 30$^\\circ$'),
              'nohigh': dict(glint_min=0., flags_exclude=['glint_high', 'glint_near_specular'],
                             label='no high or near-specular glint'),
              'noglint': dict(glint_min=0., flags_exclude=['glint_moderate', 'glint_high', 'glint_near_specular'],
                              label='no moderate, high or near-specular glint')}


def matchups(aeronet_site_levels, sel='noglint', roi=True):
    '''Matchups of the site kept by the standard selection and the sunglint screening ``sel`` (SELECTIONS), with
    the satellite statistics in the recommended ROI (``roi``); several levels are concatenated, a scene of the
    first level being kept when it is in several.'''
    s = SELECTIONS[sel]
    args = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=s['glint_min'],
                           flags_exclude=s['flags_exclude'], glint_stats='cm_iso_mix', glint_pixels=False, roi=roi)
    out, seen = [], set()
    for sl in aeronet_site_levels:
        ds = load(f'{ROOT}/matchups/sentinel2/v{radoc.__version__}', sl, args)
        keep = (selection(ds, args) & screen(ds, args)).values
        keep &= ~np.isin(ds['product'].values.astype(str), list(seen))
        ds = ds.isel(time=keep)
        ds['level'] = ('time', np.full(ds.sizes['time'], sl.rsplit('_', 1)[1]))
        seen |= set(ds['product'].values.astype(str))
        out.append(ds[[v for v in ('sat_median', 'sim_median', 'simg_median', 'glint_median', 'glint_uncertain',
                                   'glint_flags', 'glint_fraction', 'glint_angle', 'aot550', 'product', 'level')
                       if v in ds]])
    ds = xr.concat(out, 'time').sortby('time')
    ds.attrs.update(selection=s['label'], roi=roi)
    return ds


def rel_stats(ds, var='simg_median', mask_uncertain=False):
    '''Median, quartiles, robust SD (%) and median absolute difference (1e-3) per band of sim / MSI - 1.'''
    obs, sim = ds.sat_median.sel(wl=BANDS), ds[var].sel(wl=BANDS)
    rel = 100 * (sim / obs - 1)
    absd = 1e3 * (sim - obs)
    if mask_uncertain:
        unc = ds.glint_uncertain.sel(wl=BANDS).astype(bool)
        rel, absd = rel.where(~unc), absd.where(~unc)
    q = rel.quantile([0.25, 0.5, 0.75], 'time')
    med = q.sel(quantile=0.5, drop=True)
    return pd.DataFrame({'median_%': med.values, 'q25_%': q.sel(quantile=0.25).values,
                         'q75_%': q.sel(quantile=0.75).values,
                         'rsd_%': (1.4826 * np.abs(rel - med).median('time')).values,
                         'abs_median_1e3': absd.median('time').values, 'N': rel.count('time').values}, index=BANDS)


def fig_matchup_flags(site, levels, ofile, csv):
    '''Effect of the sunglint flags and of the ROI on the matchup statistics (simulation with direct glint).'''
    cases = [('box 1.2 km, glint angle > 30$^\\circ$ (previous screening)', 'angle30', False, False, 'tab:gray'),
             ('ROI, glint angle > 30$^\\circ$', 'angle30', True, False, 'tab:blue'),
             ('ROI, no high/near-specular glint, glint_uncertain bands masked', 'nohigh', True, True, 'tab:green'),
             ('ROI, no moderate/high/near-specular glint', 'noglint', True, False, 'tab:red'),
             ('ROI, no moderate+ glint, glint_uncertain bands masked', 'noglint', True, True, 'k')]
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    rows = []
    for i, (label, sel, roi, mask, c) in enumerate(cases):
        ds = matchups(levels, sel, roi)
        st = rel_stats(ds, mask_uncertain=mask)
        x = np.arange(len(BANDS)) + 0.11 * (i - 2)
        axs[0].errorbar(x, st['median_%'], yerr=[st['median_%'] - st['q25_%'], st['q75_%'] - st['median_%']],
                        fmt='o', ms=4, color=c, capsize=2, label=f'{label} (N = {ds.sizes["time"]})')
        axs[1].plot(x, st['rsd_%'], 'o', ms=5, color=c)
        rows.append(st.assign(case=label))
    axs[0].axhline(0, color='k', lw=0.8)
    axs[0].set_ylim(-60, 25)
    axs[0].set_ylabel('Simulation / MSI - 1 (%)\nmedian and IQR')
    axs[0].legend(fontsize=7, loc='lower left')
    axs[1].set_ylabel('Robust SD (%)')
    axs[1].set_yscale('log')
    axs[1].set_xticks(range(len(BANDS)))
    axs[1].set_xticklabels(BANDS)
    axs[1].set_xlabel('MSI band (nm)')
    axs[0].set_title(f'{site}: sunglint flags and region of interest of the Sentinel-2 matchups', fontsize=10)
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    tab = pd.concat(rows).round(2)
    tab.index.name = 'band'
    tab.to_csv(csv)
    return tab


def fig_matchup_bias(ds, site, ofile, csv):
    rows = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    for (var, mask), label, c, dx in [(('sim_median', False), 'simulation without direct glint', 'tab:blue', -0.16),
                                      (('simg_median', False), 'simulation with direct glint', 'tab:red', 0.),
                                      (('simg_median', True), 'with direct glint, glint_uncertain bands masked',
                                       'k', 0.16)]:
        st = rel_stats(ds, var, mask)
        x = np.arange(len(BANDS)) + dx
        ax.errorbar(x, st['median_%'], yerr=[st['median_%'] - st['q25_%'], st['q75_%'] - st['median_%']],
                    fmt='o', ms=4, color=c, capsize=3, label=label)
        key = var + ('_certain' if mask else '')
        for k in ('median_%', 'q25_%', 'q75_%', 'rsd_%', 'abs_median_1e3', 'N'):
            rows[f'{key}_{k}'] = st[k].values
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS)
    ax.set_xlabel('MSI band (nm)')
    ax.set_ylabel('Simulation / MSI - 1 (%)')
    ax.set_ylim(-60, 20)
    ax.set_title(f'{site}: Sentinel-2 MSI L1C matchups ({ds.sizes["time"]} scenes, {ds.attrs["selection"]})',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    tab = pd.DataFrame(rows, index=BANDS).round(2)
    tab.to_csv(csv)
    return tab


STYLE = {'font.family': 'serif', 'mathtext.fontset': 'stix',
         'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']}


def fig_matchup_bands(ds, site, ofile):
    '''Simulation against MSI per band, square panels with the 1:1 line and the least-squares regression line
    (slope, intercept, r, median relative bias, N), points coloured by AOT(550).'''
    with plt.rc_context({**STYLE, 'font.size': 12, 'axes.titlesize': 12}):
        fig, axs = plt.subplots(3, 4, figsize=(12.5, 9.4), gridspec_kw={'hspace': 0.3, 'wspace': 0.32})
        for ax, b in zip(axs.ravel(), BANDS):
            x, y = ds.sat_median.sel(wl=b).values, ds.simg_median.sel(wl=b).values
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, aot = x[ok], y[ok], ds.aot550.values[ok]
            order = np.argsort(aot)                      # high AOT drawn on top
            sc = ax.scatter(x[order], y[order], c=aot[order], s=12, cmap='Spectral_r', vmin=0, vmax=0.3,
                            edgecolor='k', linewidth=0.15, alpha=0.9, zorder=2)
            lo, hi = np.nanpercentile(np.r_[x, y], [0.5, 99.5])
            pad = 0.08 * (hi - lo)
            lim = [max(0, lo - pad), hi + pad]          # same range on both axes: square panel, 1:1 diagonal
            ax.plot(lim, lim, color='0.4', lw=0.9, ls='--', zorder=1, label='1:1')
            a, c = np.polyfit(x, y, 1)
            ax.plot(lim, np.polyval([a, c], lim), color='k', lw=1.3, zorder=3, label='regression')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_box_aspect(1)
            ax.locator_params(nbins=4)
            ax.tick_params(labelsize=9)
            rel = 100 * np.median(y / x - 1)
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f'{b} nm', fontsize=13, pad=3)
            fit = f'y = {a:.3f}x {c:+.4f}'.replace('-', '\u2212')
            ax.text(0.04, 0.96, f'{fit}\nr = {r:.3f}\nbias {rel:+.1f} %\nN = {ok.sum()}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
                    bbox=dict(fc='w', ec='none', alpha=0.75, pad=1.5))
        axs[0, 0].legend(loc='lower right', fontsize=9, frameon=False, handlelength=1.6)
        fig.supxlabel('MSI L1C TOA reflectance (ROI median)', fontsize=15, y=0.035)
        fig.supylabel('RadCalNet-OC TOA reflectance with direct glint', fontsize=15, x=0.06)
        fig.colorbar(sc, ax=axs, shrink=0.6, pad=0.02, label='AOT(550)')
        fig.suptitle(f'{site}: Sentinel-2 matchups per band ({ds.sizes["time"]} scenes)', fontsize=14, y=0.95)
        fig.savefig(ofile, dpi=200, bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='VEIT', help='site name of the products (RadCalNet code)')
    p.add_argument('--aeronet', default='AAOT', help='AERONET-OC site')
    p.add_argument('--level', default='lev20')
    p.add_argument('--year', type=int, default=2020, help='year of the measurements of the uncertainty budget')
    p.add_argument('--matchups', nargs='+', default=None,
                   help='matchup files <aeronet site>_<level> (default <aeronet>_<level>; several: concatenated, '
                        'first one preferred)')
    p.add_argument('--selection', default='noglint', choices=list(SELECTIONS), help='sunglint screening')
    p.add_argument('--no-roi', action='store_true', help='box of 1.2 km instead of the recommended ROI')
    p.add_argument('--fig-dir', required=True)
    p.add_argument('--only', nargs='*', default=['envelope', 'budget', 'matchups', 'flags'])
    args = p.parse_args()
    radoc.set_logging('WARNING')
    warnings.filterwarnings('ignore')
    os.makedirs(args.fig_dir, exist_ok=True)
    out = lambda name: os.path.join(args.fig_dir, name)   # noqa: E731
    if 'envelope' in args.only:
        db = products(args.site, args.level, radoc.__version__)
        fig_toa_envelope(db, args.site, out('toa_envelope.png'))
        g = fig_glint_nadir(db, args.site, out('glint_nadir.png'))
        print(f'products: {db.sizes["time"]} times; flagged at nadir per month (%):\n{g.round(1).to_string()}')
    if 'budget' in args.only:
        cache = (f'{ROOT}/studies/uncertainty_budget_v{radoc.__version__}/'
                 f'{args.site}_{args.level}_{args.year}.nc')   # the budget takes about 15 min
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        if os.path.exists(cache):
            db = xr.open_dataset(cache).load()
        else:
            db = budget(args.aeronet, args.level, args.year)
            keep = [v for v in db.data_vars if v.startswith('u_') or v in ('Rtoa', 'Rtoa_glint', 'sza', 'aot550', 'wind')]
            db[keep].to_netcdf(cache)
        tab = fig_budget(db, args.site, out('uncertainty_budget.png'), out('uncertainty_budget.csv'))
        print(f'uncertainty budget, median relative (%):\n{tab.to_string()}')
    if 'rrs' in args.only:
        q, chl = fig_rrs(args.site, args.level, radoc.__version__, out('rrs_products.png'))
        print(f'Rrs quantiles (sr-1):\n{q.to_string()}\nmedian chl_invrrs {chl:.2f} mg m-3')
    levels = args.matchups or [f'{args.aeronet}_{args.level}']
    if 'matchups' in args.only:
        ds = matchups(levels, args.selection, not args.no_roi)
        tab = fig_matchup_bias(ds, args.site, out('matchup_bias.png'), out('matchup_bias.csv'))
        fig_matchup_bands(ds, args.site, out('matchup_bands.png'))
        print(f'matchups ({ds.sizes["time"]} scenes, {ds.attrs["selection"]}, levels '
              f'{pd.Series(ds.level.values).value_counts().to_dict()}):\n{tab.to_string()}')
    if 'flags' in args.only:
        tab = fig_matchup_flags(args.site, levels, out('matchup_flags.png'), out('matchup_flags.csv'))
        print(f'flags and ROI:\n{tab.to_string()}')


if __name__ == '__main__':
    main()
