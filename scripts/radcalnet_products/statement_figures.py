'''Figures of the RadCalNet site uncertainty statement of a water site (e.g. QA4EO-WGCV-RadCalNet-VEIT-U), from the
RadCalNet-OC products, the uncertainty budget of radcalnet_oc and the Sentinel-2 matchups.

Figures (PNG, ``--fig-dir``):

- ``toa_envelope.png``: temporal distribution of the TOA reflectance of the daily products at nadir (median,
  quartiles, 5-95 %), with and without the direct glint;
- ``glint_nadir.png``: fraction of the times of the products flagged for sunglint at nadir, per month;
- ``uncertainty_budget.png``: terms of the uncertainty budget (``radcalnet_oc.uncertainty.uncertainty_budget``) for
  one measurement per month of a year of the site, at nadir: median absolute and relative uncertainties;
- ``matchup_bias.png``: median and interquartile range of the relative difference simulation - MSI per band,
  with and without the simulated glint;
- ``matchup_bands.png``: simulation (with glint) against MSI per band.

Usage:
    python statement_figures.py --site VEIT --aeronet AAOT --level lev20 --fig-dir <document>/media [--year 2020]
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
              'u_Rrs': '$R_{rs}$ $\\pm$5 %', 'u_glint': 'direct glint model', 'u_total': 'total'}
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


def matchups(aeronet_site_level):
    args = SimpleNamespace(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=30., flags_exclude=None,
                           glint_stats='cm_iso_mix', glint_pixels=False)
    ds = load(f'{ROOT}/matchups/sentinel2/v{radoc.__version__}', aeronet_site_level, args)
    keep = (selection(ds, args) & screen(ds, args)).values
    return ds.isel(time=keep)


def fig_matchup_bias(ds, site, ofile, csv):
    obs = ds.sat_median.sel(wl=BANDS)
    rows = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    for var, label, c in [('sim_median', 'simulation without direct glint', 'tab:blue'),
                          ('simg_median', 'simulation with direct glint', 'tab:red')]:
        rel = 100 * (ds[var].sel(wl=BANDS) / obs - 1)
        q = rel.quantile([0.25, 0.5, 0.75], 'time')
        x = np.arange(len(BANDS)) + (-0.12 if var == 'sim_median' else 0.12)
        ax.errorbar(x, q.sel(quantile=0.5), yerr=[q.sel(quantile=0.5) - q.sel(quantile=0.25),
                                                  q.sel(quantile=0.75) - q.sel(quantile=0.5)],
                    fmt='o', color=c, capsize=3, label=label)
        rows[var + '_median_%'] = q.sel(quantile=0.5).values
        rows[var + '_q25_%'] = q.sel(quantile=0.25).values
        rows[var + '_q75_%'] = q.sel(quantile=0.75).values
        rows[var + '_abs_median_1e3'] = (1e3 * (ds[var].sel(wl=BANDS) - obs)).median('time').values
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BANDS)
    ax.set_xlabel('MSI band (nm)')
    ax.set_ylabel('Simulation / MSI - 1 (%)')
    ax.set_ylim(-60, 20)
    ax.set_title(f'{site}: Sentinel-2 MSI L1C matchups ({ds.sizes["time"]} scenes, glint angle > 30$^\\circ$)',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(ofile, dpi=200)
    plt.close(fig)
    tab = pd.DataFrame(rows, index=BANDS).round(2)
    tab['N'] = ds.sizes['time']
    tab.to_csv(csv)
    return tab


def fig_matchup_bands(ds, site, ofile):
    fig, axs = plt.subplots(3, 4, figsize=(12, 10), gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
    for ax, b in zip(axs.ravel(), BANDS):
        x, y = ds.sat_median.sel(wl=b).values, ds.simg_median.sel(wl=b).values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        sc = ax.scatter(x, y, c=ds.aot550.values[ok], s=6, cmap='viridis', vmin=0, vmax=0.3)
        lim = [0, np.nanpercentile(np.r_[x, y], 99) * 1.1]
        ax.plot(lim, lim, 'k', lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        rel = 100 * np.median(y / x - 1)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'{b} nm\nbias {rel:+.1f} %, r = {r:.2f}', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.supxlabel('MSI L1C TOA reflectance (box median)')
    fig.supylabel('RadCalNet-OC TOA reflectance with direct glint')
    fig.colorbar(sc, ax=axs, shrink=0.6, label='AOT(550)')
    fig.suptitle(f'{site}: Sentinel-2 matchups per band ({ds.sizes["time"]} scenes)', fontsize=10)
    fig.savefig(ofile, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', default='VEIT', help='site name of the products (RadCalNet code)')
    p.add_argument('--aeronet', default='AAOT', help='AERONET-OC site')
    p.add_argument('--level', default='lev20')
    p.add_argument('--year', type=int, default=2020, help='year of the measurements of the uncertainty budget')
    p.add_argument('--fig-dir', required=True)
    p.add_argument('--only', nargs='*', default=['envelope', 'budget', 'matchups'])
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
    if 'matchups' in args.only:
        ds = matchups(f'{args.aeronet}_{args.level}')
        tab = fig_matchup_bias(ds, args.site, out('matchup_bias.png'), out('matchup_bias.csv'))
        fig_matchup_bands(ds, args.site, out('matchup_bands.png'))
        print(f'matchups:\n{tab.to_string()}')


if __name__ == '__main__':
    main()
