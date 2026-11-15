'''
Analysis of the Sentinel-2 / RadCalNet-OC TOA matchups produced by ``s2_toa_matchup.py``.

Three commands, on the gathered files ``<odir>/<site>_<level>_s2_toa_matchup.nc``:

- ``stats``: per-band median relative difference (sim - sat) / sat, median absolute percentage difference
  and median absolute difference, (i) without simulated glint for glint angles above ``--glint-min``,
  (ii) with and (iii) without the simulated direct glint for all glint angles; per sensor unit (S2A, S2B,
  S2C);
- ``drivers``: Spearman correlation of the differences with the geometry, the atmosphere and the water,
  and the differences per bin of glint angle and of aerosol optical thickness;
- ``figures``: spectral bias per site (``s2_bias_spectrum.pdf``) and difference versus glint angle with
  a scatter plot (``s2_glint_scatter.pdf``).

Selection of the matchups: valid fraction of the box > ``--frac-min``, at least ``--npix-min`` valid pixels,
nearest AERONET-OC measurement within ``--dt-max`` minutes, AOT(550) < ``--aot-max``.

Examples::

    python matchup_analysis.py stats
    python matchup_analysis.py drivers --sites AAOT_lev20 Galata_Platform_lev20
    python matchup_analysis.py figures --fig-dir figures
'''
import argparse
import glob
import os

import numpy as np
import pandas as pd
import xarray as xr

opj = os.path.join

ODIR = '/data/radcalnet/matchup/sentinel2'
#: large variables of the matchup files, not needed here
DROP = ['rec_Rtoa_spectrum', 'rec_Rglint_spectrum', 'sim_spectrum_median', 'glint_spectrum_median']
#: sites of the figures, with their labels and colours
FIG_SITES = [('AAOT_lev20', 'AAOT L2.0', '#1b6ca8'), ('Galata_Platform_lev20', 'Galata L2.0', '#d1495b'),
             ('AAOT_lev15', 'AAOT L1.5', '#66a182'), ('Section-7_Platform_lev15', 'Section-7 L1.5', '#edae49')]
DRIVERS = ['glint_angle', 'aot550', 'ang_exp', 'wind', 'sza', 'vza', 'tcwv', 'spm_norm', 'chl', 'sim_count']


def load(odir, site):
    '''Matchup file of ``site`` (``<site>_<level>``), without the spectra.'''
    return xr.open_dataset(opj(odir, f'{site}_s2_toa_matchup.nc'), drop_variables=DROP).load()


def selection(ds, args):
    '''Boolean (time) of the matchups kept, before the glint-angle criterion.'''
    return ((ds.frac_valid > args.frac_min) & (ds.n_pix >= args.npix_min) &
            (np.abs(ds.dt_nearest) <= args.dt_max) & (ds.aot550 < args.aot_max))


def band_stats(sim, obs):
    '''Per-band statistics of the differences, DataFrame (statistic, band).'''
    rel = 100 * (sim - obs) / obs
    return pd.DataFrame({'N': np.isfinite(rel).sum(0),
                         'bias_med%': np.nanmedian(rel, 0),
                         'MAPD%': np.nanmedian(np.abs(rel), 0),
                         'abs_med_1e3': 1e3 * np.nanmedian(sim - obs, 0),
                         'obs_med': np.nanmedian(obs, 0)}, index=sim.columns).T


def sites_of(args):
    if args.sites:
        return args.sites
    files = sorted(glob.glob(opj(args.odir, '*_s2_toa_matchup.nc')))
    return [os.path.basename(f).replace('_s2_toa_matchup.nc', '') for f in files]


def cmd_stats(args):
    for site in sites_of(args):
        ds = load(args.odir, site)
        base = selection(ds, args)
        far = base & (ds.glint_angle > args.glint_min)
        print(f'===== {site}: {ds.sizes["time"]} scenes, {ds.sizes["rec"]} simulations')
        for name, m, sim in [(f'without glint, glint angle > {args.glint_min:g} deg', far, 'sim_median'),
                             ('all glint angles, with simulated glint', base, 'simg_median'),
                             ('all glint angles, without simulated glint', base, 'sim_median')]:
            m = m.values
            if m.sum() == 0:
                print(f'-- {name}: no matchup')
                continue
            print(f'-- {name}')
            print(band_stats(ds[sim].isel(time=m).to_pandas(), ds.sat_median.isel(time=m).to_pandas()).round(2))
        unit = ds['product'].astype(str).str.slice(0, 3).values
        for u in ['S2A', 'S2B', 'S2C']:
            m = far.values & (unit == u)
            if m.sum() < 5:
                continue
            s = band_stats(ds.sim_median.isel(time=m).to_pandas(), ds.sat_median.isel(time=m).to_pandas())
            print(f'{u} (N={m.sum()}) bias_med%:', s.loc['bias_med%'].round(1).to_dict())
        print('quantiles 5/50/95: glint angle', np.nanpercentile(ds.glint_angle, [5, 50, 95]).round(1),
              'vza', np.nanpercentile(ds.vza, [5, 50, 95]).round(1),
              'sza', np.nanpercentile(ds.sza, [5, 50, 95]).round(1),
              'aot550', np.nanpercentile(ds.aot550, [5, 50, 95]).round(3),
              'wind', np.nanpercentile(ds.wind, [5, 50, 95]).round(1))


def cmd_drivers(args):
    from scipy import stats
    for site in sites_of(args):
        ds = load(args.odir, site)
        m = (selection(ds, args) & (ds.glint_angle > args.glint_min)).values
        d = (ds.sim_median - ds.sat_median).isel(time=m)
        df = ds[DRIVERS].isel(time=m).to_dataframe().drop(columns=['spatial_ref'], errors='ignore')
        df['year'] = df.index.year
        print(f'===== {site}: {m.sum()} matchups, glint angle > {args.glint_min:g} deg')
        for wl in args.bands:
            y = d.sel(wl=wl).values * 1e3
            rho = {c: round(stats.spearmanr(df[c], y, nan_policy='omit')[0], 2) for c in df.columns}
            print(f'{wl} nm: sim - sat (x1e3) median {np.nanmedian(y):.2f}, IQR',
                  np.nanpercentile(y, [25, 75]).round(2), 'Spearman', rho)
        diff = d.sel(wl=args.bands).to_pandas() * 1e3
        print('median sim - sat (x1e3) per glint angle bin (deg)')
        print(diff.groupby(pd.cut(df.glint_angle, [0, 40, 50, 60, 90]).values, observed=True).median().round(2))
        print('median sim - sat (x1e3) per AOT(550) bin')
        print(diff.groupby(pd.cut(df.aot550, [0, 0.08, 0.15, 0.3]).values, observed=True).median().round(2))
        print('1375 nm: sat median', float(ds.sat_median.sel(wl=1375).isel(time=m).median()),
              'sim median', float(ds.sim_median.sel(wl=1375).isel(time=m).median()))


def cmd_figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9})
    os.makedirs(args.fig_dir, exist_ok=True)

    # spectral bias per site
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for site, label, c in FIG_SITES:
        if not os.path.exists(opj(args.odir, f'{site}_s2_toa_matchup.nc')):
            continue
        ds = load(args.odir, site)
        m = (selection(ds, args) & (ds.glint_angle > args.glint_min)).values
        wl = ds.wl.values
        keep = wl != 1375  # B10: simulated ~0, relative difference -100 %
        d = (ds.sim_median - ds.sat_median).values[m][:, keep]
        rel = 100 * d / ds.sat_median.values[m][:, keep]
        ax[0].plot(wl[keep], np.nanmedian(rel, 0), 'o-', c=c, ms=3, label=f'{label} (N={m.sum()})')
        ax[1].plot(wl[keep], 1e3 * np.nanmedian(d, 0), 'o-', c=c, ms=3)
        ax[1].fill_between(wl[keep], 1e3 * np.nanpercentile(d, 25, 0), 1e3 * np.nanpercentile(d, 75, 0),
                           color=c, alpha=.12, lw=0)
    for a in ax:
        a.axhline(0, c='k', lw=.6)
        a.set_xticks([443, 665, 865, 1610, 2190])
        a.set_xlabel('S2 band (nm)')
    ax[0].set_ylabel('median (sim - sat)/sat (%)')
    ax[1].set_ylabel(r'median sim - sat ($\times10^{-3}$)')
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(opj(args.fig_dir, 's2_bias_spectrum.pdf'))

    # difference versus glint angle, and scatter plot, for one site
    site, label = args.glint_site, args.glint_site.replace('_lev20', ' L2.0').replace('_lev15', ' L1.5')
    ds = load(args.odir, site)
    base = selection(ds, args).values
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for wl, c in [(865, '#d1495b'), (2190, '#1b6ca8')]:
        ax[0].scatter(ds.glint_angle.values[base], 1e3 * (ds.sim_median - ds.sat_median).sel(wl=wl).values[base],
                      s=5, c=c, alpha=.5, label=f'{wl} nm, no glint')
        ax[0].scatter(ds.glint_angle.values[base], 1e3 * (ds.simg_median - ds.sat_median).sel(wl=wl).values[base],
                      s=5, marker='x', c=c, alpha=.5, lw=.6, label=f'{wl} nm, + Cox-Munk glint')
    ax[0].set_ylim(-15, 10)
    ax[0].axhline(0, c='k', lw=.6)
    ax[0].set_xlabel('glint angle (deg)')
    ax[0].set_ylabel(r'sim - sat ($\times10^{-3}$)')
    ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title(label, fontsize=9)
    far = (selection(ds, args) & (ds.glint_angle > args.glint_min)).values
    for wl, c in [(865, '#d1495b'), (443, '#1b6ca8'), (2190, '#66a182')]:
        ax[1].scatter(ds.sat_median.sel(wl=wl).values[far], ds.sim_median.sel(wl=wl).values[far], s=5, c=c,
                      label=f'{wl} nm')
    lim = [1e-3, 0.3]
    ax[1].plot(lim, lim, 'k', lw=.6)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel(r'S2 L1C $R_{toa}$')
    ax[1].set_ylabel(r'radcalnet-oc $R_{toa}$')
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title(f'{label}, glint angle > {args.glint_min:g} deg', fontsize=9)
    fig.tight_layout()
    fig.savefig(opj(args.fig_dir, 's2_glint_scatter.pdf'))
    print(f'figures -> {args.fig_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('command', choices=['stats', 'drivers', 'figures'])
    p.add_argument('--odir', default=ODIR, help='folder of the gathered matchup files')
    p.add_argument('--sites', nargs='+', default=None,
                   help='<site>_<level> (default: all the matchup files of --odir)')
    p.add_argument('--frac-min', type=float, default=0.5)
    p.add_argument('--npix-min', type=int, default=50)
    p.add_argument('--dt-max', type=float, default=60., help='nearest measurement to the overpass (min)')
    p.add_argument('--aot-max', type=float, default=0.3)
    p.add_argument('--glint-min', type=float, default=30., help='minimum glint angle (deg)')
    p.add_argument('--bands', type=int, nargs='+', default=[443, 665, 865, 1610, 2190], help='drivers: bands (nm)')
    p.add_argument('--fig-dir', default='.', help='figures: output folder')
    p.add_argument('--glint-site', default='AAOT_lev20', help='figures: site of the glint-angle figure')
    args = p.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    {'stats': cmd_stats, 'drivers': cmd_drivers, 'figures': cmd_figures}[args.command](args)


if __name__ == '__main__':
    main()
