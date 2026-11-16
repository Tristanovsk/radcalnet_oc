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
  a scatter plot (``s2_glint_scatter.pdf``);
- ``glint``: matchups per sunglint level (``glint_flags`` of the gathered files: none, low, moderate, high,
  near specular) with and without the simulated glint, and efficiency of screenings by glint angle and by flags
  (matchups kept, outliers removed, bias and dispersion of the kept ones); figure ``s2_glint_levels.pdf``.

Selection of the matchups: valid fraction of the box > ``--frac-min``, at least ``--npix-min`` valid pixels,
nearest AERONET-OC measurement within ``--dt-max`` minutes, AOT(550) < ``--aot-max``. With ``--glint-pixels``,
the satellite statistics without the SWIR threshold (``satg_*``) are used, and the scenes rejected by the threshold
are kept when homogeneous in the box (relative IQR at 2190 nm < 0.15: glint, not clouds). The glint screening is
either a glint angle > ``--glint-min`` or, with ``--flags-exclude``, the absence of the given flags.

Examples::

    python matchup_analysis.py stats
    python matchup_analysis.py drivers --sites AAOT_lev20 Galata_Platform_lev20
    python matchup_analysis.py figures --fig-dir figures
    python matchup_analysis.py glint --glint-stats cm_iso_mix --glint-pixels --fig-dir figures
    python matchup_analysis.py stats --glint-stats cm_iso_mix --flags-exclude glint_high glint_near_specular
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
#: sunglint flags of the gathered files (radcalnet_oc.GLINT_FLAGS)
FLAGS = {'glint_low': 1, 'glint_moderate': 2, 'glint_high': 4, 'glint_near_specular': 8}
IQR_GLINT = 0.15  # relative IQR at 2190 nm below which a scene rejected by the SWIR threshold is glint, not cloud


def suffix(args):
    return '' if args.glint_stats == 'cm_iso' else f'_{args.glint_stats}'


def load(odir, site, args=None):
    '''
    Matchup file of ``site`` (``<site>_<level>``), without the spectra. With ``args.glint_pixels``, ``sat_*`` are
    replaced by the statistics without the SWIR threshold (``satg_*``).
    '''
    sfx = '' if args is None else suffix(args)
    ds = xr.open_dataset(opj(odir, f'{site}_s2_toa_matchup{sfx}.nc'), drop_variables=DROP).load()
    if 'glint_flags' not in ds:  # files gathered before the flags: same computation as s2_toa_matchup.gather
        from radcalnet_oc.coxmunk import glint_flags
        ds['glint_fraction'] = (ds.glint_median / ds.simg_median).sel(wl=865, drop=True)
        ds['glint_flags'] = (('time',), glint_flags(ds.glint_fraction.values, ds.glint_angle.values))
    if args is not None and args.glint_pixels:
        iqr = ((ds.satg_q75 - ds.satg_q25) / ds.satg_median).sel(wl=2190, drop=True)
        ds['glint_scene'] = (ds.frac_valid_glint > args.frac_min) & (ds.n_pix_glint >= args.npix_min) & (iqr < IQR_GLINT)
        for st in ['median', 'q25', 'q75', 'mean', 'std']:
            ds[f'sat_{st}'] = ds[f'satg_{st}']
    return ds


def selection(ds, args):
    '''Boolean (time) of the matchups kept, before the glint screening.'''
    valid = (ds.frac_valid > args.frac_min) & (ds.n_pix >= args.npix_min)
    if 'glint_scene' in ds:
        valid = valid | ds.glint_scene
    return valid & (np.abs(ds.dt_nearest) <= args.dt_max) & (ds.aot550 < args.aot_max)


def screen(ds, args):
    '''Boolean (time) of the glint screening: glint angle > --glint-min, or none of the flags --flags-exclude.'''
    if args.flags_exclude:
        mask = np.uint8(sum(FLAGS[f] for f in args.flags_exclude))
        return (ds.glint_flags & mask) == 0
    return ds.glint_angle > args.glint_min


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
    files = sorted(glob.glob(opj(args.odir, f'*_s2_toa_matchup{suffix(args)}.nc')))
    return [os.path.basename(f).replace(f'_s2_toa_matchup{suffix(args)}.nc', '') for f in files]


def screening(args):
    return (f'without {"/".join(args.flags_exclude)}' if args.flags_exclude
            else f'glint angle > {args.glint_min:g} deg')


def cmd_stats(args):
    for site in sites_of(args):
        ds = load(args.odir, site, args)
        base = selection(ds, args)
        far = base & screen(ds, args)
        print(f'===== {site}: {ds.sizes["time"]} scenes, {ds.sizes["rec"]} simulations')
        for name, m, sim in [(f'without glint, {screening(args)}', far, 'sim_median'),
                             (f'with simulated glint, {screening(args)}', far, 'simg_median'),
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
        ds = load(args.odir, site, args)
        m = (selection(ds, args) & screen(ds, args)).values
        d = (ds.sim_median - ds.sat_median).isel(time=m)
        df = ds[DRIVERS].isel(time=m).to_dataframe().drop(columns=['spatial_ref'], errors='ignore')
        df['year'] = df.index.year
        print(f'===== {site}: {m.sum()} matchups, {screening(args)}')
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
    args.fig_dir = args.fig_dir or '.'
    os.makedirs(args.fig_dir, exist_ok=True)

    # spectral bias per site
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for site, label, c in FIG_SITES:
        if not os.path.exists(opj(args.odir, f'{site}_s2_toa_matchup{suffix(args)}.nc')):
            continue
        ds = load(args.odir, site, args)
        m = (selection(ds, args) & screen(ds, args)).values
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
    ds = load(args.odir, site, args)
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
    far = (selection(ds, args) & screen(ds, args)).values
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
    ax[1].set_title(f'{label}, {screening(args)}', fontsize=9)
    fig.tight_layout()
    fig.savefig(opj(args.fig_dir, 's2_glint_scatter.pdf'))
    print(f'figures -> {args.fig_dir}')


def glint_level(flags):
    """Sunglint level of each scene from the bitmask: none, low, moderate, high (+ near specular)."""
    flags = np.asarray(flags)
    level = np.select([flags & FLAGS['glint_high'] > 0, flags & FLAGS['glint_moderate'] > 0,
                       flags & FLAGS['glint_low'] > 0], ['high', 'moderate', 'low'], 'none')
    return np.where(flags & FLAGS['glint_near_specular'] > 0, np.char.add(level, ', near specular'), level)


LEVELS = ['none', 'low', 'low, near specular', 'moderate', 'moderate, near specular', 'high',
          'high, near specular']
#: candidate screenings: (label, glint-angle threshold or None, flags excluded)
SCREENINGS = [('none', None, []), ('glint angle > 30 deg', 30., []), ('glint angle > 40 deg', 40., []),
              ('exclude high', None, ['glint_high']),
              ('exclude near specular', None, ['glint_near_specular']),
              ('exclude high, near specular', None, ['glint_high', 'glint_near_specular']),
              ('exclude moderate (keep <= low)', None, ['glint_moderate']),
              ('exclude moderate, near specular', None, ['glint_moderate', 'glint_near_specular'])]


def cmd_glint(args):
    """Matchups per sunglint level and efficiency of the screenings (all the sites of --sites pooled)."""
    rows = []
    for site in sites_of(args):
        ds = load(args.odir, site, args)
        d = ds.isel(time=selection(ds, args).values)
        df = pd.DataFrame({'site': site, 'gflags': d.glint_flags.values.astype(int),
                           'glint_angle': d.glint_angle.values, 'glint_fraction': d.glint_fraction.values},
                          index=pd.DatetimeIndex(d.time.values))
        for wl in args.bands:
            obs = d.sat_median.sel(wl=wl).values
            for name, sim in [('', 'sim_median'), ('g', 'simg_median')]:
                df[f'rel{name}{wl}'] = 100 * (d[sim].sel(wl=wl).values - obs) / obs
                df[f'abs{name}{wl}'] = 1e3 * (d[sim].sel(wl=wl).values - obs)
        rows.append(df)
    df = pd.concat(rows)
    df['level'] = glint_level(df.gflags.values)
    print(f'===== {len(df)} matchups ({", ".join(sites_of(args))}), glint statistics {args.glint_stats}, '
          f'{"satellite without SWIR threshold (homogeneous glint scenes kept)" if args.glint_pixels else "clear pixels"}')
    print('-- number of matchups per site and sunglint level')
    print(pd.crosstab(df.site, df.level).reindex(columns=[l for l in LEVELS if l in set(df.level)]).to_string())

    print('-- median (sim - sat)/sat (%) per sunglint level, without (rel) and with (relg) the simulated glint')
    g = df.groupby('level')
    cols = [f'rel{wl}' for wl in args.bands] + [f'relg{wl}' for wl in args.bands]
    out = g[['glint_angle', 'glint_fraction'] + cols].median()
    out.insert(0, 'N', g.size())
    print(out.reindex([l for l in LEVELS if l in out.index]).round(2).to_string())
    print('-- median absolute percentage difference with the glint (MAPD %) and median sim - sat with the glint '
          '(x1e-3) per sunglint level')
    out = pd.concat([g[[f'relg{wl}' for wl in args.bands]].agg(lambda x: np.nanmedian(np.abs(x)))
                     .add_prefix('MAPD '), g[[f'absg{wl}' for wl in args.bands]].median()], axis=1)
    print(out.reindex([l for l in LEVELS if l in out.index]).round(2).to_string())

    # efficiency of the screenings: outliers of the simulation with the glint
    out_rel, out_abs = args.outlier_rel, args.outlier_abs
    outlier = (np.abs(df[f'relg{args.ref_band}']) > out_rel) | (np.abs(df[f'absg{args.swir_band}']) > out_abs)
    print(f'-- efficiency of the screenings; outlier: |sim+glint - sat| > {out_rel:g} % at {args.ref_band} nm or '
          f'> {out_abs:g}e-3 at {args.swir_band} nm ({outlier.sum()} of {len(df)} matchups)')
    res = {}
    for label, angle, flags in SCREENINGS:
        if angle is not None:
            keep = df.glint_angle > angle
        else:
            mask = sum(FLAGS[f] for f in flags)
            keep = (df.gflags & mask) == 0
        k = df[keep]
        r = {'kept': int(keep.sum()), 'kept %': 100 * keep.mean(),
             'outliers removed %': 100 * (outlier & ~keep).sum() / max(outlier.sum(), 1),
             'good kept %': 100 * (~outlier & keep).sum() / max((~outlier).sum(), 1),
             'outliers in kept %': 100 * outlier[keep].mean()}
        r['Youden'] = r['outliers removed %'] + r['good kept %'] - 100
        for wl in [args.ref_band, args.swir_band]:
            r[f'bias {wl} %'] = k[f'relg{wl}'].median()
            r[f'IQR {wl} %'] = k[f'relg{wl}'].quantile(.75) - k[f'relg{wl}'].quantile(.25)
            r[f'MAPD {wl} %'] = np.nanmedian(np.abs(k[f'relg{wl}']))
        res[label] = r
    print(pd.DataFrame(res).T.round(1).to_string())

    if args.fig_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 9})
        os.makedirs(args.fig_dir, exist_ok=True)
        wls = [wl for wl in ds.wl.values if wl != 1375]
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        colors = {'none': '#999999', 'low': '#1b6ca8', 'moderate': '#66a182', 'high': '#d1495b'}
        allrel = {}
        for site in sites_of(args):
            ds = load(args.odir, site, args)
            d = ds.isel(time=selection(ds, args).values)
            lev = glint_level(d.glint_flags.values)
            for name, sim in [('', 'sim_median'), ('g', 'simg_median')]:
                rel = (100 * (d[sim] - d.sat_median) / d.sat_median).sel(wl=wls).values
                for l in np.unique(lev):
                    allrel.setdefault((name, l), []).append(rel[lev == l])
        for i, name in enumerate(['', 'g']):
            for l in LEVELS:
                if (name, l) not in allrel:
                    continue
                rel = np.concatenate(allrel[(name, l)])
                base = l.split(',')[0]
                ax[i].plot(wls, np.nanmedian(rel, 0), 'o--' if 'near' in l else 'o-', c=colors[base], ms=3,
                           label=f'{l} (N={len(rel)})')
            ax[i].axhline(0, c='k', lw=.6)
            ax[i].set_xticks([443, 665, 865, 1610, 2190])
            ax[i].set_xlabel('S2 band (nm)')
            ax[i].set_title(['without simulated glint', f'with simulated glint ({args.glint_stats})'][i], fontsize=9)
        ax[0].set_ylabel('median (sim - sat)/sat (%)')
        ax[0].set_ylim(-60, 80)
        ax[1].legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(opj(args.fig_dir, 's2_glint_levels.pdf'))
        print(f'figure -> {opj(args.fig_dir, "s2_glint_levels.pdf")}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('command', choices=['stats', 'drivers', 'figures', 'glint'])
    p.add_argument('--odir', default=ODIR, help='folder of the gathered matchup files')
    p.add_argument('--sites', nargs='+', default=None,
                   help='<site>_<level> (default: all the matchup files of --odir)')
    p.add_argument('--frac-min', type=float, default=0.5)
    p.add_argument('--npix-min', type=int, default=50)
    p.add_argument('--dt-max', type=float, default=60., help='nearest measurement to the overpass (min)')
    p.add_argument('--aot-max', type=float, default=0.3)
    p.add_argument('--glint-min', type=float, default=30., help='minimum glint angle (deg)')
    p.add_argument('--bands', type=int, nargs='+', default=[443, 665, 865, 1610, 2190], help='drivers: bands (nm)')
    p.add_argument('--fig-dir', default=None, help='figures, glint: output folder of the figures (figures: default .)')
    p.add_argument('--glint-site', default='AAOT_lev20', help='figures: site of the glint-angle figure')
    p.add_argument('--glint-stats', default='cm_iso', help='glint statistics of the matchup files (suffix)')
    p.add_argument('--glint-pixels', action='store_true',
                   help='satellite statistics without the SWIR threshold, homogeneous glint scenes kept')
    p.add_argument('--flags-exclude', nargs='+', default=None, choices=list(FLAGS),
                   help='glint screening by flags instead of --glint-min')
    p.add_argument('--ref-band', type=int, default=865, help='glint: band of the relative outlier criterion')
    p.add_argument('--swir-band', type=int, default=2190, help='glint: band of the absolute outlier criterion')
    p.add_argument('--outlier-rel', type=float, default=30., help='glint: relative outlier threshold (%%)')
    p.add_argument('--outlier-abs', type=float, default=5., help='glint: absolute outlier threshold (x1e-3)')
    args = p.parse_args()
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    {'stats': cmd_stats, 'drivers': cmd_drivers, 'figures': cmd_figures, 'glint': cmd_glint}[args.command](args)


if __name__ == '__main__':
    main()
