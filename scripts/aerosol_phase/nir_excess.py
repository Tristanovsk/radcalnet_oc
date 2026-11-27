'''
The NIR excess of the satellite over the radcalnet_oc 1.2 simulation, common to Sentinel-3 OLCI and Sentinel-2 MSI.

Results of 27 November 2026 (OLCI: VEIT, GABU, CORO, N 1735 without high or near-specular glint; MSI: VEIT L2.0,
CORO L1.5 with the heavy-tailed glint in the region of interest):

- ``bands``: the excess is additive. At 865 nm it is +1.6 to +2.4e-3 for OLCI (S3A 2.0-2.4, S3B 1.6-2.1, three
  sites) and +1.6 to +1.9e-3 for MSI. It is nearly flat from 709 to 885 nm. OLCI against the simulation has a slope
  of 1.02 and an intercept of 1.9e-3 at 865 nm. S3A is 2-3 % above S3B in the blue (calibration), within about
  0.5e-3 of it in the NIR.
- ``drivers``: in OLCI the excess is not sunglint: it is largest beyond 70 deg of glint angle and for the scenes
  without glint flag. It grows with the viewing zenith angle, the AOT and toward backscattering, and decreases with
  the sun zenith angle (about 0 in winter, 2.7e-3 in summer).
- ``gas``: in the H2O bands the excess is attenuated like the simulated signal (in-band / continuum 0.52 vs 0.45 at
  OLCI 940 nm, 0.82 vs 0.78 at 900 nm, 0.56 vs 0.38 for MSI B9). It comes from below or inside the water vapour,
  not from the sensor (offset, stray light) or high cirrus.
- ``phase``: dR(865) = c0 + dP(theta) tau865 / (4 mu_s mu_v) gives the same dP curve for both sensors: about -0.04
  at 90-130 deg and +0.03 at 150-170 deg of scattering angle, also at a fixed sun zenith angle (30-50 deg). dP is
  defined up to the offset c0 (1.84e-3 OLCI, 1.90e-3 MSI), so only its angular shape is meaningful. The AERONET
  phase functions are compared in ``pfn_compare.py``.

Usage:
    python nir_excess.py [bands|drivers|gas|phase|all]
'''
import sys
import warnings

import numpy as np
import pandas as pd

from phase_common import geometry, load_s2, load_s3, tau865

warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)


def ols(X, y, names):
    '''Least squares with an intercept and centred predictors: coefficient, t value, R2.'''
    X = np.asarray(X, float); y = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    A = np.column_stack([np.ones(ok.sum()), X[ok] - X[ok].mean(0)])
    b, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    r = y[ok] - A @ b
    se = np.sqrt(np.diag(r @ r / (ok.sum() - A.shape[1]) * np.linalg.inv(A.T @ A)))
    t = pd.DataFrame({'beta': b, 't': b / se}, index=['mean'] + list(names))
    return t, 1 - r.var() / y[ok].var()


def bands():
    d, dR = load_s3()
    rows = []
    for site in ['VEIT', 'GABU', 'CORO']:
        m = d.site.values == site
        for plat in ['S3A', 'S3B']:
            mp = m & (d.platform.values == plat)
            for b in ['Oa03', 'Oa06', 'Oa08', 'Oa11', 'Oa12', 'Oa16', 'Oa17', 'Oa18', 'Oa21']:
                x = d.sel(band=b).isel(time=mp)
                rs, ro = x.Rsim.values, (x.Rsim * x.ratio_refl).values
                ok = np.isfinite(rs) & np.isfinite(ro)
                slope, icpt = np.polyfit(rs[ok], ro[ok], 1)
                rows.append(dict(site=site, platform=plat, band=b, wl=round(float(np.nanmedian(x.lambda0)), 1),
                                 N=int(ok.sum()), dR_1e3=1e3 * np.median(ro[ok] - rs[ok]),
                                 ratio_pct=100 * (np.nanmedian(x.ratio_refl) - 1), slope=slope, icpt_1e3=1e3 * icpt))
    t = pd.DataFrame(rows)
    print('OLCI - simulation, reflectance (1e-3), ratio (%) and regression OLCI = slope sim + icpt')
    print(t.pivot_table(index='band', columns=['site', 'platform'], values='dR_1e3').round(2).to_string())
    print(t.pivot_table(index='band', columns=['site', 'platform'], values='ratio_pct').round(1).to_string())
    print(t.groupby('band')[['wl', 'slope', 'icpt_1e3']].median().round(3).to_string())
    for site in ['VEIT', 'CORO']:
        d2, e = load_s2(site)
        sat, sim = d2.sat_median, d2.simg_median
        t = pd.DataFrame({'sim': sim.median('time').values, 'dR_1e3': 1e3 * e.median('time').values,
                          'ratio_pct': 100 * ((sat / sim) - 1).median('time').values}, index=d2.wl.values)
        for w in t.index:
            x, y = sim.sel(wl=w).values, sat.sel(wl=w).values
            ok = np.isfinite(x) & np.isfinite(y)
            t.loc[w, 'slope'], icpt = np.polyfit(x[ok], y[ok], 1)
            t.loc[w, 'icpt_1e3'] = 1e3 * icpt
        print(f'\nMSI {site} (N {d2.sizes["time"]}), MSI - simulation with glint')
        print(t.drop(index=1375).round(3).T.to_string())


def drivers():
    d, dR = load_s3()
    theta, m = geometry(d.sza.values, d.vza.values, d.azi.values)
    df = pd.DataFrame({k: d[k].values for k in ['sza', 'vza', 'glint_angle', 'aot550', 'wind', 'tcwv',
                                                'cloud_fraction', 'dt_min']})
    df['theta'] = theta
    df['glint_level'] = d.glint_level.values
    df['month'] = pd.DatetimeIndex(d.time.values).month
    for b in ['Oa08', 'Oa12', 'Oa17', 'Oa21']:
        df['dR_' + b] = 1e3 * dR.sel(band=b).values
    cols = ['dR_Oa08', 'dR_Oa12', 'dR_Oa17', 'dR_Oa21']
    print('OLCI excess (1e-3), medians per bin')
    for var, bins in [('glint_angle', [0, 30, 40, 50, 60, 70, 90]), ('aot550', [0, .05, .1, .15, .2, .3, 1]),
                      ('sza', [0, 30, 40, 50, 60, 75]), ('vza', [0, 10, 20, 30, 40, 50]),
                      ('theta', [90, 110, 120, 130, 140, 150, 180]), ('dt_min', [-61, -30, -15, 0, 15, 30, 61])]:
        g = df.groupby(pd.cut(df[var], bins), observed=True)
        print(f'\nby {var}'); print(g[cols].median().round(2).assign(N=g.size()).to_string())
    g = df.groupby('glint_level')
    print('\nby glint level'); print(g[cols].median().round(2).assign(N=g.size()).to_string())
    g = df.groupby('month')
    print('\nby month'); print(g[['dR_Oa17', 'aot550', 'glint_angle', 'sza']].median().round(2).assign(N=g.size()).to_string())
    t, r2 = ols(df[['vza', 'sza', 'aot550', 'theta', 'glint_angle']], df.dR_Oa17, ['vza', 'sza', 'aot550', 'theta', 'glint_angle'])
    print(f'\nOLS of the Oa17 excess (1e-3), centred predictors, R2 {r2:.3f}'); print(t.round(4).to_string())
    for site in ['VEIT', 'CORO']:
        d2, e = load_s2(site)
        theta, m = geometry(d2.sza.values, d2.vza.values, d2.azi.values)
        df = pd.DataFrame({'aot550': d2.aot550.values, 'sza': d2.sza.values, 'theta': theta,
                           'glint_angle': d2.glint_angle.values, 'B10_1e3': 1e3 * d2.sat_median.sel(wl=1375).values})
        for w in [865, 1610, 2190]:
            df[f'dR_{w}'] = 1e3 * e.sel(wl=w).values
        cols = ['dR_865', 'dR_1610', 'dR_2190', 'B10_1e3']
        print(f'\nMSI {site} excess (1e-3)')
        for var, bins in [('aot550', [0, .05, .1, .15, .2, .3]), ('sza', [0, 30, 40, 50, 60, 75]),
                          ('theta', [90, 130, 140, 150, 160, 180]), ('glint_angle', [0, 30, 40, 50, 60, 90])]:
            g = df.groupby(pd.cut(df[var], bins), observed=True)
            print(f'by {var}'); print(g[cols].median().round(2).assign(N=g.size()).to_string())


def gas():
    '''In-band / continuum ratio of the simulated signal and of the excess. A source below the absorber is
    attenuated like the surface signal; an additive sensor term or a layer above the absorber is not.'''
    d, dR = load_s3()
    R, wl = d.Rsim, d.lambda0.median('time')

    def lin(x, b, b1, b2):
        w, w1, w2 = [float(wl.sel(band=k)) for k in (b, b1, b2)]
        return x.sel(band=b1) + (x.sel(band=b2) - x.sel(band=b1)) * (w - w1) / (w2 - w1)

    print('OLCI, band / continuum: simulated signal | excess')
    for b, b1, b2, gas_ in [('Oa20', 'Oa18', 'Oa21', 'H2O 940'), ('Oa19', 'Oa18', 'Oa21', 'H2O 900'),
                            ('Oa11', 'Oa10', 'Oa12', 'H2O 709 (weak)'), ('Oa13', 'Oa12', 'Oa16', 'O2-A 761')]:
        rc, ec = lin(R, b, b1, b2), lin(dR, b, b1, b2)
        T, a = (R.sel(band=b) / rc).values, (dR.sel(band=b) / ec).values
        ok = np.isfinite(T) & np.isfinite(a) & (np.abs(ec.values) > 1e-3)
        print(f'  {gas_:15s} {np.median(T[ok]):.2f} | {np.median(a[ok]):.2f} '
              f'(IQR {np.percentile(a[ok], 25):.2f}-{np.percentile(a[ok], 75):.2f}, N {ok.sum()})')
    print('  (O2-A: OLCI is deeper than the simulation beyond the excess, the open O2-A issue, ROADMAP task 26)')
    for site in ['VEIT', 'CORO']:
        d2, e = load_s2(site)
        T = (d2.simg_median.sel(wl=945) / d2.simg_median.sel(wl=865)).values
        a = (e.sel(wl=945) / e.sel(wl=865)).values
        ok = np.isfinite(a) & (np.abs(e.sel(wl=865).values) > 1e-3)
        print(f'MSI {site}, B9 / B8A: {np.median(T[ok]):.2f} | {np.median(a[ok]):.2f} '
              f'(IQR {np.percentile(a[ok], 25):.2f}-{np.percentile(a[ok], 75):.2f}, N {ok.sum()})')


def phase():
    '''dR(865) = c0 + dP tau865 / (4 mu_s mu_v), then dP per scattering-angle bin with c0 removed.'''
    d, dR = load_s3()
    theta, m = geometry(d.sza.values, d.vza.values, d.azi.values)
    frames = [pd.DataFrame(dict(sensor='OLCI', theta=theta, sza=d.sza.values, tau=tau865(d.aot550.values, d.ang_exp.values),
                                m=m, dR=dR.sel(band='Oa17').values))]
    for site in ['VEIT', 'CORO']:
        d2, e = load_s2(site)
        theta, m = geometry(d2.sza.values, d2.vza.values, d2.azi.values)
        frames.append(pd.DataFrame(dict(sensor='MSI', theta=theta, sza=d2.sza.values,
                                        tau=tau865(d2.aot550.values, d2.ang_exp.values), m=m, dR=e.sel(wl=865).values)))
    df = pd.concat(frames, ignore_index=True).dropna()
    df['x'] = df.tau * df.m
    for sensor, g in df.groupby('sensor'):
        A = np.column_stack([np.ones(len(g)), g.x])
        b, *_ = np.linalg.lstsq(A, g.dR, rcond=None)
        r = g.dR - A @ b
        se = np.sqrt(np.diag(r @ r / (len(g) - 2) * np.linalg.inv(A.T @ A)))
        print(f'{sensor}: dR865 = {1e3 * b[0]:.2f}e-3 (+/-{1e3 * se[0]:.2f}) + {b[1]:.3f} (+/-{se[1]:.3f}) '
              f'tau865 / (4 mu_s mu_v), N {len(g)}')
        g = g.assign(dP=(g.dR - b[0]) / g.x)
        gg = g[g.tau > 0.03]
        bins = [90, 120, 130, 140, 150, 160, 170, 180]
        print('  dP per scattering-angle bin (tau865 > 0.03):')
        print(gg.groupby(pd.cut(gg.theta, bins), observed=True).dP.agg(['median', 'size']).round(3).T.to_string())
        gs = gg[(gg.sza > 30) & (gg.sza < 50)]
        print('  sza 30-50 deg only:',
              gs.groupby(pd.cut(gs.theta, [90, 140, 150, 160, 180]), observed=True).dP.median().round(3).to_dict())


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, f in [('bands', bands), ('drivers', drivers), ('gas', gas), ('phase', phase)]:
        if what in (name, 'all'):
            print(f'\n######## {name}'); f()
