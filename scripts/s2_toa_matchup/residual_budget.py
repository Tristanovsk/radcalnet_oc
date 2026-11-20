'''
Physical residual budget of the Sentinel-2 TOA matchups (visible to SWIR).

The residual E = sat - sim (MSI L1C minus the radcalnet-oc simulation with the heavy-tailed direct glint,
``simg_median``) is compared with the terms the simulation does not contain, each predicted by a physical model
from independent inputs, with no coefficient fitted on the matchups, and with the uncertainty of the simulation and
of the observation. The question is whether E - (predicted terms) stays within its uncertainty (closure).

Predicted terms (reflectance, sat - sim > 0 when the simulation lacks them)
-------------------------------------------------------------------------
1. Rayleigh scattering above the water vapour in B10 (1375 nm). In the saturated water-vapour band the light
   scattered by the molecules above the absorbing layer escapes absorption; the simulation multiplies the whole
   path reflectance by the gaseous transmittance (scattering below the absorber) and gives ~0. In single scattering,
   R_above = f_above tau_r(1375) P_R(Theta) / (4 mu_s mu_v), f_above the fraction of the molecules above the
   altitude where the band becomes opaque (0.55 +/- 0.08: 4-6 km, p/p0 = 0.61-0.47). It is a term of B10 only and
   is removed from B10 before the cirrus estimate.
2. Thin cirrus (and stratospheric aerosol) from B10: what remains in B10, Delta_B10 = B10 - R_above, was scattered
   above the water vapour (Gordon 1999, MODIS ATBD of the normalised water-leaving radiance, 3.1.1.9.1: in the first
   approximation the reflectance at 1380 nm is the contribution of the high layer, and its simplest correction
   delta(lambda) = rho(1380) is within a factor of 2 of the full multiple-scattering one). The high layer is added at
   all bands as C(lambda) = Delta_B10 s(lambda) / K_a: K_a = 0.9 +/- 0.1 for the residual water vapour above the
   cirrus in B10, s(lambda) the spectral ratio of the ice cloud reflectance to 1375 nm (1 in the visible-NIR where
   ice does not absorb; 0.95 +/- 0.10 at 1610 nm and 0.85 +/- 0.15 at 2190 nm, ice absorption lowering the
   single-scattering albedo; bounds, not measured here).
3. Whitecaps (absent from OSOAA): WC = T(theta_s) T(theta_v) 0.22 f(U) a(lambda), coverage
   f = 2.95e-6 U^3.52 (Monahan and O'Muircheartaigh 1980; Gordon and Wang 1994, Eq. 3, effective reflectance 0.22),
   spectral factor a(lambda) of the foam relative to the visible (Frouin et al. 1996: -40 % at 0.85 um, -85 % at
   1.65 um; 0.10 at 2.19 um, where the water absorbs more), total transmittances of a Lambertian term. The scatter
   of the coverage is of the order of f itself (Gordon and Wang 1994): u = 100 %.
4. Land environment (adjacency): 6S environment functions of the molecules and aerosols on the Natural Earth land
   polygons (``sixs_terms.py``), land reflectance of vegetation (an upper bound): term = half of the bound, u = half.

Uncertainty-only terms (zero mean, standard uncertainties)
---------------------------------------------------------
- direct glint: ``glint_unc`` (half range of the plausible slope statistics at the wind +/- 1.5 m/s);
- AOT(550) +/- 0.01 (AERONET-OC), times the sensitivity dRtoa/dAOT of the simulation itself, from the simulations of
  the measurements of each window (pooled within-scene slope of ``rec_Rtoa`` against ``rec_aot550``);
- aerosol model: half range of the mixtures compatible with the measured AOT spectrum (validation report, Table
  "aerosol": slopes per unit AOT of the retrieved, six-model and WASO+MACL mixtures), times AOT(550);
- MSI L1C absolute calibration: 2 % of the observed reflectance (k = 1, assumed);
- water-leaving term: 5 % of Rrs, taken as 1.0 / 0.3 % of the TOA reflectance at 443 / 560 nm (budget of the
  validation report) and 0.15e-3 at 865 nm (5e-5 sr-1 of Rrs(865)), nil beyond;
- time: ``simg_std``, the spread of the simulations of the measurements within the window (the atmosphere and the
  water of the overpass are represented by the median of the window).

The terms are combined in quadrature. The script prints, per site and band, the medians of E, of the predicted terms,
of the remaining residual r = E - sum(predicted) and of its standard uncertainty u, the fraction of the matchups with
|r| < u and < 2u (68 and 95 % expected if the budget closes), and r per glint-angle bin and per S2 unit; it writes
one CSV per site with the budget of each matchup.

Usage::

    python residual_budget.py [--odir /data/radcalnet/matchups/sentinel2/v1.2.0] [--csv-dir DIR]
'''
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import cartopy.io.shapereader as shpreader
import shapely

from sixs_terms import environment_weights, rayleigh_tau, diffuse_rayleigh, diffuse_aerosol, env_density  # noqa: F401

opj = os.path.join

ODIR = '/data/radcalnet/matchups/sentinel2/v1.2.0'
SITES = {'AAOT_lev20': ('VEIT', (45.3139, 12.5083)), 'Section-7_Platform_lev15': ('CORO', (44.5458, 29.4466))}
SUFFIX = '_cm_iso_mix'
DROP = ['rec_Rtoa_spectrum', 'rec_Rglint_spectrum', 'sim_spectrum_median', 'glint_spectrum_median']
BANDS = [443, 560, 665, 865, 1610, 2190]
SEL = dict(frac_min=0.5, npix_min=50, dt_max=60., aot_max=0.3, glint_min=30.)

#: fraction of the molecules above the water-vapour layer opaque at 1375 nm (4-6 km)
F_ABOVE, U_F_ABOVE = 0.55, 0.08
#: water vapour above the cirrus in B10
K_A, U_K_A = 0.9, 0.1
#: ice cloud reflectance relative to 1375 nm (value, standard uncertainty)
S_ICE = {443: (1., .05), 560: (1., .05), 665: (1., .05), 865: (1., .05), 1610: (.95, .10), 2190: (.85, .15)}
#: foam reflectance relative to the visible (Frouin et al. 1996; Gordon and Wang 1994 / SeaDAS a_wc at 665, 865)
A_FOAM = {443: 1., 560: 1., 665: .889, 865: .645, 1610: .15, 2190: .10}
#: land reflectance of vegetation (upper bound of the environment reflectance)
RHO_LAND = {443: .04, 560: .08, 665: .05, 865: .30, 1610: .22, 2190: .12}
#: half range of the AOT-spectrum-compatible aerosol mixtures, per unit AOT(550) (validation report)
U_AER_MODEL = {443: 1.75e-3, 560: 1.75e-3, 665: 1.75e-3, 865: 1.75e-3, 1610: .65e-3, 2190: .70e-3}
U_CAL = 0.02
#: water-leaving term (fraction of the TOA reflectance, or absolute at 865 nm)
U_WATER_REL = {443: .010, 560: .003, 665: 0., 865: 0., 1610: 0., 2190: 0.}
U_WATER_ABS = {443: 0., 560: 0., 665: 0., 865: .15e-3, 1610: 0., 2190: 0.}
U_AOT = 0.01
GLINT_BINS = [0, 20, 30, 40, 50, 60, 90]


def load(odir, site, glint_min=SEL['glint_min']):
    ds = xr.open_dataset(opj(odir, f'{site}_s2_toa_matchup{SUFFIX}.nc'), drop_variables=DROP).load()
    keep = ((ds.frac_valid > SEL['frac_min']) & (ds.n_pix >= SEL['npix_min'])
            & (np.abs(ds.dt_nearest) <= SEL['dt_max']) & (ds.aot550 < SEL['aot_max'])
            & (ds.glint_angle > glint_min))
    return ds, ds.isel(time=keep.values)


def scattering_angle(sza, vza, raa):
    '''Scattering angle (deg) of the single scattering, relative azimuth raa (deg, as in the matchup files: 180 on the specular side, 0 for the backscattering).'''
    mus, muv = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    cos_t = -mus * muv - np.sqrt(1 - mus ** 2) * np.sqrt(1 - muv ** 2) * np.cos(np.radians(raa))
    return np.degrees(np.arccos(np.clip(cos_t, -1, 1)))


def rayleigh_above(ds):
    '''Single-scattering Rayleigh reflectance of the molecules above the water vapour at 1375 nm, and its u.'''
    mus, muv = np.cos(np.radians(ds.sza.values)), np.cos(np.radians(ds.vza.values))
    cos_t = np.cos(np.radians(scattering_angle(ds.sza.values, ds.vza.values, ds.raa.values)))
    ss = rayleigh_tau(1375.) * ds.pressure.values / 1013.25 * 0.75 * (1 + cos_t ** 2) / (4 * mus * muv)
    return F_ABOVE * ss, U_F_ABOVE * ss


def aot_sensitivity(full, band):
    '''dRtoa/dAOT(550) of the simulation: pooled within-scene slope of rec_Rtoa against rec_aot550.'''
    rec = pd.DataFrame({'scene': full.rec_scene.values, 'aot': full.rec_aot550.values,
                        'r': full.rec_Rtoa.sel(wl=band).values}).dropna()
    rec = rec[rec.groupby('scene').aot.transform('count') > 2]
    da = rec.aot - rec.groupby('scene').aot.transform('mean')
    dr = rec.r - rec.groupby('scene').r.transform('mean')
    return float((da * dr).sum() / (da ** 2).sum())


def total_transmittance(tau_r, tau_a, mu):
    return np.exp(-(tau_r + tau_a) / mu) + diffuse_rayleigh(tau_r, mu) + diffuse_aerosol(tau_a, mu)


def budget(full, ds, weights):
    '''Budget of each matchup (DataFrame, one row per matchup and band).'''
    rows = []
    r_above, u_above = rayleigh_above(ds)
    b10 = ds.sat_median.sel(wl=1375).values
    d_b10 = b10 - r_above
    u_b10 = np.hypot(u_above, 1e-4 / np.sqrt(12))   # Rayleigh above, quantisation of the L1C (1e-4)
    mus, muv = np.cos(np.radians(ds.sza.values)), np.cos(np.radians(ds.vza.values))
    unit = ds['product'].astype(str).str.slice(0, 3).values
    for band in BANDS:
        sat = ds.sat_median.sel(wl=band).values
        sim = ds.simg_median.sel(wl=band).values
        E = sat - sim
        # cirrus
        s, us = S_ICE[band]
        c = s / K_A
        u_c = c * np.hypot(us / s, U_K_A / K_A)
        C = c * d_b10
        u_C = np.hypot(u_c * d_b10, c * u_b10)
        # whitecaps
        tau_r, tau_a = rayleigh_tau(band), ds.aot550.values * (band / 550.) ** -ds.ang_exp.values
        T = total_transmittance(tau_r, tau_a, mus) * total_transmittance(tau_r, tau_a, muv)
        U = ds.wind.values
        WC = T * 0.22 * 2.95e-6 * U ** 3.52 * A_FOAM[band]
        u_WC = WC.copy()
        # environment
        mu_s, mu_v = mus, muv
        env_up = total_transmittance(tau_r, tau_a, mu_s) * (
            diffuse_rayleigh(tau_r, mu_v) * weights['W_R'] + diffuse_aerosol(tau_a, mu_v) * weights['W_P']) \
            * RHO_LAND[band]
        ENV, u_ENV = env_up / 2, env_up / 2
        # uncertainty-only terms
        u_glint = ds.glint_unc.sel(wl=band).values
        u_aot = abs(aot_sensitivity(full, band)) * U_AOT * np.ones_like(sat)
        u_model = U_AER_MODEL[band] * ds.aot550.values
        u_cal = U_CAL * sat
        u_water = np.hypot(U_WATER_REL[band] * sat, U_WATER_ABS[band])
        u_time = ds.simg_std.sel(wl=band).fillna(0).values
        pred = C + WC + ENV
        r = E - pred
        u = np.sqrt(u_C ** 2 + u_WC ** 2 + u_ENV ** 2 + u_glint ** 2 + u_aot ** 2 + u_model ** 2 + u_cal ** 2
                    + u_water ** 2 + u_time ** 2)
        rows.append(pd.DataFrame({
            'time': ds.time.values, 'band': band, 'unit': unit, 'glint_angle': ds.glint_angle.values,
            'sza': ds.sza.values, 'vza': ds.vza.values, 'azi': ds.azi.values,
            'glint_flags': ds.glint_flags.values, 'glint_fraction': ds.glint_fraction.values,
            'wind': U, 'aot550': ds.aot550.values, 'sat': sat, 'sim': sim, 'E': E, 'B10': b10, 'R_above': r_above,
            'cirrus': C, 'whitecaps': WC, 'environment': ENV, 'pred': pred, 'r': r,
            'u_cirrus': u_C, 'u_whitecaps': u_WC, 'u_environment': u_ENV, 'u_glint': u_glint, 'u_aot': u_aot,
            'u_aer_model': u_model, 'u_cal': u_cal, 'u_water': u_water, 'u_time': u_time, 'u': u}))
    return pd.concat(rows, ignore_index=True)


def report(name, df):
    med = df.groupby('band').median(numeric_only=True)
    k = 1e3
    print(f'\n===== {name}: {df.time.nunique()} matchups (glint angle > {df.glint_angle.min():.0f} deg), x1e-3 (medians)')
    cols = ['sat', 'E', 'cirrus', 'whitecaps', 'environment', 'pred', 'r']
    t = (med[cols] * k).round(2)
    t['r %'] = (100 * med['r'] / med['sat']).round(1)
    t['E %'] = (100 * med['E'] / med['sat']).round(1)
    print(t.to_string())
    print('-- standard uncertainties (medians, x1e-3) and closure')
    ucols = [c for c in df.columns if c.startswith('u_')] + ['u']
    t = (med[ucols] * k).round(2)
    z = (df.r / df.u).abs()
    t['|r|<u'] = (z < 1).groupby(df.band).mean().round(2)
    t['|r|<2u'] = (z < 2).groupby(df.band).mean().round(2)
    t['med r/u'] = (df.r / df.u).groupby(df.band).median().round(2)
    print(t.to_string())
    print('-- remaining residual r per glint-angle bin (median, x1e-3)')
    gb = pd.cut(df.glint_angle, GLINT_BINS)
    print((df.groupby([gb, 'band'], observed=True).r.median().unstack() * k).round(2).to_string())
    print('-- remaining residual r per S2 unit (median, x1e-3) and N')
    print((df.groupby(['unit', 'band']).r.median().unstack() * k).round(2)
          .assign(N=df[df.band == 865].groupby('unit').size()).to_string())
    print('-- check of the cirrus model (not a fit): slope of E against the B10 excess, with the glint angle, '
          'against the predicted s / K_a')
    for band in BANDS:
        x = df[df.band == band]
        X = np.column_stack([np.ones(len(x)), x.B10 - x.R_above, x.glint_angle])
        beta = np.linalg.lstsq(X, x.E.values, rcond=None)[0]
        s, us = S_ICE[band]
        c, u_c = s / K_A, s / K_A * np.hypot(us / s, U_K_A / K_A)
        print(f'   {band}: observed {beta[1]:.2f}, predicted {c:.2f} +/- {u_c:.2f}')
    print('-- B10: observed, Rayleigh above the water vapour, cirrus excess (p10/p50/p90, x1e-3)')
    b = df[df.band == 865]
    for v in ['B10', 'R_above']:
        print(f'   {v:8s}', np.round(k * np.percentile(b[v], [10, 50, 90]), 2))
    print('   excess  ', np.round(k * np.percentile(b.B10 - b.R_above, [10, 50, 90]), 2))


#: sun zenith classes (deg) and minimum number of matchups of a class of the model-gap table
SZA_CLASSES = [(0., 40.), (40., 55.), (55., 90.)]
N_MIN = 20


def glint_level(flags):
    '''Glint level of the bitmask of radcalnet_oc.coxmunk.glint_flags (near specular first).'''
    f = np.asarray(flags).astype(int)
    return np.select([f & 8 > 0, f & 4 > 0, f & 2 > 0, f & 1 > 0],
                     ['near_specular', 'high', 'moderate', 'low'], 'none')


def gap(g):
    '''Model gap of a class: the part of the remaining residual r not covered by the known terms u,
    sqrt(median(r)^2 + max(0, s_r^2 - median(u)^2)), s_r = 1.4826 MAD(r) (robust spread).'''
    r = g.r.values
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    return np.sqrt(np.median(r) ** 2 + max(0., s ** 2 - np.median(g.u.values) ** 2))


def model_gap_table(df):
    '''
    Table of the model gap per glint level, sun zenith class and band (reflectance), from the budget of all the
    matchups (all glint angles, both sites pooled). A class with fewer than N_MIN matchups takes the value of its
    glint level over all the sun zenith angles (``provisional`` = 1, also when the level itself has fewer).
    '''
    df = df.assign(level=glint_level(df.glint_flags))
    rows = []
    for level, gl in df.groupby('level'):
        for lo, hi in SZA_CLASSES:
            gs = gl[(gl.sza >= lo) & (gl.sza < hi)]
            for band in BANDS:
                n = int((gs.band == band).sum())
                if n >= N_MIN:
                    u, prov = gap(gs[gs.band == band]), 0
                else:
                    ga = gl[gl.band == band]
                    u, prov = gap(ga), int(len(ga) < N_MIN)
                    n = len(ga)
                rows.append(dict(level=level, sza_min=lo, sza_max=hi, wl=band, u_model=round(float(u), 6), n=n,
                                 provisional=prov))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--odir', default=ODIR)
    p.add_argument('--csv-dir', default=None)
    p.add_argument('--glint-min', type=float, default=SEL['glint_min'],
                   help='minimum glint angle (deg); 0 keeps the near-specular scenes (CSV suffixed _all)')
    p.add_argument('--model-table', default=None,
                   help='write the model-gap table (glint level x sun zenith x band) of the matchups to this CSV '
                        '(with --glint-min 0)')
    args = p.parse_args()
    land = shapely.unary_union(list(shpreader.Reader(
        shpreader.natural_earth('10m', 'physical', 'land')).geometries()))
    frames = []
    for site, (code, (lat, lon)) in SITES.items():
        full, ds = load(args.odir, site, args.glint_min)
        weights = environment_weights(lat, lon, land)
        df = budget(full, ds, weights)
        frames.append(df)
        report(f'{code} ({site}, coast at {weights["coast_km"]:.0f} km)', df)
        sens = {b: round(1e3 * aot_sensitivity(full, b), 1) for b in BANDS}
        print('   dRtoa/dAOT of the simulation (x1e-3 per unit AOT):', sens)
        if args.csv_dir:
            os.makedirs(args.csv_dir, exist_ok=True)
            sfx = '' if args.glint_min == SEL['glint_min'] else '_all'
            df.to_csv(opj(args.csv_dir, f'{code}_residual_budget{sfx}.csv'), index=False)
    if args.model_table:
        table = model_gap_table(pd.concat(frames, ignore_index=True))
        with open(args.model_table, 'w') as f:
            f.write('# Model gap of the radcalnet-oc TOA simulation (standard uncertainty, reflectance) per sunglint level\n'
                    '# (radcalnet_oc.coxmunk.glint_flags at 865 nm, near specular first), sun zenith class and band:\n'
                    '# part of the residual MSI - simulation (with the heavy-tailed glint) of the Sentinel-2 matchups\n'
                    '# of VEIT (AAOT L2.0) and CORO (Section-7 L1.5) that the physical budget does not explain,\n'
                    '# sqrt(median(r)^2 + max(0, s_r^2 - median(u)^2)), r the residual after the predicted terms\n'
                    '# (cirrus, whitecaps, environment), u the known uncertainties (glint, AOT, aerosol model, MSI\n'
                    '# calibration 2 %, water, time), s_r = 1.4826 MAD(r); n matchups; provisional = 1: fewer than\n'
                    f'# {N_MIN} matchups in the level. Made by scripts/s2_toa_matchup/residual_budget.py --glint-min 0\n'
                    f'# --model-table, radcalnet_oc matchups v1.2.0, {pd.Timestamp.now():%Y-%m-%d}.\n')
            table.to_csv(f, index=False)
        print(f'model-gap table -> {args.model_table}')
        print((table.pivot_table(index=['level', 'sza_min'], columns='wl', values='u_model') * 1e3).round(2)
              .to_string())


if __name__ == '__main__':
    main()
