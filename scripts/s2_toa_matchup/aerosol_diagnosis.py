'''
Diagnosis of the AOT dependence of the NIR-SWIR deficit of the Sentinel-2 TOA matchups: aerosol model.

For each selected matchup (selection of ``matchup_analysis.py``), at the geometry of the scene and with the
AERONET-OC measurement nearest to the overpass (AOT(550), wind, normalized spectral AOT):

1. the path reflectance R_atm of each of the six OPAC models of the look-up table, at 865, 1610 and 2190 nm
   (one preparation of the LUT with an identity matrix of mixtures: the models are combined linearly at the same
   AOT(550) in ``LUT.lut_preparation``, so that the reflectance of any mixture is the same combination);
2. the normalized AOT spectrum of each model at the AERONET-OC bands and at 1610 and 2190 nm;
3. the mixtures: retrieved with ``kernel.Aerosol`` (DESE/MACL/WASO, as in ``s2_toa_matchup.py``), retrieved with
   the six models (non-negative least squares, proportions summing to 1), and fixed alternatives.

A last test checks whether the hygroscopic growth of the fine mode could explain the missing SWIR reflectance:
Angstrom exponents and single-scattering path reflectance per unit AOT(550) of OPAC models at relative humidities
of 0 to 95 % (MOPSMAP scattering matrices, ``--opac-dir``), at the median geometry of the matchups.

The analysis compares the retrieved mixtures with the measured AOT spectrum (residuals at 865 and 1020 nm,
extrapolation to the SWIR) and, for each alternative mixture, shifts the deficit by the change of R_atm and fits
again deficit = offset + slope x AOT(550): a mixture that removes the slope explains the AOT dependence.

The scenes are cached per site in ``--cache`` (netCDF). Example::

    python aerosol_diagnosis.py --n-jobs 4
'''
import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import nnls

import radcalnet_oc as radoc
from s2_toa_matchup import aeronet_inputs

from radcalnet_oc import __version__ as RADCALNET_OC_VERSION  # noqa: E402
ODIR = f'/data/radcalnet/matchups/sentinel2/v{RADCALNET_OC_VERSION}'   # one folder per radcalnet_oc version
CACHE = ODIR + '/aerosol_diagnosis'
SITES = {'AAOT_lev20': ('AAOT', 'lev20'), 'Galata_Platform_lev20': ('Galata_Platform', 'lev20'),
         'AAOT_lev15': ('AAOT', 'lev15')}
BANDS = [865, 1610, 2190]
MODELS = list(radoc.lut.AEROSOL_MODELS)
#: alternative mixtures (proportions of AOT(550))
ALTERNATIVES = {'MACL': {'MACL_rh70': 1.}, 'COAV': {'COAV_rh70': 1.}, 'DESE': {'DESE_rh70': 1.},
                'COAV+MACL (default)': {'COAV_rh70': .5, 'MACL_rh70': .5}}


def selection(ds, dt_max=60, aot_max=0.3, glint_min=30):
    return ((ds.frac_valid > 0.5) & (ds.n_pix >= 50) & (np.abs(ds.dt_nearest) <= dt_max)
            & (ds.aot550 < aot_max) & (ds.glint_angle > glint_min))


def nearest_records(ds):
    '''Time of the AERONET-OC measurement nearest to each scene (from the rec_* variables).'''
    rec = pd.DataFrame({'scene': ds.rec_scene.values, 'time': ds.rec_time.values, 'dt': np.abs(ds.rec_dt.values)})
    return rec.sort_values('dt').groupby('scene').first()['time']


def fit_six(naot, naot_models, weight=10.):
    '''Proportions of the six models fitting the normalized AOT (non-negative, sum to 1 by a weighted row).'''
    # naot_models: (model, wavelength); rows of A: wavelengths, then the sum constraint
    A = np.vstack([naot_models.T, weight * np.ones(naot_models.shape[0])])
    b = np.append(naot, weight)
    f, _ = nnls(A, b)
    return f / f.sum()


def scene(site, level, s, record, naot_lut):
    '''Path reflectance of the six models and AOT spectra for one scene (Dataset).'''
    logging.getLogger().setLevel(logging.ERROR)
    warnings.simplefilter('ignore')
    lut = radoc.LUT()
    eye = xr.DataArray(np.eye(len(MODELS)), coords={'mix': MODELS, 'model': MODELS}, dims=('mix', 'model'))
    wind = float(record.wind) if np.isfinite(record.wind) else 2.
    aot = float(record.aot550)
    lut.lut_preparation(wind=wind, sza=[float(s.sza)], vza=[float(s.vza)], azi=[float(s.azi)], aot_refs=[aot],
                        aerosol_combination=eye)
    R = lut.Rdiff_lut.squeeze(drop=True).interp(wl=BANDS, method='quadratic')
    naot = record.naot.dropna('wl_aeronet')
    wl_a = naot.wl_aeronet.values
    nm = naot_lut.interp(wl=np.r_[wl_a, BANDS[1:]], method='quadratic').transpose('model', 'wl')
    aerosol = radoc.Aerosol(naot.expand_dims(time=[0]), naot_lut)
    aerosol.process()
    f3 = aerosol.model_db.aerosol_combination.squeeze('time', drop=True).reindex(model=MODELS, fill_value=0.)
    f6 = fit_six(naot.values, nm.sel(model=MODELS).values[:, :len(wl_a)])
    out = xr.Dataset({
        'Ratm_model': R.rename(mix='model').transpose('model', 'wl'),
        'naot_model': nm.sel(model=MODELS),
        'naot_meas': ('wl_aeronet', naot.values),
        'f3': ('model', f3.values), 'f6': ('model', f6),
        'aot550': aot, 'wind': wind, 'ang_exp': float(record.ang_exp)},
        coords={'wl_aeronet': wl_a})
    return out.expand_dims(time=[s.time.values])


def compute(args, site_key):
    from joblib import Parallel, delayed
    site, level = SITES[site_key]
    cache = os.path.join(args.cache, f'{site_key}.nc')
    if os.path.exists(cache):
        return xr.open_dataset(cache).load()
    ds = xr.open_dataset(f'{args.odir}/{site_key}_s2_toa_matchup.nc')
    sel = ds.sel(time=selection(ds))
    rec_time = nearest_records(ds)
    records = aeronet_inputs(site, level).load()
    naot_lut = radoc.LUT().naot_lut
    jobs = [(sel.sel(time=t), records.sel(time=rec_time[pd.Timestamp(t)])) for t in sel.time.values
            if pd.Timestamp(t) in rec_time.index][:args.max_scenes]
    res = Parallel(n_jobs=args.n_jobs)(delayed(scene)(site, level, s, r, naot_lut) for s, r in jobs)
    out = xr.concat(res, 'time')
    D = (sel.sim_median - sel.sat_median).sel(wl=BANDS, time=out.time)
    out['D'] = D.reset_coords(drop=True).rename('D')
    out['sim'] = sel.sim_median.sel(wl=BANDS, time=out.time).reset_coords(drop=True)
    for v in ['sza', 'vza', 'glint_angle']:
        out[v] = sel[v].sel(time=out.time).reset_coords(drop=True)
    os.makedirs(args.cache, exist_ok=True)
    out.to_netcdf(cache)
    return out


def mixture(f, ds):
    return (ds.Ratm_model * f).sum('model')


def fit_line(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[ok], y[ok], 1)
    return a, b


def analysis(data):
    pd.set_option('display.width', 200)
    ds = xr.concat(list(data.values()), 'time', join='outer')
    ds['site'] = ('time', np.concatenate([[k] * v.sizes['time'] for k, v in data.items()]))
    aot = ds.aot550.values
    bins = pd.cut(aot, [0, 0.08, 0.15, 0.3])
    counts = ', '.join(f'{k}: {v.sizes["time"]}' for k, v in data.items())
    print(f'=== {ds.sizes["time"]} matchups ({counts})')

    print('\n=== 1. retrieved DESE/MACL/WASO mixture (mean proportion of AOT(550)) per AOT(550) bin')
    f3 = ds.f3.to_pandas()[['DESE_rh70', 'MACL_rh70', 'WASO_rh0']]
    print(f3.groupby(bins, observed=True).mean().assign(N=f3.groupby(bins, observed=True).size()).round(3).to_string())
    f6 = ds.f6.to_pandas()
    print('    six-model retrieval')
    print(f6.groupby(bins, observed=True).mean().round(3).to_string())

    print('\n=== 2. normalized AOT: retrieved mixture / measured at 865 and 1020 nm, and Angstrom exponents')
    n3 = (ds.naot_model * ds.f3).sum('model')
    rows = pd.DataFrame({'mix/meas 865': (n3.sel(wl=865) / ds.naot_meas.sel(wl_aeronet=865)).values,
                         'mix/meas 1020': (n3.sel(wl=1020) / ds.naot_meas.sel(wl_aeronet=1020)).values,
                         'alpha meas 865-1020': (-np.log(ds.naot_meas.sel(wl_aeronet=1020) / ds.naot_meas.sel(wl_aeronet=865))
                                                 / np.log(1020 / 865)).values,
                         'alpha mix 865-1020': (-np.log(n3.sel(wl=1020) / n3.sel(wl=865)) / np.log(1020 / 865)).values,
                         'mix AOT2190/AOT550': n3.sel(wl=2190).values,
                         'meas extrap. 2190/550': (ds.naot_meas.sel(wl_aeronet=1020) * (2190 / 1020) ** (
                             np.log(ds.naot_meas.sel(wl_aeronet=1020) / ds.naot_meas.sel(wl_aeronet=865)) / np.log(1020 / 865))).values})
    print(rows.groupby(bins, observed=True).median().round(3).to_string())

    print('\n=== 3. deficit (x1e-3) = offset + slope x AOT(550) after replacing the retrieved mixture')
    R3 = mixture(ds.f3, ds)
    alts = {'retrieved (DESE/MACL/WASO)': R3, 'six-model retrieval': mixture(ds.f6, ds)}
    for name, comp in ALTERNATIVES.items():
        f = xr.DataArray([comp.get(m, 0.) for m in MODELS], coords={'model': MODELS}, dims='model')
        alts[name] = mixture(f, ds)
    # fine + coarse mixture constrained in the near infrared: proportion of MACL in WASO + MACL such that the
    # normalized AOT of the mixture equals the measured one at 1020 nm (or 865 nm)
    for wl_nir in (865, 1020):
        n_w, n_m = ds.naot_model.sel(model='WASO_rh0', wl=wl_nir), ds.naot_model.sel(model='MACL_rh70', wl=wl_nir)
        f_m = ((ds.naot_meas.sel(wl_aeronet=wl_nir) - n_w) / (n_m - n_w)).clip(0, 1)
        f = xr.concat([f_m if m == 'MACL_rh70' else (1 - f_m) if m == 'WASO_rh0' else 0. * f_m for m in MODELS],
                      dim=xr.DataArray(MODELS, dims='model', name='model'))
        alts[f'WASO+MACL fitted at {wl_nir} nm'] = mixture(f, ds)
        print(f'    MACL proportion fitted at {wl_nir} nm, median per AOT bin: ' +
              ', '.join(f'{v:.2f}' for v in f_m.to_pandas().groupby(bins, observed=True).median().values))
    rows = []
    for name, R in alts.items():
        D = 1e3 * (ds.D + (R - R3))   # gaseous transmittance close to 1 in these bands (neglected)
        r = {'mixture': name}
        for b in BANDS:
            a, s = fit_line(aot, D.sel(wl=b).values)
            r[f'offset {b}'], r[f'slope {b}'] = a, s
            r[f'median {b}'] = float(np.nanmedian(D.sel(wl=b)))
        rows.append(r)
    print(pd.DataFrame(rows).set_index('mixture').round(2).to_string())

    print('\n=== 4. consistency of the mixtures with the measured AOT spectrum (medians)')
    meas = ds.naot_meas
    wla = meas.wl_aeronet.values

    def alpha(n, w1, w2):
        return float((-np.log(n(w2) / n(w1)) / np.log(w2 / w1)).median())

    rows = {'measured (AERONET-OC)': {'alpha 443-865': alpha(lambda w: meas.sel(wl_aeronet=w), 443, 865),
                                      'alpha 865-1020': alpha(lambda w: meas.sel(wl_aeronet=w), 865, 1020),
                                      'naot 1020': float(meas.sel(wl_aeronet=1020).median())}}
    weights = {'retrieved (DESE/MACL/WASO)': ds.f3, 'six-model retrieval': ds.f6}
    for name, comp in {**ALTERNATIVES, 'WASO+MACL 70/30': {'WASO_rh0': .7, 'MACL_rh70': .3}}.items():
        weights[name] = xr.DataArray([comp.get(m, 0.) for m in MODELS], coords={'model': MODELS}, dims='model')
    for name, w in weights.items():
        n = (ds.naot_model * w).sum('model')
        resid = n.sel(wl=wla).rename(wl='wl_aeronet') - meas
        rows[name] = {'alpha 443-865': alpha(lambda x: n.sel(wl=x), 443, 865),
                      'alpha 865-1020': alpha(lambda x: n.sel(wl=x), 865, 1020),
                      'naot 1020': float(n.sel(wl=1020).median()),
                      'rms residual 400-1020': float(np.sqrt((resid ** 2).mean('wl_aeronet')).median())}
    print(pd.DataFrame(rows).T.round(3).to_string())

    print('\n    median deficit (x1e-3) per AOT bin, retrieved and six-model mixtures')
    for name in ['retrieved (DESE/MACL/WASO)', 'six-model retrieval', 'MACL']:
        D = (1e3 * (ds.D + (alts[name] - R3))).sel(wl=BANDS).to_pandas()
        print(f'  {name}'); print(D.groupby(bins, observed=True).median().round(2).to_string())

    return ds


# ---------------------------------------------------------------------------------------------------
# hygroscopic growth of the fine mode: single scattering from the OPAC (MOPSMAP) scattering matrices
# ---------------------------------------------------------------------------------------------------
OPAC_DIR = '/DATA/git/vrtc/mopsmap_tbx/scatmat/ascii'
HYGRO_CASES = [('WASO', 0), ('WASO', 50), ('WASO', 70), ('COAV', 70), ('COAV', 80), ('COAV', 90), ('COAV', 95),
               ('URBA', 70), ('URBA', 90), ('MACL', 70), ('MACL', 90), ('DESE', 70)]


def opac_optics(model, rh, wl, directory=OPAC_DIR):
    '''Extinction and scattering cross sections and phase function (mean 1 over the sphere) of an OPAC file.'''
    f = f'{directory}/scat_mat_opac_{model}_rh{rh}_wl{wl:.1f}.csv'
    with open(f) as fh:
        cext = float(fh.readline().split(':')[1])
        csca = float(fh.readline().split(':')[1])
    t = pd.read_csv(f, sep=r'\s+', skiprows=4, names=['angle', 's11', 'ms12', 's22', 's33'])
    theta = np.radians(t.angle.values)
    return cext, csca, t.angle.values, t.s11.values / (np.trapezoid(t.s11.values * np.sin(theta), theta) / 2)


def hygroscopic_test(ds, directory=OPAC_DIR):
    '''
    Single-scattering path reflectance per unit AOT(550), omega tau(wl)/tau(550) P(Theta) / (4 mu0 muv), and
    Angstrom exponents of OPAC models at several relative humidities, at the median geometry of the matchups.
    '''
    mu0, muv = np.cos(np.radians(ds.sza.values)), np.cos(np.radians(ds.vza.values))
    # scattering angle from the glint angle (azimuth convention of the LUT, 0 deg towards the Sun)
    theta = np.degrees(np.arccos(np.cos(np.radians(ds.glint_angle.values)) - 2 * mu0 * muv))
    th, m0, mv = np.nanmedian(theta), np.nanmedian(mu0), np.nanmedian(muv)
    print(f'\n=== 5. hygroscopic growth of the fine mode: OPAC single scattering at the median geometry '
          f'(scattering angle {th:.0f} deg, p10-p90 {np.nanpercentile(theta, 10):.0f}-'
          f'{np.nanpercentile(theta, 90):.0f})')
    rows = {}
    wls = [450, 550, 850, 1000, 1600, 2200]
    for model, rh in HYGRO_CASES:
        c = {wl: opac_optics(model, rh, wl, directory) for wl in wls}
        n = {wl: c[wl][0] / c[550][0] for wl in wls}
        r = {'alpha 450-850': -np.log(n[850] / n[450]) / np.log(850 / 450),
             'alpha 850-1000': -np.log(n[1000] / n[850]) / np.log(1000 / 850),
             'naot 2200': n[2200]}
        for wl in (850, 1600, 2200):
            cext, csca, angle, P = c[wl]
            r[f'SS/AOT {wl}'] = 1e3 * csca / cext * n[wl] * np.interp(th, angle, P) / (4 * m0 * mv)
        r['P(theta) 2200'] = np.interp(th, c[2200][2], c[2200][3])
        rows[f'{model}_rh{rh}'] = r
    print('    measured (AERONET-OC, median): alpha 443-865 = {:.2f}, alpha 865-1020 = {:.2f}'.format(
        float((-np.log(ds.naot_meas.sel(wl_aeronet=865) / ds.naot_meas.sel(wl_aeronet=443)) / np.log(865 / 443)).median()),
        float((-np.log(ds.naot_meas.sel(wl_aeronet=1020) / ds.naot_meas.sel(wl_aeronet=865)) / np.log(1020 / 865)).median())))
    print(pd.DataFrame(rows).T.round(3).to_string())
    lut_waso = (1e3 * ds.Ratm_model.sel(model='WASO_rh0', wl=2190) / ds.aot550).median()
    print(f'    check: path reflectance of WASO_rh0 in the LUT per unit AOT at 2190 nm (Rayleigh and surface '
          f'included) = {float(lut_waso):.1f} x1e-3')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--odir', default=ODIR, help='directory of the matchup files')
    p.add_argument('--cache', default=CACHE, help='directory of the per-site results')
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--max-scenes', type=int, default=None, help='first scenes of each site only (test)')
    p.add_argument('--opac-dir', default=OPAC_DIR, help='directory of the OPAC (MOPSMAP) scattering matrices')
    args = p.parse_args()
    data = {k: compute(args, k) for k in SITES}
    ds = analysis(data)
    hygroscopic_test(ds, args.opac_dir)


if __name__ == '__main__':
    main()
