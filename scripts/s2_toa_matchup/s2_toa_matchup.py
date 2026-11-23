'''
Sentinel-2 / RadCalNet-OC matchups at the top of atmosphere (TOA) over an AERONET-OC site.

For each Sentinel-2 L1C scene (subsets around the AERONET-OC sites, see grstbx) with AERONET-OC
measurements within ``--window`` hours of the overpass:

1. satellite: TOA reflectance statistics of the valid pixels in a box of ``2 * --half-size`` m centred
   on the site, and mean sun and viewing geometry;
2. in situ: AERONET-OC data from the shared database (``aeronet_oc_db``), hyperspectral Rrs
   reconstructed with invRrs (``radcalnet_oc.reconstruction``, nadir view);
3. simulation: radcalnet-oc TOA reflectance of each AERONET-OC measurement, computed for the sun and
   viewing geometry of the satellite scene (the atmosphere and water of the measurement, the geometry
   of the overpass), band-integrated with the spectral responses of the scene (solar-irradiance
   weighted, as the L1C reflectance), without and with the direct sunglint (Cox and Munk, wind speed
   of the measurement), which the look-up tables of radcalnet-oc do not contain.

One netCDF file per scene is cached in ``<odir>/scenes`` (already processed scenes are skipped), and
the matchups are gathered into ``<odir>/<site>_<level>_s2_toa_matchup.nc``:

- dimensions ``time`` (overpass) and ``wl`` (nominal band wavelength, nm);
- ``sat_*`` satellite statistics, ``sim_*`` statistics of the simulations of the measurements of the
  window, ``rec_*`` (dimension ``rec``) the simulation of each measurement.

The slope statistics of the direct sunglint are set with ``--glint-stats`` (``cm_iso`` by default, or the
heavy-tailed ``cm_iso_t`` and ``cm_iso_mix`` of ``radcalnet_oc.Sunglint`` with ``--glint-nu``, ``--glint-f-tail``,
``--glint-sigma2-tail``). When they differ from those of a cached scene, the glint of the scene is recomputed at the
gathering, with the atmospheric factor (gaseous and direct transmittances) of the cached simulation, which does not
depend on the slope statistics; the output file then gets the suffix ``_<glint-stats>`` (and ``_core`` when the
core slope variance ``--glint-s0 + --glint-s1 W`` differs from Cox-Munk). ``--reglint`` only gathers
the cached scenes (no new scene), e.g. to switch the glint statistics without rerunning the simulations.

With ``--roi``, the satellite statistics of the gathered file are computed in the recommended region of interest
of the site (``ROI``: square of half-size 200 m at AAOT/VEIT and 300 m at Section-7/CORO, minus the fixed structure
+ 100 m; README step 9), read again from the subsets at the gathering; the cached scenes keep the box of
``--half-size`` and the simulations are unchanged. Output suffix ``_roi``.

Example::

    python s2_toa_matchup.py --site AAOT --level lev20 --n-jobs 4
    python s2_toa_matchup.py --site AAOT --level lev20 --reglint --glint-stats cm_iso_mix
    python s2_toa_matchup.py --site AAOT --level lev20 --reglint --glint-stats cm_iso_mix --roi
'''
import argparse
import glob
import logging
import os
import re
import time as _time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import aeronet_oc_db
import radcalnet_oc as radoc
from radcalnet_oc.aeronet_oc import aeronet_inputs  # noqa: F401 (radcalnet-oc inputs of a site; used by the other scripts)
from radcalnet_oc.reconstruction import Reconstruction

opj = os.path.join

S2_ROOT = '/data/satellite/Sentinel-2/subset/aeronet/L1C'
from radcalnet_oc import __version__ as RADCALNET_OC_VERSION  # noqa: E402
ODIR = f'/data/radcalnet/matchups/sentinel2/v{RADCALNET_OC_VERSION}'   # one folder per radcalnet_oc version

# radcalnet-oc input variables taken from the AERONET-OC measurements
AERONET_VARS = ['tcwv', 'tco3', 'tcno2', 'pressure', 'wind', 'aot550', 'aot865', 'ang_exp', 'chl', 'sza_aeronet']
RECONSTRUCTION_VARS = ['chl_invrrs', 'spm_norm', 'rmsd']


# ---------------------------------------------------------------------------------------------------
# AERONET-OC: radcalnet_oc.aeronet_oc.aeronet_inputs (imported above)
# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------------------------------
def s2_inventory(tile, root=S2_ROOT):
    '''L1C subsets of a tile: DataFrame (index: sensing time) with satellite, product name and path.'''
    files = sorted(glob.glob(opj(root, tile, '*', '*MSIL1C*.nc')))
    rows = []
    for f in files:
        name = os.path.basename(f)[:-3]
        sat, _, sensing, baseline, orbit, _, processing = name.split('_')
        rows.append(dict(time=pd.Timestamp(sensing), satellite=sat, baseline=baseline, orbit=orbit,
                         processing=processing, product=name, path=f))
    df = pd.DataFrame(rows).set_index('time').sort_index()
    # several processings of the same acquisition: keep the latest
    return df.sort_values('processing').groupby(level=0).tail(1).sort_index()


#: recommended region of interest (scripts/s2_toa_matchup/roi_selection.ipynb, README step 9): square of half-size
#: ``half_size`` m minus the fixed structures dilated by ``buffer`` m (structure pixels of
#: ``<ROI_DIR>/<code>_structures.csv``, written by ``roi_selection.py structures <code>``)
ROI_DIR = '/data/radcalnet/studies/roi_selection'
ROI = {'AAOT': dict(code='VEIT', half_size=200., buffer=100.),
       'Section-7_Platform': dict(code='CORO', half_size=300., buffer=100.)}


def roi_settings(site):
    '''Half-size (m), buffer (m) and structure pixels (N x 2 offsets to the site, m) of the ROI of ``site``.'''
    if site not in ROI:
        raise ValueError(f'no ROI defined for {site} (ROI: {", ".join(ROI)})')
    r = ROI[site]
    xy = pd.read_csv(opj(ROI_DIR, f'{r["code"]}_structures.csv'), comment='#')[['x_m', 'y_m']].values
    return r['half_size'], r['buffer'], xy


def s2_extract(path, lon, lat, half_size=600., swir_max=0.03, cirrus_max=0.01, exclude_xy=None, buffer=0.):
    '''
    TOA reflectance statistics in the box of ``2 * half_size`` m centred on (lon, lat), without the pixels
    within ``buffer`` m of the points ``exclude_xy`` (offsets to the site, m: fixed structures; ROI).

    Two sets of pixels:

    - ``sat_*``: finite reflectance in all the bands, SWIR-1 (1610 nm) < ``swir_max`` (clouds, land,
      strong sunglint) and cirrus (1375 nm) < ``cirrus_max``;
    - ``satg_*``: same without the SWIR-1 threshold, i.e. keeping the sunglint (but also the low clouds:
      to be screened with the simulation, e.g. with the SWIR residual after adding the simulated glint).
    '''
    with xr.open_dataset(path) as ds:
        epsg = int(re.search(r'AUTHORITY\["EPSG","(\d+)"\]\]$', ds.spatial_ref.attrs['crs_wkt']).group(1))
        x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        box = ds.sel(x=slice(x0 - half_size, x0 + half_size), y=slice(y0 + half_size, y0 - half_size)).load()
        srf = ds.SRF.load()
        wl_true = ds.wl_true.load()
    bands = box.bands
    roi = xr.DataArray(np.ones((box.sizes['y'], box.sizes['x']), bool), coords={'y': box.y, 'x': box.x},
                       dims=('y', 'x'))
    if exclude_xy is not None and len(exclude_xy) and roi.size:
        X, Y = np.meshgrid(box.x.values - x0, box.y.values - y0)
        d = np.hypot(X[..., None] - exclude_xy[:, 0], Y[..., None] - exclude_xy[:, 1]).min(-1)
        roi = roi & (d > buffer)
    n_tot = int(roi.sum())
    if n_tot == 0:  # subset made around another site
        return None, None
    cirrus_ok = bands.notnull().all('wl') & (bands.sel(wl=1375) < cirrus_max) & roi
    out = xr.Dataset(coords={'wl': box.wl.values})
    for prefix, valid in [('sat', cirrus_ok & (bands.sel(wl=1610) < swir_max)), ('satg', cirrus_ok)]:
        stacked = bands.where(valid).stack(pix=('y', 'x')).dropna('pix', how='all')
        n_valid = stacked.sizes['pix']
        if n_valid > 0:
            out[prefix + '_median'] = stacked.median('pix')
            out[prefix + '_q25'] = stacked.quantile(0.25, 'pix').drop_vars('quantile')
            out[prefix + '_q75'] = stacked.quantile(0.75, 'pix').drop_vars('quantile')
            out[prefix + '_mean'] = stacked.mean('pix')
            out[prefix + '_std'] = stacked.std('pix')
        else:
            for v in ['median', 'q25', 'q75', 'mean', 'std']:
                out[prefix + '_' + v] = ('wl', np.full(box.sizes['wl'], np.nan))
        out = out.reset_coords(drop=True)
        suffix = '' if prefix == 'sat' else '_glint'
        out['n_pix' + suffix] = n_valid
        out['frac_valid' + suffix] = n_valid / n_tot
    out['wl_true'] = wl_true
    out = out.reset_coords(drop=True)
    out['sza'] = float(box.sza.mean())
    out['vza_band'] = box.vza.mean(['x', 'y'])
    out['vza'] = float(box.vza.mean())
    # relative azimuth of the subsets: (viewing - solar azimuth) % 360; azimuth of the radiative transfer
    # (0 deg: viewing direction towards the Sun, sunglint): (180 - raa) % 360 folded on [0, 180]
    raa = float(np.degrees(np.angle(np.exp(1j * np.radians(box.raa)).mean())) % 360)
    azi = (180. - raa) % 360
    out['raa'] = raa
    out['azi'] = azi if azi <= 180 else 360 - azi
    # angle between the viewing direction and the specular reflection of the Sun (deg)
    s0, sv, a = np.radians(out.sza), np.radians(out.vza), np.radians(out.azi)
    out['glint_angle'] = np.degrees(np.arccos(np.cos(s0) * np.cos(sv) + np.sin(s0) * np.sin(sv) * np.cos(a)))
    out['time'] = box.time.values
    return out, srf


def band_integration(Rtoa, E0, srf):
    '''
    Band TOA reflectances: :math:`\\int R E_0 S d\\lambda / \\int E_0 S d\\lambda`, with the spectral
    responses ``srf`` (wl, wl_hr) interpolated on the wavelengths of the simulation.
    '''
    wl = Rtoa.wl.values
    s = srf.rename(wl_hr='wl_sim', wl='band').interp(wl_sim=wl).fillna(0).values  # (band, wl)
    num = np.trapezoid(Rtoa.values[..., None, :] * E0.values[..., None, :] * s, wl, axis=-1)
    den = np.trapezoid(E0.values[..., None, :] * s, wl, axis=-1)
    return num / den


# ---------------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------------
_RECONSTRUCTION = None
#: wave slope statistics of the sunglint (the wind direction is not measured: isotropic Cox and Munk)
GLINT_STATS = 'cm_iso'
#: parameters of the heavy-tailed statistics (fit on the SWIR excess of the matchups, glint_tail_fit.py)
CM_COEF = (3e-3, 512e-5)  # Cox-Munk mean square slope 0.003 + 0.00512 W
GLINT_KW = {'cm_iso': {}, 'cm_iso_t': {'nu': 7.}, 'cm_iso_mix': {'f_tail': 0.05, 'sigma2_tail': 0.12}}


def glint_kw(args):
    '''Slope statistics and parameters of the command line.'''
    kw = {'cm_iso_t': {'nu': args.glint_nu},
          'cm_iso_mix': {'f_tail': args.glint_f_tail, 'sigma2_tail': args.glint_sigma2_tail}}.get(args.glint_stats, {})
    coef = (args.glint_s0, args.glint_s1)
    if coef != CM_COEF:  # core slope variance s0 + s1 W of the isotropic statistics (Cox-Munk by default)
        kw = {**kw, 'sigma2_coef': coef}
    return args.glint_stats, kw


def output_suffix(stats, kw):
    '''Suffix of the gathered file: '' for cm_iso, _<stats> otherwise, + _core for a non Cox-Munk core.'''
    sfx = '' if stats == 'cm_iso' else f'_{stats}'
    return sfx + ('_core' if 'sigma2_coef' in kw else '')


def surface_glint(sza, vza, azi, wind, stats=GLINT_STATS, kw=None):
    '''Surface sunglint reflectance (Cox and Munk family) for each wind speed.'''
    g = radoc.Sunglint(sza, vza, 180. - azi)
    return np.array([max(g.sunglint(float(w), stats=stats, **(kw or {}))[0], 0.) for w in np.atleast_1d(wind)])


def sunglint_toa(process, ds, sza, vza, azi, stats=GLINT_STATS, kw=None):
    '''
    Direct sunglint at the TOA, not included in the look-up tables of radcalnet-oc (see the tutorial
    *Sunglint contribution to the TOA signal*): Cox and Munk reflectance of the wind speed of each
    measurement, attenuated by the direct transmittance (Rayleigh + aerosol, Beer-Lambert) and by the
    gaseous transmittance of the simulation along the Sun-surface-sensor path.

    :return: TOA sunglint reflectance (time, wl), surface sunglint reflectance (time)
    '''
    full_wl = process.full_wl
    mu0, muv = np.cos(np.radians(sza)), np.cos(np.radians(vza))
    tau_r = process.lut.rot.interp(wl=full_wl, kwargs={'fill_value': 'extrapolate'}) * ds.pressure / 1013.25
    tau_a = process.lut.aot_lut.interp(aot_ref=ds.aot550).interp(wl=full_wl, method='quadratic')
    Tdir = np.exp(-(tau_r + tau_a) * (1 / mu0 + 1 / muv))
    Tg = (process.Tg_d.sel(wl=full_wl) * process.Tg_u.sel(wl=full_wl)).squeeze(drop=True)
    wind = process.wind.values
    Rg = surface_glint(sza, vza, azi, wind, stats, kw)
    Rg = xr.DataArray(Rg, coords={'time': ds.time}, dims='time')
    Rg_toa = (Tg * Tdir * Rg).reset_coords(drop=True).transpose('time', 'wl')
    return Rg_toa, Rg


def simulate(records, sat, srf, glint=(GLINT_STATS, {})):
    '''radcalnet-oc simulation of the measurements ``records`` for the geometry of the scene ``sat``.'''
    global _RECONSTRUCTION
    if _RECONSTRUCTION is None:
        _RECONSTRUCTION = Reconstruction()
    hyp = _RECONSTRUCTION.reconstruct(records.Rrs_Lwn, records.sza_aeronet, chl=records.chl)
    hyp = hyp.where(hyp.success, drop=True)
    if hyp.sizes['time'] == 0:
        return None
    records = records.sel(time=hyp.time)

    ds = records[AERONET_VARS + ['naot']].copy()
    ds['Rrs'] = hyp.Rrs_nadir
    ds['chl_invrrs'] = hyp.chl
    ds['spm_norm'] = hyp.spm_norm
    ds['rmsd'] = hyp.rmsd
    # geometry and date of the overpass
    t = pd.Timestamp(sat.time.values)
    ds['sza'] = xr.full_like(ds.aot550, float(sat.sza))
    ds['day_of_year'] = xr.full_like(ds.aot550, t.dayofyear)

    single = ds.sizes['time'] == 1
    if single:
        # radcalnet_oc.Process squeezes a time dimension of size 1: duplicate the measurement
        ds = xr.concat([ds, ds.assign_coords(time=ds.time + np.timedelta64(1, 's'))], 'time')
    naot = ds.naot.dropna('wl_aeronet', how='all')
    process = radoc.Process(ds, vza=xr.DataArray([float(sat.vza)], coords={'vza': [float(sat.vza)]}, dims='vza'),
                            azi=xr.DataArray([float(sat.azi)], coords={'azi': [float(sat.azi)]}, dims='azi'))
    aerosol = radoc.Aerosol(naot, process.lut.naot_lut)
    aerosol.process()
    process.execute(aerosol_combination=aerosol.model_db.aerosol_combination)
    db = process.radcalnet_db
    Rtoa = db.Rtoa.transpose('time', 'wl')
    E0 = db.E0.transpose('time', 'wl')
    Rglint, Rg = sunglint_toa(process, ds, float(sat.sza), float(sat.vza), float(sat.azi), *glint)
    if single:
        ds, Rtoa, E0 = ds.isel(time=[0]), Rtoa.isel(time=[0]), E0.isel(time=[0])
        Rglint, Rg = Rglint.isel(time=[0]), Rg.isel(time=[0])
    rec = xr.Dataset(coords={'time': ds.time.values, 'wl': sat.wl.values})
    rec['Rtoa'] = (('time', 'wl'), band_integration(Rtoa, E0, srf))
    rec['Rglint'] = (('time', 'wl'), band_integration(Rglint, E0, srf))
    rec['Rg_surface'] = ('time', Rg.values)
    for v in AERONET_VARS + RECONSTRUCTION_VARS:
        rec[v] = ds[v]
    # spectra at 1 nm (Gaussian 3 nm FWHM), for the plots
    spectral = radoc.Spectral(np.arange(350, 2500, 1), 3)
    rec['Rtoa_spectrum'] = spectral.convolve2(Rtoa.rename('Rtoa'), expon=2)['Rtoa'].rename(wl='wl_spectrum')
    rec['Rglint_spectrum'] = spectral.convolve2(Rglint.rename('Rglint'), expon=2)['Rglint'].rename(
        wl='wl_spectrum')
    return rec.reset_coords(drop=True)


def process_scene(product, path, site, lon, lat, records, args):
    filename = opj(args.odir, 'scenes', f'{site}_{args.level}', product + '.nc')
    if os.path.exists(filename) and not args.overwrite:
        return filename, 'cached'
    t0 = _time.time()
    try:
        sat, srf = s2_extract(path, lon, lat, args.half_size, args.swir_max, args.cirrus_max)
        if sat is None:
            status = 'site outside the subset'
            sat, rec = xr.Dataset(), None
        elif int(sat.n_pix_glint) == 0:
            status = 'no valid pixel'
            rec = None
        else:
            rec = simulate(records, sat, srf, glint_kw(args))
            status = 'no valid in situ' if rec is None else f'{rec.sizes["time"]} measurements'
    except Exception as e:  # not cached: retried at the next run
        logging.exception(f'{product}: failed')
        return None, f'error: {e!r}'
    scene = sat.copy()
    scene['product'] = product
    if rec is not None:
        t = pd.Timestamp(sat.time.values)
        rec = rec.rename(time='rec_time').rename({v: 'rec_' + v for v in rec.data_vars})
        rec['rec_dt'] = ('rec_time', ((rec.rec_time.values - t.to_datetime64()) / np.timedelta64(1, 'm')))
        rec['rec_dt'].attrs['unit'] = 'min'
        scene = xr.merge([scene, rec], compat='override', join='outer')
    stats, kw = glint_kw(args)
    scene.attrs.update(site=site, level=args.level, status=status, glint_stats=stats,
                       glint_kw=repr(sorted(kw.items())))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scene.to_netcdf(filename + '.tmp')
    os.replace(filename + '.tmp', filename)
    logging.info(f'{product}: {status} ({_time.time() - t0:.0f} s)')
    return filename, status


# ---------------------------------------------------------------------------------------------------
# gathering
# ---------------------------------------------------------------------------------------------------
def reglint(s, stats, kw):
    '''
    Direct sunglint of a cached scene recomputed with other slope statistics: the TOA glint is the surface glint
    times an atmospheric factor (gaseous and direct transmittances) that does not depend on the statistics, taken
    from the cached simulation (median of the other measurements of the scene when the cached surface glint
    underflows).
    '''
    old = s.rec_Rg_surface
    ok = old > 1e-30
    factor = (s.rec_Rglint / old).where(ok)
    factor = factor.fillna(factor.median('rec_time'))
    factor_spec = (s.rec_Rglint_spectrum / old).where(ok)
    factor_spec = factor_spec.fillna(factor_spec.median('rec_time'))
    new = xr.DataArray(surface_glint(float(s.sza), float(s.vza), float(s.azi), s.rec_wind.values, stats, kw),
                       coords=old.coords, dims=old.dims)
    s['rec_Rg_surface'] = new
    s['rec_Rglint'] = (factor * new).transpose(*s.rec_Rglint.dims)
    s['rec_Rglint_spectrum'] = (factor_spec * new).astype('float32').transpose(*s.rec_Rglint_spectrum.dims)
    return s


#: MSI bands of the absorption (945 nm water vapour, 1375 nm cirrus), excluded from the glint uncertainty test
ABSORPTION_BANDS = [945, 1375]


def glint_unc_scene(s):
    """
    Uncertainty of the TOA glint of a scene (radcalnet_oc.coxmunk.glint_uncertainty: half range of the plausible
    slope statistics at the wind +/- its uncertainty), per measurement with the atmospheric factor of the cached
    simulation, median over the measurements (wl).
    """
    old = s.rec_Rg_surface
    ok = old > 1e-30
    factor = (s.rec_Rglint / old).where(ok)
    factor = factor.fillna(factor.median('rec_time'))
    u = np.array([radoc.coxmunk.glint_uncertainty(float(s.sza), float(s.vza), 180. - float(s.azi), float(w))[0]
                  for w in s.rec_wind.values])
    return (factor * xr.DataArray(u, coords=old.coords, dims=old.dims)).median('rec_time')


#: satellite statistics of a scene (s2_extract), replaced by restat_scene
SAT_VARS = [f'{p}_{v}' for p in ('sat', 'satg') for v in ('median', 'q25', 'q75', 'mean', 'std')] + \
    ['n_pix', 'frac_valid', 'n_pix_glint', 'frac_valid_glint']


def make_restat(site, tile, args):
    '''Function recomputing the satellite statistics of a cached scene in the ROI of ``site`` (``--roi``), from
    the L1C subset of the scene; the simulations are kept.'''
    half, buffer, xy = roi_settings(site)
    paths = s2_inventory(tile, args.s2_root).set_index('product').path
    info = aeronet_oc_db.site_info(level=args.level).loc[site]
    lon, lat = float(info.longitude), float(info.latitude)

    def restat(s):
        prod = str(s['product'].values)
        if prod not in paths.index:
            return None
        sat, _ = s2_extract(paths[prod], lon, lat, half, args.swir_max, args.cirrus_max, xy, buffer)
        if sat is None:
            return None
        for v in SAT_VARS:
            s[v] = sat[v]
        return s
    restat.attrs = dict(roi_half_size_m=half, roi_buffer_m=buffer, roi_structures=f'{ROI[site]["code"]}_structures.csv',
                        roi=f'square of half-size {half:g} m minus the fixed structures + {buffer:g} m')
    return restat


def gather(files, site, level, glint=(GLINT_STATS, {}), restat=None):
    stats, kw = glint
    scenes, recs = [], []
    n_reglint = 0
    for f in files:
        with xr.open_dataset(f) as s:
            s = s.load()
        if 'rec_Rtoa' not in s:
            continue
        if restat is not None:
            s = restat(s)
            if s is None:
                continue
        if (s.attrs.get('glint_stats', 'cm_iso'), s.attrs.get('glint_kw', '[]')) != (stats, repr(sorted(kw.items()))):
            s = reglint(s, stats, kw)
            n_reglint += 1
        t = s.time.values
        r = s[[v for v in s.data_vars if v.startswith('rec_')]].rename(rec_time='rec')
        r = r.assign_coords(rec_time=('rec', r.rec.values)).drop_vars('rec')
        r['rec_scene'] = ('rec', np.full(r.sizes['rec'], t))
        recs.append(r)
        sim = s.rec_Rtoa
        inear = int(np.argmin(np.abs(s.rec_dt.values)))
        sc = s.drop_dims('rec_time')
        sc['sim_median'] = sim.median('rec_time')
        sc['sim_q25'] = sim.quantile(0.25, 'rec_time').drop_vars('quantile')
        sc['sim_q75'] = sim.quantile(0.75, 'rec_time').drop_vars('quantile')
        sc['sim_mean'] = sim.mean('rec_time')
        sc['sim_std'] = sim.std('rec_time')
        sc['sim_count'] = sim.sizes['rec_time']
        # simulation of the measurement nearest to the overpass
        sc['sim_nearest'] = sim.isel(rec_time=inear, drop=True)
        sc['dt_nearest'] = float(s.rec_dt[inear])
        sc['sim_spectrum_median'] = s.rec_Rtoa_spectrum.median('rec_time')
        # with the direct sunglint
        simg = s.rec_Rtoa + s.rec_Rglint
        sc['simg_median'] = simg.median('rec_time')
        sc['simg_q25'] = simg.quantile(0.25, 'rec_time').drop_vars('quantile')
        sc['simg_q75'] = simg.quantile(0.75, 'rec_time').drop_vars('quantile')
        sc['simg_std'] = simg.std('rec_time')
        sc['simg_nearest'] = simg.isel(rec_time=inear, drop=True)
        sc['glint_median'] = s.rec_Rglint.median('rec_time')
        sc['Rg_surface'] = s.rec_Rg_surface.median('rec_time')
        sc['glint_spectrum_median'] = s.rec_Rglint_spectrum.median('rec_time')
        sc['glint_unc'] = glint_unc_scene(s)
        for v in AERONET_VARS + RECONSTRUCTION_VARS:
            sc[v] = s['rec_' + v].median('rec_time')
        sc = sc.drop_vars('time', errors='ignore')
        scenes.append(sc.expand_dims(time=[t]))
    out = xr.concat(scenes, 'time', data_vars='all', coords='minimal', compat='override', join='outer')
    rec = xr.concat(recs, 'rec', data_vars='all', coords='minimal', compat='override', join='outer')
    if n_reglint:
        logging.info(f'glint of {n_reglint} cached scenes recomputed with {stats} {kw}')
    # sunglint flags of each scene (radcalnet_oc.GLINT_FLAGS): fraction of the direct glint in the simulated TOA
    # reflectance with glint at 865 nm (median of the measurements) and glint angle
    out['glint_fraction'] = (out.glint_median / out.simg_median).sel(wl=865, drop=True)
    out['glint_fraction'].attrs = {'description': 'glint_median / simg_median at 865 nm'}
    # glint uncertainty relative to the simulated TOA reflectance with glint, and spectral flag glint_uncertain
    # (per band) above radcalnet_oc.coxmunk.GLINT_UNC_THRESHOLD
    out['glint_unc'].attrs = {'description': 'uncertainty of the TOA glint: half range of radcalnet_oc.coxmunk.'
                                             'GLINT_ENSEMBLE at the wind +/- WIND_UNC (median of the measurements)'}
    ratio = out.glint_unc / out.simg_median
    out['glint_uncertain'] = (ratio > radoc.coxmunk.GLINT_UNC_THRESHOLD).astype(np.uint8)
    out['glint_uncertain'].attrs = radoc.coxmunk.uncertain_attrs()
    rel = ratio.drop_sel(wl=ABSORPTION_BANDS)
    out['glint_unc_rel'] = rel.max('wl')
    out['glint_unc_rel'].attrs = {'description': 'largest glint_unc / simg_median over the bands (without '
                                                 f'{ABSORPTION_BANDS} nm)'}
    out['glint_unc_wl'] = rel.fillna(-1).idxmax('wl')
    out['glint_flags'] = (('time',), radoc.glint_flags(out.glint_fraction.values, out.glint_angle.values))
    out['glint_flags'].attrs = radoc.coxmunk.flag_attrs()
    out = xr.merge([out, rec], compat='override', join='outer')
    out.attrs = dict(title=f'Sentinel-2 / RadCalNet-OC TOA matchups, {site} {level}',
                     site=site, level=level,
                     created=pd.Timestamp.now('UTC').isoformat(),
                     radcalnet_oc=radoc.process.package_version(), aeronet_oc_db=aeronet_oc_db.__version__,
                     glint_stats=stats, glint_kw=repr(sorted(kw.items())),
                     comment='sat_*: satellite statistics in the box around the site (clear pixels, SWIR '
                             'threshold); satg_*: same without the SWIR threshold (sunglint kept); sim_*: '
                             'statistics of the radcalnet-oc simulations of the AERONET-OC measurements '
                             'within the time window, for the geometry of the overpass, without sunglint; '
                             'simg_*: same with the direct sunglint; glint_*: TOA sunglint; rec_*: each '
                             'simulation')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--site', default='AAOT')
    p.add_argument('--level', default='lev20')
    p.add_argument('--tile', default=None, help='Sentinel-2 tile (default: aeronet_oc_db site table)')
    p.add_argument('--start', default='2017-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--window', type=float, default=2., help='half time window around the overpass (h)')
    p.add_argument('--half-size', type=float, default=600., help='half size of the box (m)')
    p.add_argument('--roi', action='store_true',
                   help='satellite statistics of the gathered file in the recommended ROI of the site (ROI: square '
                        'minus the fixed structures + buffer), recomputed from the subsets for the cached scenes '
                        '(the cache keeps the box of --half-size); output suffix _roi')
    p.add_argument('--swir-max', type=float, default=0.03)
    p.add_argument('--cirrus-max', type=float, default=0.01)
    p.add_argument('--s2-root', default=S2_ROOT)
    p.add_argument('--odir', default=ODIR)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='process the first N scenes only (tests)')
    p.add_argument('--glint-stats', default=GLINT_STATS, choices=list(GLINT_KW),
                   help='wave slope statistics of the direct sunglint')
    p.add_argument('--glint-nu', type=float, default=GLINT_KW['cm_iso_t']['nu'], help='nu of cm_iso_t')
    p.add_argument('--glint-f-tail', type=float, default=GLINT_KW['cm_iso_mix']['f_tail'], help='f_tail of cm_iso_mix')
    p.add_argument('--glint-sigma2-tail', type=float, default=GLINT_KW['cm_iso_mix']['sigma2_tail'],
                   help='sigma2_tail of cm_iso_mix')
    p.add_argument('--glint-s0', type=float, default=CM_COEF[0],
                   help='core mean square slope s0 + s1 W of the isotropic statistics: s0')
    p.add_argument('--glint-s1', type=float, default=CM_COEF[1], help='s1 (Cox-Munk 0.00512; 0.0035 fitted)')
    p.add_argument('--reglint', action='store_true',
                   help='only gather the cached scenes (glint recomputed with --glint-stats), no new scene')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger('radcalnet_oc').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    stats, kw = glint_kw(args)
    suffix = output_suffix(stats, kw) + ('_roi' if args.roi else '')
    ofile = opj(args.odir, f'{args.site}_{args.level}_s2_toa_matchup{suffix}.nc')
    if args.reglint:
        files = sorted(glob.glob(opj(args.odir, 'scenes', f'{args.site}_{args.level}', '*.nc')))
        with xr.open_dataset(opj(args.odir, f'{args.site}_{args.level}_s2_toa_matchup.nc')) as ref:
            # settings and provenance of the simulations of the reference file (only the glint is recomputed)
            attrs = {k: ref.attrs[k] for k in ['window_h', 'half_size_m', 'swir_max', 'cirrus_max', 'tile',
                                               'radcalnet_oc', 'aeronet_oc_db', 'created'] if k in ref.attrs}
        restat = make_restat(args.site, args.tile or attrs['tile'], args) if args.roi else None
        out = gather(files, args.site, args.level, (stats, kw), restat)
        if restat is not None:
            attrs.update(restat.attrs)
            attrs.pop('half_size_m', None)
        attrs['glint_recomputed'] = (f'{pd.Timestamp.now("UTC").isoformat()}, radcalnet_oc '
                                     f'{radoc.process.package_version()}')
        out.attrs.update(attrs)
        out.to_netcdf(ofile)
        logging.info(f'{out.sizes["time"]} matchups, {out.sizes["rec"]} simulations -> {ofile}')
        return

    info = aeronet_oc_db.site_info(level=args.level).loc[args.site]
    tile = args.tile or info.s2_tile
    lon, lat = float(info.longitude), float(info.latitude)
    insitu = aeronet_inputs(args.site, args.level).load()

    inv = s2_inventory(tile, args.s2_root)
    inv = inv.loc[args.start:args.end]
    window = pd.Timedelta(hours=args.window)
    tasks = []
    for t, row in inv.iterrows():
        rec = insitu.sel(time=slice(t - window, t + window))
        rec = rec.where(rec.aot550.notnull() & (rec.Rrs_Lwn.count('wl') >= 5), drop=True)
        if rec.sizes['time'] > 0:
            tasks.append((row['product'], row.path, rec))
    if args.limit:
        tasks = tasks[:args.limit]
    logging.info(f'{args.site} {args.level}, tile {tile}: {len(inv)} scenes, {len(tasks)} with AERONET-OC '
                 f'data within {args.window:g} h')

    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(process_scene)(prod, path, args.site, lon, lat, rec, args) for prod, path, rec in tasks)
    else:
        results = [process_scene(prod, path, args.site, lon, lat, rec, args) for prod, path, rec in tasks]

    errors = [st for f, st in results if f is None]
    if errors:
        logging.warning(f'{len(errors)} scene(s) failed (see the log), they will be retried at the next run')
    restat = make_restat(args.site, tile, args) if args.roi else None
    out = gather([f for f, _ in results if f is not None], args.site, args.level, (stats, kw), restat)
    out.attrs.update(window_h=args.window, swir_max=args.swir_max, cirrus_max=args.cirrus_max, tile=tile,
                     **(restat.attrs if restat is not None else {'half_size_m': args.half_size}))
    out.to_netcdf(ofile)
    logging.info(f'{out.sizes["time"]} matchups, {out.sizes["rec"]} simulations -> {ofile}')


if __name__ == '__main__':
    main()
