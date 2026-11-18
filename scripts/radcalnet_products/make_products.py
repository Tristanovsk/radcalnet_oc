'''RadCalNet-OC daily products of an AERONET-OC site: simulated TOA reflectance at nadir (10 nm), with the direct
sunglint and its flags, and a copy without the glinted times.

For each AERONET-OC measurement of the site (``aeronet_oc_db``, level ``--level``, from ``--start`` to ``--end``):
hyperspectral Rrs at nadir reconstructed with invRrs (``radcalnet_oc.reconstruction``) from Rrs_Lwn, aerosol mixture
retrieved from the spectral AOT (``radcalnet_oc.Aerosol``), simulation at nadir (``radcalnet_oc.Process``, settings
of the installed version), direct sunglint at the wind of the station (``Process.compute_sunglint``, slope statistics
``--glint-stats``). The simulation is run per month (restart: months already written are skipped), then written as
one file per day, convolved with super-Gaussian responses of 10 nm FWHM every 10 nm (350-2490 nm, exponent 3), as the
RadCalNet-OC products before version 1.2:

    <odir>/<site>/<yyyy>/<mm>/<dd>/<site>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc

Variables: as before (Rtoa, Ed, E0, Rrs, sza, aerosol and gas inputs, aerosol_combination) and, new in version 1.2,
``wind``, ``Rglint`` (TOA direct glint at nadir, not included in ``Rtoa``), ``Rtoa_glint`` (``Rtoa + Rglint``),
``Rglint_unc`` (spread of the plausible slope statistics, wind +/- 1.5 m s-1), ``glint_uncertain`` (per band,
``Rglint_unc / Rtoa_glint`` > 5 %), ``glint_fraction`` (865 nm), ``glint_angle`` and ``glint_flags`` (bitmask
``radcalnet_oc.GLINT_FLAGS``: low >= 1 %, moderate >= 10 %, high >= 50 % of the TOA reflectance at 865 nm, glint angle
< 30 deg). The copy ``<odir>/<site>_glint_filtered/...`` keeps only the times without the bits of
``--filter-flags`` (moderate and near-specular by default); with ``--filter-uncertain``, also without
``glint_uncertain`` in 400-900 nm (strict: at nadir, this removes most of the summer times of the platforms).

Usage:
    python make_products.py --site AAOT --level lev20 --odir /data/radcalnet/data_v1.2.0/radcalnet --name VEIT
    python make_products.py --site Section-7_Platform --level lev15 --start 2024-08-19
'''
import argparse
import logging
import os
import sys
import time as _time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import radcalnet_oc as radoc
from radcalnet_oc.coxmunk import GLINT_FLAGS
from radcalnet_oc.reconstruction import Reconstruction

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 's2_toa_matchup'))
from s2_toa_matchup import aeronet_inputs  # noqa: E402

opj = os.path.join
logger = logging.getLogger('make_products')

ODIR = '/data/radcalnet/data_v1.2.0/radcalnet'
CENTRAL_WL = np.arange(350, 2500, 10)
FWHM, EXPON = 10., 3
INPUT_VARS = ['tcwv', 'tco3', 'tcno2', 'pressure', 'wind', 'aot550', 'aot865', 'ang_exp']
GLINT_SCALARS = ['glint_angle', 'glint_fraction', 'glint_flags', 'glint_unc_rel', 'glint_unc_wl', 'Rg_surface']
UNC_RANGE = (400, 900)   # glint_uncertain checked in this range for the filtered copy (nm)


def monthly_file(odir, site, level, year, month):
    return opj(odir, 'monthly', site, f'{site}_radcalnet-oc_{level}_{year}_{month:02}_full_spectral.nc')


def daily_file(odir, site, level, day):
    return opj(odir, site, f'{day:%Y}', f'{day:%m}', f'{day:%d}',
               f'{site}_radcalnet-oc_{level}_{day:%Y%m%d}_inc10nm_res10nm.nc')


def simulate_month(records, reconstruction, glint_stats):
    '''Full-spectral simulation at nadir of the measurements of one month (None if fewer than 2 are valid).'''
    records = records.where(records.aot550.notnull() & records.naot.notnull().any('wl_aeronet'), drop=True)
    records = records.where(records.Rrs_Lwn.notnull().sum('wl') >= 4, drop=True)
    if records.sizes['time'] < 2:
        return None
    hyp = reconstruction.reconstruct(records.Rrs_Lwn, records.sza_aeronet, chl=records.chl)
    hyp = hyp.where(hyp.success, drop=True)
    if hyp.sizes['time'] < 2:  # radcalnet_oc.Process squeezes a time dimension of size 1
        return None
    records = records.sel(time=hyp.time)
    ds = records[INPUT_VARS + ['naot']].copy()
    ds['Rrs'] = hyp.Rrs_nadir
    ds['Rrs'].attrs = {'unit': 'sr-1', 'description': 'remote-sensing reflectance at nadir reconstructed with '
                       'invRrs from the AERONET-OC Rrs_Lwn'}
    ds['sza'] = records.sza_aeronet
    ds['day_of_year'] = ds.time.dt.dayofyear
    nadir = xr.DataArray([0.], coords={'vza': [0.]}, dims='vza')
    process = radoc.Process(ds, vza=nadir, azi=xr.DataArray([0.], coords={'azi': [0.]}, dims='azi'))
    aerosol = radoc.Aerosol(ds.naot.dropna('wl_aeronet', how='all'), process.lut.naot_lut)
    aerosol.process()
    process.execute(aerosol_combination=aerosol.model_db.aerosol_combination, sunglint=glint_stats)
    db = process.radcalnet_db.squeeze(drop=True)
    for v in ['chl', 'rmsd', 'spm_norm']:
        if v in hyp:
            db[v + '_invrrs'] = hyp[v]
    db['chl_invrrs'].attrs = {'unit': 'mg m-3', 'description': 'chlorophyll-a concentration of the invRrs fit'}
    db['rmsd_invrrs'].attrs = {'unit': 'sr-1', 'description': 'rmsd of the invRrs fit with the AERONET-OC bands'}
    return db


def to_10nm(db):
    '''Convolution of the spectral variables with 10 nm responses every 10 nm (Rrs interpolated).'''
    spectral = radoc.Spectral(CENTRAL_WL, FWHM)
    out = []
    for name, values in db.data_vars.items():
        if name == 'glint_uncertain':   # recomputed on the 10 nm bands below
            continue
        if 'wl' not in values.dims:
            out.append(values)
        elif name.startswith('Rrs'):
            out.append(values.interp(wl=CENTRAL_WL))
        else:
            out.append(spectral.convolve2(values.rename(name), expon=EXPON)[name])
    out = xr.merge(out, combine_attrs='override', compat='override', join='exact')
    # spectral flag of the glint uncertainty on the 10 nm bands
    unc = (out.Rglint_unc / out.Rtoa_glint > float(db.glint_uncertain.attrs.get('threshold', 0.05)))
    out['glint_uncertain'] = unc.astype(np.uint8)
    out['glint_uncertain'].attrs = db.glint_uncertain.attrs
    for name in out.data_vars:
        if name in db:
            out[name].attrs = db[name].attrs
    out.attrs = db.attrs
    assert out.sizes['wl'] == CENTRAL_WL.size, out.sizes
    return out


def glint_ok(day, filter_flags, uncertain=False):
    '''Times without the glint flags ``filter_flags`` (and without glint_uncertain in UNC_RANGE).'''
    mask = sum(GLINT_FLAGS[f] for f in filter_flags)
    ok = (day.glint_flags.astype(int) & mask) == 0
    if uncertain:
        ok = ok & ~day.glint_uncertain.sel(wl=slice(*UNC_RANGE)).any('wl')
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', required=True, help='AERONET-OC site (aeronet_oc_db name)')
    p.add_argument('--name', help='site name of the products (default: --site)')
    p.add_argument('--level', default='lev20', choices=['lev15', 'lev20'])
    p.add_argument('--start', default='2017-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--odir', default=ODIR)
    p.add_argument('--glint-stats', default='cm_iso_mix')
    p.add_argument('--filter-flags', nargs='*', default=['glint_moderate', 'glint_near_specular'],
                   choices=list(GLINT_FLAGS))
    p.add_argument('--filter-uncertain', action='store_true',
                   help='also remove the times with glint_uncertain in 400-900 nm from the filtered copy')
    p.add_argument('--max-times', type=int, default=400,
                   help='largest number of measurements simulated at once (memory, about 2 GB per 400)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    radoc.set_logging('WARNING')
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s | %(message)s')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    name = args.name or args.site

    inputs = aeronet_inputs(args.site, args.level).sel(time=slice(args.start, args.end))
    logger.info(f'{args.site} {args.level}: {inputs.sizes["time"]} measurements '
                f'{pd.Timestamp(inputs.time.values[0]):%Y-%m-%d} - {pd.Timestamp(inputs.time.values[-1]):%Y-%m-%d}')
    reconstruction = Reconstruction()
    t_all = _time.time()
    n_days = n_kept = n_times = 0
    for key, records in inputs.groupby(inputs.time.dt.strftime('%Y-%m')):
        year, month = (int(x) for x in key.split('-'))
        mfile = monthly_file(args.odir, name, args.level, year, month)
        if os.path.exists(mfile) and not args.overwrite:
            db = xr.open_dataset(mfile).load()
        else:
            t0 = _time.time()
            try:
                # chunks of at most --max-times measurements (memory: full spectral grid per time)
                n = records.sizes['time']
                parts = [simulate_month(records.isel(time=slice(i, i + args.max_times)), reconstruction,
                                        args.glint_stats) for i in range(0, n, args.max_times)]
                parts = [x for x in parts if x is not None]
                db = xr.concat(parts, 'time', data_vars='minimal', coords='minimal', compat='override',
                               combine_attrs='override') if parts else None
            except Exception:
                logger.exception(f'{year}-{month:02}: failed')
                continue
            if db is None:
                logger.info(f'{year}-{month:02}: fewer than 2 valid measurements')
                continue
            db.attrs.update(site=name, aeronet_oc_site=args.site, aeronet_oc_level=args.level, geometry='nadir view')
            os.makedirs(os.path.dirname(mfile), exist_ok=True)
            db.to_netcdf(mfile + '.tmp')
            os.replace(mfile + '.tmp', mfile)
            logger.info(f'{year}-{month:02}: {db.sizes["time"]} times ({_time.time() - t0:.0f} s)')
        db10 = to_10nm(db)
        for day, ds_day in db10.groupby(db10.time.dt.floor('D')):
            day = pd.Timestamp(day)
            n_days += 1
            n_times += ds_day.sizes['time']
            for odir, sub in [(args.odir, ds_day), (args.odir + '_glint_filtered', None)]:
                if sub is None:
                    sub = ds_day.isel(time=np.flatnonzero(glint_ok(ds_day, args.filter_flags, args.filter_uncertain).values))
                    sub.attrs['glint_filter'] = f'times with glint_flags {"/".join(args.filter_flags)} removed' + (
                        f', and with glint_uncertain in {UNC_RANGE[0]}-{UNC_RANGE[1]} nm' if args.filter_uncertain
                        else '')
                    n_kept += sub.sizes['time']
                    if sub.sizes['time'] == 0:
                        continue
                f = daily_file(odir, name, args.level, day)
                if os.path.exists(f) and not args.overwrite:
                    continue
                os.makedirs(os.path.dirname(f), exist_ok=True)
                sub.to_netcdf(f)
    logger.info(f'done: {n_days} days, {n_times} times, {n_kept} kept after the glint filter '
                f'({_time.time() - t_all:.0f} s)')


if __name__ == '__main__':
    main()
