'''
RadCalNet-OC products of an AERONET-OC site: simulated TOA reflectance at nadir every 10 nm, with the direct
sunglint and its flags, one file per day, and a copy without the glinted times.

For each AERONET-OC measurement of the site (:py:func:`radcalnet_oc.aeronet_oc.aeronet_inputs`): hyperspectral
Rrs at nadir reconstructed with invRrs (:py:class:`~radcalnet_oc.reconstruction.Reconstruction`), aerosol mixture
retrieved from the spectral AOT (:py:class:`~radcalnet_oc.kernel.Aerosol`), simulation at nadir
(:py:class:`~radcalnet_oc.process.Process`, settings of the installed version) with the direct sunglint at the
wind of the station (``Process.execute(sunglint=...)``). The simulation is run per month and cached (restart:
the months already simulated are read from the cache), then convolved with super-Gaussian responses of 10 nm FWHM
every 10 nm (350-2490 nm, exponent 3) and written as one file per day::

    <odir>/daily/<name>/<yyyy>/<mm>/<dd>/<name>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc
    <odir>/daily_glint_filtered/<name>/<yyyy>/<mm>/<dd>/<name>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc

The ``daily_glint_filtered`` copy keeps only the times without the glint flags ``filter_flags`` (moderate and
near-specular glint by default), and optionally without ``glint_uncertain`` in 400-900 nm.

Python::

    from radcalnet_oc.products import make_products
    make_products('AAOT', 'lev20', name='VEIT', start='2025-07-01', end='2025-07-31', odir='products')

Command line (same options)::

    radcalnet_oc-products --site AAOT --level lev20 --name VEIT --start 2025-07-01 --end 2025-07-31 --odir products
'''
import argparse
import logging
import os
import time as _time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .aeronet_oc import aeronet_inputs
from .coxmunk import GLINT_FLAGS
from .kernel import Aerosol
from .lut import Spectral
from .process import Process

opj = os.path.join
logger = logging.getLogger(__name__)

#: central wavelengths of the 10 nm bands of the products (nm)
CENTRAL_WL = np.arange(350, 2500, 10)
#: full width at half maximum of the super-Gaussian responses of the 10 nm bands (nm)
FWHM = 10.
#: exponent of the super-Gaussian responses of the 10 nm bands
EXPON = 3
#: AERONET-OC inputs of the simulation (see :py:func:`~radcalnet_oc.aeronet_oc.aeronet_inputs`)
INPUT_VARS = ['tcwv', 'tco3', 'tcno2', 'pressure', 'wind', 'aot550', 'aot865', 'ang_exp']
#: glint flags removed from the ``daily_glint_filtered`` copy by default
FILTER_FLAGS = ('glint_moderate', 'glint_near_specular')
#: spectral range (nm) where ``glint_uncertain`` is checked for the filtered copy (``filter_uncertain``)
UNC_RANGE = (400, 900)
#: first date of the products
START = '2017-01-01'


def monthly_file(cache, site, level, year, month):
    '''Path of the cached full-spectral simulation of one month.'''
    return opj(cache, site, f'{site}_radcalnet-oc_{level}_{year}_{month:02}_full_spectral.nc')


def daily_file(odir, site, level, day):
    '''Path of the daily file of ``day`` (:py:class:`pandas.Timestamp`) in ``odir`` (``<odir>/<site>/<yyyy>/...``).'''
    return opj(odir, site, f'{day:%Y}', f'{day:%m}', f'{day:%d}',
               f'{site}_radcalnet-oc_{level}_{day:%Y%m%d}_inc10nm_res10nm.nc')


def simulate_month(records, reconstruction, glint_stats='cm_iso_mix', vza=None, azi=None, return_process=False):
    '''
    Full-spectral simulation at nadir of AERONET-OC measurements (typically one month), with the direct sunglint.

    The measurements without aerosol data or with fewer than 4 valid Rrs bands are discarded, then those whose
    invRrs fit failed.

    :param records: output of :py:func:`~radcalnet_oc.aeronet_oc.aeronet_inputs` (time selection)
    :param reconstruction: :py:class:`~radcalnet_oc.reconstruction.Reconstruction`
    :param glint_stats: slope statistics of the sunglint (``Process.execute(sunglint=...)``)
    :param vza: viewing zenith angles (:py:class:`xarray.DataArray` of dimension ``vza``; default nadir, as the
        products)
    :param azi: relative azimuth angles (:py:class:`xarray.DataArray` of dimension ``azi``; default 0)
    :param return_process: also return the :py:class:`~radcalnet_oc.process.Process` (e.g. for
        :py:func:`~radcalnet_oc.uncertainty.uncertainty_budget`)
    :return: ``Process.radcalnet_db`` with the dimensions of size 1 squeezed (time, wl at nadir) and the invRrs
        outputs (``chl_invrrs``, ``rmsd_invrrs``, ``spm_norm_invrrs``) and provenance attributes (``invrrs_*``),
        or None if fewer than 2 measurements are valid; with ``return_process``, ``(db, process)`` (``(None,
        None)``)
    '''
    records = records.where(records.aot550.notnull() & records.naot.notnull().any('wl_aeronet'), drop=True)
    records = records.where(records.Rrs_Lwn.notnull().sum('wl') >= 4, drop=True)
    if records.sizes['time'] < 2:
        return (None, None) if return_process else None
    hyp = reconstruction.reconstruct(records.Rrs_Lwn, records.sza_aeronet, chl=records.chl)
    hyp = hyp.where(hyp.success, drop=True)
    if hyp.sizes['time'] < 2:  # radcalnet_oc.Process squeezes a time dimension of size 1
        return (None, None) if return_process else None
    records = records.sel(time=hyp.time)
    ds = records[INPUT_VARS + ['naot']].copy()
    ds['Rrs'] = hyp.Rrs_nadir
    ds['Rrs'].attrs = {'unit': 'sr-1', 'description': 'remote-sensing reflectance at nadir reconstructed with '
                       'invRrs from the AERONET-OC Rrs_Lwn'}
    ds['sza'] = records.sza_aeronet
    ds['day_of_year'] = ds.time.dt.dayofyear
    vza = xr.DataArray([0.], coords={'vza': [0.]}, dims='vza') if vza is None else vza
    azi = xr.DataArray([0.], coords={'azi': [0.]}, dims='azi') if azi is None else azi
    process = Process(ds, vza=vza, azi=azi)
    aerosol = Aerosol(ds.naot.dropna('wl_aeronet', how='all'), process.lut.naot_lut)
    aerosol.process()
    process.execute(aerosol_combination=aerosol.model_db.aerosol_combination, sunglint=glint_stats)
    db = process.radcalnet_db.squeeze(drop=True)
    for v in ['chl', 'rmsd', 'spm_norm']:
        if v in hyp:
            db[v + '_invrrs'] = hyp[v]
    db['chl_invrrs'].attrs = {'unit': 'mg m-3', 'description': 'chlorophyll-a concentration of the invRrs fit'}
    db['rmsd_invrrs'].attrs = {'unit': 'sr-1', 'description': 'rmsd of the invRrs fit with the AERONET-OC bands'}
    db.attrs.update(hyp.attrs)   # invRrs version and settings of the reconstruction
    return (db, process) if return_process else db


def to_10nm(db):
    '''
    Convolution of the spectral variables of a simulation with the 10 nm bands of the products
    (:data:`~radcalnet_oc.products.CENTRAL_WL`, super-Gaussian responses of :data:`~radcalnet_oc.products.FWHM`
    nm FWHM, exponent :data:`~radcalnet_oc.products.EXPON`); the remote-sensing reflectances are interpolated and
    ``glint_uncertain`` is recomputed on the 10 nm bands.

    :param db: full-spectral simulation (output of :py:func:`~radcalnet_oc.products.simulate_month`)
    :return: :py:class:`xarray.Dataset` with the dimension ``wl`` of :data:`~radcalnet_oc.products.CENTRAL_WL`
    '''
    spectral = Spectral(CENTRAL_WL, FWHM)
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


def glint_ok(day, filter_flags=FILTER_FLAGS, uncertain=False):
    '''
    Times without the glint flags ``filter_flags`` (names of :data:`~radcalnet_oc.coxmunk.GLINT_FLAGS`) and, with
    ``uncertain``, without ``glint_uncertain`` in :data:`~radcalnet_oc.products.UNC_RANGE`.

    :param day: products (variables ``glint_flags`` and, with ``uncertain``, ``glint_uncertain``)
    :return: boolean :py:class:`xarray.DataArray` along ``time``
    '''
    mask = sum(GLINT_FLAGS[f] for f in filter_flags)
    ok = (day.glint_flags.astype(int) & mask) == 0
    if uncertain:
        ok = ok & ~day.glint_uncertain.sel(wl=slice(*UNC_RANGE)).any('wl')
    return ok


def glint_filtered(day, filter_flags=FILTER_FLAGS, uncertain=False):
    '''
    Glint-filtered copy of daily products: the times of :py:func:`~radcalnet_oc.products.glint_ok`, with the
    attribute ``glint_filter``.
    '''
    sub = day.isel(time=np.flatnonzero(glint_ok(day, filter_flags, uncertain).values))
    sub.attrs = dict(day.attrs)   # not shared with ``day`` (the daily files of the next days)
    sub.attrs['glint_filter'] = f'times with glint_flags {"/".join(filter_flags)} removed' + (
        f', and with glint_uncertain in {UNC_RANGE[0]}-{UNC_RANGE[1]} nm' if uncertain else '')
    return sub


def write_daily(db10, odir, name, level, filter_flags=FILTER_FLAGS, filter_uncertain=False, overwrite=False):
    '''
    Write the daily files of 10 nm products and their glint-filtered copy (days without any time left after the
    filter are not written in the copy; existing files are kept unless ``overwrite``).

    :param db10: products of one or several days (output of :py:func:`~radcalnet_oc.products.to_10nm`)
    :param odir: release folder (sub-folders ``daily`` and ``daily_glint_filtered``)
    :param name: site name of the products
    :param level: AERONET-OC level (file name)
    :return: number of days, of times and of times kept in the filtered copy
    '''
    n_days = n_kept = n_times = 0
    for day, ds_day in db10.groupby(db10.time.dt.floor('D')):
        day = pd.Timestamp(day)
        n_days += 1
        n_times += ds_day.sizes['time']
        for folder, sub in [(opj(odir, 'daily'), ds_day), (opj(odir, 'daily_glint_filtered'), None)]:
            if sub is None:
                sub = glint_filtered(ds_day, filter_flags, filter_uncertain)
                n_kept += sub.sizes['time']
                if sub.sizes['time'] == 0:
                    continue
            f = daily_file(folder, name, level, day)
            if os.path.exists(f) and not overwrite:
                continue
            os.makedirs(os.path.dirname(f), exist_ok=True)
            sub.to_netcdf(f)
    return n_days, n_times, n_kept


def make_products(site, level='lev20', name=None, start=START, end=None, odir='radcalnet_oc_products',
                  cache=None, glint_stats='cm_iso_mix', filter_flags=FILTER_FLAGS, filter_uncertain=False,
                  max_times=400, overwrite=False, reconstruction=None):
    '''
    RadCalNet-OC daily products of an AERONET-OC site (batch processing, see the module documentation).

    The months are processed one after the other; the full-spectral simulation of each month is cached in
    ``cache`` and read again on the next runs (restart after an interruption, or new filter settings). The
    months whose simulation fails are logged and skipped.

    :param site: AERONET-OC site name of the database (e.g. ``'AAOT'``)
    :param level: ``'lev15'`` or ``'lev20'``
    :param name: site name of the products (e.g. ``'VEIT'``; default: ``site``)
    :param start: first date
    :param end: last date (included), or None for the end of the record
    :param odir: release folder of the products (``daily`` and ``daily_glint_filtered``)
    :param cache: folder of the monthly full-spectral simulations (default ``<odir>/monthly``)
    :param glint_stats: slope statistics of the sunglint
    :param filter_flags: glint flags removed from the filtered copy
    :param filter_uncertain: also remove the times with ``glint_uncertain`` in 400-900 nm from the filtered copy
    :param max_times: largest number of measurements simulated at once (memory: about 2 GB per 400)
    :param overwrite: recompute the cached months and rewrite the existing daily files
    :param reconstruction: :py:class:`~radcalnet_oc.reconstruction.Reconstruction` (default settings if None)
    :return: dict with the numbers of ``months``, ``days``, ``times``, ``kept`` (times of the filtered copy) and
        the list of the ``failed`` months
    '''
    from .reconstruction import Reconstruction
    name = name or site
    cache = cache or opj(odir, 'monthly')
    filter_flags = list(filter_flags)
    inputs = aeronet_inputs(site, level, start=start, end=end)
    if inputs.sizes['time'] == 0:
        logger.warning(f'{site} {level}: no measurement from {start} to {end}')
        return dict(months=0, days=0, times=0, kept=0, failed=[])
    logger.info(f'{site} {level}: {inputs.sizes["time"]} measurements '
                f'{pd.Timestamp(inputs.time.values[0]):%Y-%m-%d} - {pd.Timestamp(inputs.time.values[-1]):%Y-%m-%d}')
    reconstruction = reconstruction or Reconstruction()
    t_all = _time.time()
    n_months = n_days = n_kept = n_times = 0
    failed = []
    for key, records in inputs.groupby(inputs.time.dt.strftime('%Y-%m')):
        year, month = (int(x) for x in key.split('-'))
        mfile = monthly_file(cache, name, level, year, month)
        if os.path.exists(mfile) and not overwrite:
            with xr.open_dataset(mfile) as f:
                db = f.load()
        else:
            t0 = _time.time()
            try:
                # chunks of at most max_times measurements (memory: full spectral grid per time)
                n = records.sizes['time']
                parts = [simulate_month(records.isel(time=slice(i, i + max_times)), reconstruction, glint_stats)
                         for i in range(0, n, max_times)]
                parts = [x for x in parts if x is not None]
                db = xr.concat(parts, 'time', data_vars='minimal', coords='minimal', compat='override',
                               combine_attrs='override') if parts else None
            except Exception:
                logger.exception(f'{year}-{month:02}: failed')
                failed.append(key)
                continue
            if db is None:
                logger.info(f'{year}-{month:02}: fewer than 2 valid measurements')
                continue
            db.attrs.update(site=name, aeronet_oc_site=site, aeronet_oc_level=level, geometry='nadir view')
            os.makedirs(os.path.dirname(mfile), exist_ok=True)
            db.to_netcdf(mfile + '.tmp')
            os.replace(mfile + '.tmp', mfile)
            logger.info(f'{year}-{month:02}: {db.sizes["time"]} times ({_time.time() - t0:.0f} s)')
        n_months += 1
        d, t, k = write_daily(to_10nm(db), odir, name, level, filter_flags, filter_uncertain, overwrite)
        n_days += d
        n_times += t
        n_kept += k
    logger.info(f'done: {n_days} days, {n_times} times, {n_kept} kept after the glint filter '
                f'({_time.time() - t_all:.0f} s)')
    return dict(months=n_months, days=n_days, times=n_times, kept=n_kept, failed=failed)


def parser(odir=None, cache=None, description=None, prog=None):
    '''
    Command-line parser of :py:func:`~radcalnet_oc.products.make_products` (``radcalnet_oc-products``).

    :param odir: default of ``--odir`` (required if None)
    :param cache: default of ``--cache`` (``<odir>/monthly`` if None)
    '''
    p = argparse.ArgumentParser(prog=prog, description=description or __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', required=True, help='AERONET-OC site (aeronet_oc_db name)')
    p.add_argument('--name', help='site name of the products (default: --site)')
    p.add_argument('--level', default='lev20', choices=['lev15', 'lev20'])
    p.add_argument('--start', default=START)
    p.add_argument('--end', default=None)
    p.add_argument('--odir', default=odir, required=odir is None,
                   help='release folder of the products (daily, daily_glint_filtered)')
    p.add_argument('--cache', default=cache,
                   help='folder of the monthly full-spectral simulations'
                        + ('' if cache else ' (default <odir>/monthly)'))
    p.add_argument('--glint-stats', default='cm_iso_mix')
    p.add_argument('--filter-flags', nargs='*', default=list(FILTER_FLAGS), choices=list(GLINT_FLAGS))
    p.add_argument('--filter-uncertain', action='store_true',
                   help='also remove the times with glint_uncertain in 400-900 nm from the filtered copy')
    p.add_argument('--max-times', type=int, default=400,
                   help='largest number of measurements simulated at once (memory, about 2 GB per 400)')
    p.add_argument('--overwrite', action='store_true')
    return p


def main(argv=None, odir=None, cache=None, description=None, prog=None):
    '''
    Command line ``radcalnet_oc-products``: :py:func:`~radcalnet_oc.products.make_products` for one site. The
    progress is logged at ``INFO`` level on the standard error, the messages of the rest of the package at
    ``WARNING`` level.

    :param argv: arguments (default: ``sys.argv[1:]``)
    :param odir: default of ``--odir`` (required if None)
    :param cache: default of ``--cache`` (``<odir>/monthly`` if None)
    :return: exit status (0)
    '''
    args = parser(odir, cache, description, prog).parse_args(argv)
    from .log import set_logging
    set_logging('WARNING')
    logger.setLevel(logging.INFO)
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(name)s | %(message)s')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    summary = make_products(args.site, args.level, name=args.name, start=args.start, end=args.end,
                            odir=args.odir, cache=args.cache, glint_stats=args.glint_stats,
                            filter_flags=args.filter_flags, filter_uncertain=args.filter_uncertain,
                            max_times=args.max_times, overwrite=args.overwrite)
    if summary['failed']:
        logger.warning(f'failed months (rerun to retry them): {", ".join(summary["failed"])}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
