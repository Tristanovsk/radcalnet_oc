# coding=utf-8
'''
Uncertainty budget of the simulated TOA reflectance (see the Uncertainty section of :doc:`/methods`).

The standard uncertainty of each input of :py:class:`~radcalnet_oc.process.Process` is propagated to the TOA
reflectance by central finite differences through the full simulation (the aerosol mixture retrieved for the
nominal simulation is kept), and the uncertainty of the direct sunglint (:py:meth:`Process.compute_sunglint
<radcalnet_oc.process.Process.compute_sunglint>`: ensemble of plausible wave slope statistics and wind
uncertainty) is added as a model term. The model gap (:py:func:`model_uncertainty`), the part of the Sentinel-2
matchup residual that neither the inputs nor the glint model explain, is added per sunglint level and sun zenith
angle. The terms are assumed independent and combined in quadrature.

Example::

    process = Process(input_db=input_db, vza=vza, azi=azi)
    process.execute(sunglint='cm_iso_mix')
    db = uncertainty_budget(process)
    db.u_total_rel.sel(wl=865, method='nearest')
'''

import copy
import logging
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

#: standard uncertainties of the inputs, ``name: (kind, value)`` with ``kind`` ``'abs'`` (unit of the input) or
#: ``'rel'`` (fraction of the input). PROVISIONAL defaults, to be replaced by the values of the RadCalNet
#: uncertainty assessment: AOT(550) 0.01 (AERONET direct sun), water vapour 10 %, ozone 3 %, pressure 2 hPa, wind
#: 1.5 m s-1 (sky reflection of the look-up tables only; the direct glint has its own term) and Rrs 5 %
INPUT_UNC = {'aot550': ('abs', 0.01),
             'tcwv': ('rel', 0.10),
             'tco3': ('rel', 0.03),
             'pressure': ('abs', 2.),
             'wind': ('abs', 1.5),
             'Rrs': ('rel', 0.05)}
#: table of the model gap (glint level x sun zenith class x band), made from the Sentinel-2 matchups by
#: ``scripts/s2_toa_matchup/residual_budget.py --glint-min 0 --model-table``
MODEL_UNC_FILE = files('radcalnet_oc.data.auxdata') / 'model_uncertainty_s2.csv'
#: lower bounds of the perturbed inputs
LOWER_BOUND = {'aot550': 0., 'tcwv': 0., 'tco3': 0., 'wind': 0., 'Rrs': 0.}


MODEL_DESCRIPTION = {'glint': ' (ensemble of slope statistics and wind)',
                     'model': ' gap per sunglint level and sun zenith angle (Sentinel-2 matchups, model_uncertainty)'}


def _perturbed(input_db, name, kind, value, sign, Rrs_name='Rrs'):
    '''Copy of the input dataset with ``name`` shifted by ``sign`` times its standard uncertainty.'''
    db = input_db.copy()
    var = Rrs_name if name == 'Rrs' else name
    x = db[var]
    dx = value * np.abs(x) if kind == 'rel' else value
    new = x + sign * dx
    if name in LOWER_BOUND:
        new = new.clip(min=LOWER_BOUND[name])
    db[var] = new.assign_attrs(x.attrs)
    if name == 'aot550' and 'aot865' in db:  # keep the AOT spectrum (same Angstrom exponent)
        db['aot865'] = db['aot865'] * (new / x).fillna(1.)
    return db


def glint_level(flags):
    '''Glint level (str array) of the bitmask of :py:func:`~radcalnet_oc.coxmunk.glint_flags`: ``'near_specular'``
    (glint angle < 30 deg), else ``'high'``, ``'moderate'``, ``'low'`` or ``'none'`` (direct glint >= 50, 10, 1 % of
    the TOA reflectance at 865 nm, or less).'''
    f = np.asarray(flags).astype(int)
    return np.select([f & 8 > 0, f & 4 > 0, f & 2 > 0, f & 1 > 0],
                     ['near_specular', 'high', 'moderate', 'low'], 'none')


def model_uncertainty(db, table=MODEL_UNC_FILE):
    r'''
    Model gap of the TOA reflectance per sunglint level and sun zenith angle (after
    :py:meth:`~radcalnet_oc.process.Process.compute_sunglint`, which sets ``glint_flags``).

    The table gives, per glint level, sun zenith class and Sentinel-2 band (443-2190 nm), the part of the residual
    MSI - simulation of the matchups at VEIT and CORO that the physical budget does not explain:
    :math:`\sqrt{\tilde r^2 + \max(0, s_r^2 - \tilde u^2)}` with :math:`\tilde r` the median residual after the
    predicted missing terms (cirrus, whitecaps, land environment), :math:`s_r` its robust spread and
    :math:`\tilde u` the median of the known uncertainties. It grows with the glint (865 nm: about 1.5, 1.7,
    2.5 and 5-7 x 1e-3 for the levels none, low, moderate and high / near specular) and, in the blue and green,
    with the sun zenith angle (443 nm: 0.2-1.2 x 1e-3 below 55 deg, 2.5 x 1e-3 above, level low). Empirical: two
    sites, Sentinel-2 geometry (viewing zenith < 12 deg); classes with few matchups use the value of their level
    (column ``provisional``). Interpolated linearly in log(wavelength) between the bands, constant beyond.

    :param db: ``radcalnet_db`` with ``glint_flags`` and the coordinate ``sza``
    :param table: CSV of the table (default ``MODEL_UNC_FILE``, ``data/auxdata/model_uncertainty_s2.csv`` of the
        package) or :py:class:`pandas.DataFrame`
    :return: standard uncertainty (reflectance), dimensions of ``glint_flags`` and ``wl``
    '''
    t = table if isinstance(table, pd.DataFrame) else pd.read_csv(table, comment='#')
    wl = db.wl.values
    flags = db.glint_flags
    sza = db.sza.broadcast_like(flags)
    level = glint_level(flags.values).ravel()
    zen = np.asarray(sza.values, float).ravel()
    out = np.full((level.size, wl.size), np.nan)
    for (lev, lo, hi), g in t.groupby(['level', 'sza_min', 'sza_max']):
        m = (level == lev) & (zen >= lo) & (zen < hi)
        if m.any():
            g = g.sort_values('wl')
            out[m] = np.interp(np.log(wl), np.log(g.wl.values), g.u_model.values)
    return xr.DataArray(out.reshape(flags.shape + (wl.size,)), coords={**flags.coords, 'wl': wl},
                        dims=flags.dims + ('wl',))


def uncertainty_budget(process, inputs=None, glint=True, Rrs_name='Rrs', model=True):
    '''
    Uncertainty budget of the TOA reflectance of a :py:class:`~radcalnet_oc.process.Process` already executed.

    For each input of ``inputs``, the simulation is run again with the input shifted by plus and minus its
    standard uncertainty (other inputs and aerosol mixture unchanged) and ``u_<input> = |Rtoa(+) - Rtoa(-)| / 2``.
    With ``glint`` and if :py:meth:`~radcalnet_oc.process.Process.compute_sunglint` was run, ``u_glint`` is
    ``Rglint_unc`` (half range of the plausible slope statistics, treated as a standard uncertainty: on the
    Sentinel-2 matchups, twice its value covers 61--77 % of the glint residuals). The total
    ``u_total = sqrt(sum u_i^2)`` assumes independent terms; ``u_total_rel`` is relative to the TOA reflectance
    (with the direct glint when computed).

    :param process: :py:class:`~radcalnet_oc.process.Process` after :py:meth:`~radcalnet_oc.process.Process.execute`
    :param inputs: standard uncertainties of the inputs, ``{name: (kind, value)}`` (default :data:`INPUT_UNC`);
        inputs absent from the input dataset are skipped (e.g. ``wind``, then the default wind is perturbed)
    :param glint: include the uncertainty of the direct glint
    :param model: include the model gap per sunglint level and sun zenith angle (:py:func:`model_uncertainty`,
        ``u_model``), if :py:meth:`~radcalnet_oc.process.Process.compute_sunglint` was run
    :param Rrs_name: name of the remote-sensing reflectance in the input dataset
    :return: ``process.radcalnet_db`` with the variables ``u_<term>``, ``u_total`` and ``u_total_rel`` (spectral)
        and the attribute ``uncertainty_inputs``
    '''
    inputs = INPUT_UNC if inputs is None else inputs
    db = process.radcalnet_db
    aerosol_combination = process.lut.aerosol_combination
    # the perturbed runs share the LUT and gas objects of the process: their state is restored at the end
    states = {obj: dict(obj.__dict__) for obj in (process.lut, process.gas_trans)}
    terms = {}
    try:
        for name, (kind, value) in inputs.items():
            input_db = process.input_db
            if name == 'wind' and 'wind' not in input_db:
                input_db = input_db.assign(wind=process.wind)
            var = Rrs_name if name == 'Rrs' else name
            if var not in input_db:
                logger.warning(f'uncertainty budget: {var} not in the input dataset, skipped')
                continue
            R = []
            for sign in (1, -1):
                p = copy.copy(process)
                p.input_db = _perturbed(input_db, name, kind, value, sign, Rrs_name)
                p.set_param(Rrs_name=Rrs_name)
                p.execute(aerosol_combination=aerosol_combination)
                R.append(p.Rtoa.reset_coords(drop=True))
            terms[name] = (np.abs(R[0] - R[1]) / 2).squeeze()
    finally:
        for obj, state in states.items():
            obj.__dict__.clear()
            obj.__dict__.update(state)

    if glint and 'Rglint_unc' in db:
        terms['glint'] = db.Rglint_unc
    if model and 'glint_flags' in db:
        terms['model'] = model_uncertainty(db).reset_coords(drop=True)
    total = None
    for name, u in terms.items():
        u = u.transpose(*[d for d in db.Rtoa.dims if d in u.dims])
        db[f'u_{name}'] = u
        db[f'u_{name}'].attrs = {'unit': '-', 'description': f'standard uncertainty of the TOA reflectance due to '
                                 f'{name}' + (f' ({inputs[name][1]:g} {"relative" if inputs[name][0] == "rel" else "absolute"})'
                                              if name in inputs else MODEL_DESCRIPTION.get(name, ''))}
        total = u ** 2 if total is None else total + u ** 2
    db['u_total'] = np.sqrt(total)
    db['u_total'].attrs = {'unit': '-', 'description': 'combined standard uncertainty of the TOA reflectance '
                           f'(quadrature of {list(terms)}, independent terms)'}
    reference = db.Rtoa_glint if 'Rtoa_glint' in db else db.Rtoa
    db['u_total_rel'] = db.u_total / reference
    db['u_total_rel'].attrs = {'unit': '-', 'description': f'u_total / {reference.name}'}
    db.attrs['uncertainty_inputs'] = repr(inputs)
    process.radcalnet_db = db
    return db
