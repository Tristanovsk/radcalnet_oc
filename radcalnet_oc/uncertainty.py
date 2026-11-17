# coding=utf-8
'''
Uncertainty budget of the simulated TOA reflectance (see the Uncertainty section of :doc:`/methods`).

The standard uncertainty of each input of :py:class:`~radcalnet_oc.process.Process` is propagated to the TOA
reflectance by central finite differences through the full simulation (the aerosol mixture retrieved for the
nominal simulation is kept), and the uncertainty of the direct sunglint (:py:meth:`Process.compute_sunglint
<radcalnet_oc.process.Process.compute_sunglint>`: ensemble of plausible wave slope statistics and wind
uncertainty) is added as a model term. The terms are assumed independent and combined in quadrature.

Example::

    process = Process(input_db=input_db, vza=vza, azi=azi)
    process.execute(sunglint='cm_iso_mix')
    db = uncertainty_budget(process)
    db.u_total_rel.sel(wl=865, method='nearest')
'''

import copy
import logging

import numpy as np
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
#: lower bounds of the perturbed inputs
LOWER_BOUND = {'aot550': 0., 'tcwv': 0., 'tco3': 0., 'wind': 0., 'Rrs': 0.}


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


def uncertainty_budget(process, inputs=None, glint=True, Rrs_name='Rrs'):
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
    total = None
    for name, u in terms.items():
        u = u.transpose(*[d for d in db.Rtoa.dims if d in u.dims])
        db[f'u_{name}'] = u
        db[f'u_{name}'].attrs = {'unit': '-', 'description': f'standard uncertainty of the TOA reflectance due to '
                                 f'{name}' + (f' ({inputs[name][1]:g} {"relative" if inputs[name][0] == "rel" else "absolute"})'
                                              if name in inputs else ' (ensemble of slope statistics and wind)')}
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
