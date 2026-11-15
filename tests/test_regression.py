"""Regression test: band-averaged outputs of the clear-water template against a reference file.

A failure means that the results changed. If the change is deliberate (new default, physics change), update
the reference with ``python tests/reference.py`` and mention it in CHANGELOG.md.
"""

import numpy as np
import pytest
import xarray as xr

from reference import REFERENCE_FILE, simulate

#: relative tolerance: numerical noise between platforms and library versions, far below any change of
#: the physics (which moves the absorbing bands by percents)
RTOL = 1e-4
#: absolute tolerance, for the saturated bands (e.g. 1375 nm, reflectance ~0): reflectance (-) and
#: irradiance (mW m-2 nm-1)
ATOL = {'Rtoa': 1e-7, 'Ratm': 1e-7, 'Ed': 1e-4}


@pytest.fixture(scope='module')
def current():
    return simulate()


@pytest.fixture(scope='module')
def reference():
    with xr.open_dataset(REFERENCE_FILE) as ref:
        return ref.load()


def test_reference_variables(current, reference):
    assert set(current.data_vars) == set(reference.data_vars)


@pytest.mark.parametrize('name', ['Rtoa_default', 'Ratm_default', 'Ed_default',
                                  'Rtoa_decoupled_reptran', 'Ratm_decoupled_reptran', 'Ed_decoupled_reptran'])
def test_against_reference(current, reference, name):
    cur, ref = xr.align(current[name], reference[name], join='exact')
    excess = np.abs(cur - ref) - (RTOL * np.abs(ref) + ATOL[name.split('_')[0]])
    rel = (np.abs(cur - ref) / np.abs(ref)).where(excess > 0).max(set(cur.dims) - {'wl'})
    bad = rel.where(rel.notnull(), drop=True)
    assert bad.size == 0, (f'{name} differs from the reference (relative tolerance {RTOL:g}) at '
                           f'{dict(zip(bad.wl.values.tolist(), bad.values.round(6).tolist()))} (wl: relative '
                           'difference); if the change is deliberate, run `python tests/reference.py`')
