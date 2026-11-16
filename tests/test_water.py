"""Pure water optical properties and spectral extension of the remote-sensing reflectance."""

from importlib.resources import files

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc.water import PureWater, extend_rrs, rrs_to_u, u_to_rrs

TEMPLATES = files('radcalnet_oc.data.template')
WL = np.arange(350., 2501., 1.)


@pytest.fixture(scope='module')
def turbid():
    return xr.open_dataset(TEMPLATES / 'template_turbid_water.nc').Rrs.isel(time=slice(0, 3)).load()


def test_pure_water_coefficients():
    aw, bbw = PureWater().coefficients([440., 700., 1240., 1450., 2449., 2500.])
    np.testing.assert_allclose(aw.sel(wl=[440., 700.]), [0.0064, 0.624], rtol=0.05)  # Pope and Fry (1997)
    assert float(aw.sel(wl=1450.)) > float(aw.sel(wl=1240.)) > 100       # Kou et al. (1993)
    assert float(aw.sel(wl=2500.)) > float(aw.sel(wl=2449.))             # extrapolated beyond the table
    assert (bbw > 0).all() and float(bbw.sel(wl=440.)) > float(bbw.sel(wl=700.))


def test_rrs_u_inverse():
    u = xr.DataArray(np.linspace(0., 0.4, 9))
    np.testing.assert_allclose(rrs_to_u(u_to_rrs(u)), u, atol=1e-12)


def test_extension_reproduces_template(turbid):
    """Cut at 900 nm and extended, the turbid template is recovered within 15 % of its values at
    900-1150 nm (reconstructed with the full invRrs model) and within 4 % of its peak."""
    ext, bbp = extend_rrs(turbid.sel(wl=slice(None, 900)), WL)
    assert (bbp > 1).all()
    ref = turbid.interp(wl=np.arange(901., 1149., 1.))
    err = abs(ext.sel(wl=ref.wl) - ref)
    assert float((err.max('wl') / turbid.max('wl')).max()) < 0.04
    assert float(abs(ext.sel(wl=ref.wl).integrate('wl') / ref.integrate('wl') - 1).max()) < 0.15


def test_extension_full_range(turbid):
    ext, bbp = extend_rrs(turbid, WL)
    assert ext.dims == ('time', 'wl') and np.isfinite(ext).all()
    # inside the range of the input: linear interpolation
    np.testing.assert_allclose(ext.sel(wl=slice(400, 1100)), turbid.interp(wl=ext.wl.sel(wl=slice(400, 1100))))
    # continuous at the end of the input, positive and small in the short-wave infrared
    np.testing.assert_allclose(ext.sel(wl=1151.), turbid.isel(wl=-1), rtol=0.05)
    swir = ext.sel(wl=[1240., 1640., 2130.])
    assert (swir > 0).all() and (swir < 0.01).all()
    # absorption bands of water: lower reflectance
    assert (ext.sel(wl=1450.) < ext.sel(wl=1240.)).all()


def test_extension_zero_and_missing(turbid):
    zero, bbp = extend_rrs(turbid, WL, method='zero')
    assert float(abs(zero.sel(wl=slice(1151, None))).max()) == 0 and bbp.isnull().all()
    with pytest.raises(ValueError):
        extend_rrs(turbid, WL, method='linear')
    # gaps and spectra of different ranges, and a spectrum without valid values
    rrs = turbid.copy()
    rrs[0, rrs.wl > 1000] = np.nan
    rrs[1, (rrs.wl > 600) & (rrs.wl < 650)] = np.nan
    rrs[2] = np.nan
    ext, bbp = extend_rrs(rrs, WL)
    assert np.isfinite(ext.isel(time=[0, 1])).all() and ext.isel(time=2).isnull().all()
    assert np.isfinite(bbp.isel(time=[0, 1])).all() and np.isnan(bbp.isel(time=2))
