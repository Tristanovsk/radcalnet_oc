"""Look-up tables, spectral functions, auxiliary data and spectral responses."""

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc import lut as lut_module
from radcalnet_oc.config import LUT_FILE_BACKUP
from radcalnet_oc.lut import (LUT, AuxData, SolarIrradiance, Spectral, Gamma2sigma, gaussian,
                              super_gaussian, super_gaussian_fwhm2sigma)

from conftest import FULL_LUT, requires_full_lut


@pytest.fixture(scope='module')
def light_lut():
    """LUT object with the light TOA LUT of the package."""
    return LUT(lut_file=LUT_FILE_BACKUP)


# ------------------------------------
# spectral functions
# ------------------------------------
def test_gamma2sigma():
    assert Gamma2sigma(2 * np.sqrt(2 * np.log(2))) == pytest.approx(1)


def test_gaussian_normalized():
    x = np.arange(400, 700, 0.1)
    g = gaussian(x, 550., 10.)
    assert np.trapezoid(g, x) == pytest.approx(1, rel=1e-4)
    assert x[np.argmax(g)] == pytest.approx(550)


@pytest.mark.parametrize('expon', [2., 3., 6.])
def test_super_gaussian_fwhm(expon):
    fwhm, mu = 20., 550.
    sigma = super_gaussian_fwhm2sigma(fwhm, expon)
    peak = super_gaussian(mu, mu=mu, sigma=sigma, expon=expon)
    assert super_gaussian(mu + fwhm / 2, mu=mu, sigma=sigma, expon=expon) / peak == pytest.approx(0.5)
    assert super_gaussian(mu - fwhm / 2, mu=mu, sigma=sigma, expon=expon) / peak == pytest.approx(0.5)


def test_super_gaussian_is_gaussian_for_expon_2():
    x = np.linspace(500, 600, 11)
    np.testing.assert_allclose(super_gaussian(x, mu=550., sigma=10., expon=2.), gaussian(x, 550., 10.),
                               rtol=1e-5)


# ------------------------------------
# auxiliary data
# ------------------------------------
def test_rayleigh_optical_thickness():
    rot = AuxData().rot
    assert float(rot.interp(wl=550)) == pytest.approx(0.0973, rel=0.02)   # Bodhaine et al. (1999)
    assert (rot.sel(wl=slice(350, 2500)).diff('wl') < 0).all()            # decreases with wavelength


def test_auxdata_interpolation():
    wl = np.arange(400, 900, 10)
    aux = AuxData(wl=wl)
    np.testing.assert_array_equal(aux.rot.wl, wl)
    np.testing.assert_array_equal(aux.sunglint_eps.wl, wl)
    assert np.isfinite(aux.rot).all()


def test_solar_irradiance():
    solar = SolarIrradiance()
    for name in ['tsis', 'thuillier', 'gueymard', 'kurucz']:
        f0 = getattr(solar, name)
        # extraterrestrial irradiance in mW m-2 nm-1: about 1850 at 550 nm, 960 at 865 nm
        assert float(f0.sel(wl=slice(545, 555)).mean()) == pytest.approx(1870, rel=0.05), name
        assert float(f0.sel(wl=slice(860, 870)).mean()) == pytest.approx(960, rel=0.05), name


# ------------------------------------
# spectral responses
# ------------------------------------
def test_convolution_of_constant_and_linear_signals():
    wl = np.arange(350, 1000, 0.5)
    spectral = Spectral(central_wl=np.array([443., 560., 865.]), fwhm=20.)
    constant = xr.DataArray(np.full(wl.size, 0.3), coords={'wl': wl}, dims='wl', name='R')
    np.testing.assert_allclose(spectral.convolve2(constant).values, 0.3, rtol=1e-6)
    # symmetric responses: the band value of a linear signal is its value at the central wavelength
    linear = xr.DataArray(1e-3 * wl, coords={'wl': wl}, dims='wl', name='R')
    np.testing.assert_allclose(spectral.convolve2(linear).values, 1e-3 * spectral.central_wl, rtol=1e-4)


def test_convolution_multidimensional():
    wl = np.arange(350, 1000, 1.)
    signal = xr.DataArray(np.ones((4, wl.size)), coords={'time': range(4), 'wl': wl},
                          dims=('time', 'wl'), name='Rtoa')
    spectral = Spectral(central_wl=np.array([490., 665.]), fwhm=np.array([10., 30.]))
    bands = spectral.convolve2(signal)
    assert isinstance(bands, xr.Dataset)
    assert dict(bands.Rtoa.sizes) == {'time': 4, 'wl': 2}
    np.testing.assert_allclose(bands.Rtoa.values, 1, rtol=1e-6)


# ------------------------------------
# look-up tables
# ------------------------------------
def test_light_lut_used_without_full_lut():
    """Without the full TOA LUT (download switched off), the light LUT of the package is used."""
    assert LUT().lut_file == LUT_FILE_BACKUP
    assert LUT(lut_file='/no/such/lut.nc').lut_file == LUT_FILE_BACKUP


def test_lut_units(light_lut):
    assert float(light_lut.aero_lut.wl.min()) > 300            # nm
    assert float(light_lut.trans_lut.wl.min()) > 300
    assert set(light_lut.naot_lut.dims) == {'model', 'wl'}
    np.testing.assert_allclose(light_lut.naot_lut.interp(wl=550).values, 1, rtol=0.05)


@pytest.mark.parametrize('azi', [[0.], [20.], [45.], [360.], [0., 90.], [170., 180., 190.]])
def test_lut_preparation_azimuth(light_lut, azi):
    """Regression (a4dcb10): finite Rayleigh and path reflectances also when an azimuth is a node of the
    table, including 0 and 360 deg."""
    light_lut.lut_preparation(sza=[30., 50.], vza=[0., 10.], azi=azi, aot_refs=[0., 0.1, 0.3])
    rray, rdiff = light_lut.Rray, light_lut.Rdiff_lut
    np.testing.assert_array_equal(rray.azi, azi)
    assert np.isfinite(rray).all() and (rray > 0).all()
    assert np.isfinite(rdiff).all() and (rdiff > 0).all()
    # aerosols add path reflectance in the visible
    vis = dict(wl=slice(400, 700))
    assert (rdiff.sel(aot_ref=0.3, **vis) > rdiff.sel(aot_ref=0., **vis)).all()


def test_lut_preparation_transmittance(light_lut):
    light_lut.lut_preparation(sza=[20., 60.], vza=[0.], azi=[90.], aot_refs=[0., 0.5])
    t_ed = light_lut.trans_Ed
    assert ((t_ed > 0) & (t_ed <= 1)).all()
    # lower transmittance for a longer path and more aerosols
    assert (t_ed.sel(sza=60) < t_ed.sel(sza=20)).all()
    assert (t_ed.sel(aot_ref=0.5) < t_ed.sel(aot_ref=0.)).all()


def test_aerosol_combination(light_lut):
    """A mixture is between its pure models; the order of the models does not matter."""
    kwargs = dict(sza=[40.], vza=[0.], azi=[90.], aot_refs=[0.3])
    models = lut_module.AEROSOL_MODELS
    rdiff = {}
    for name, comb in [('COAV', [0, 1, 0, 0, 0, 0]), ('MACL', [0, 0, 0, 1, 0, 0]),
                       ('mix', [0, 0.5, 0, 0.5, 0, 0])]:
        light_lut.lut_preparation(aerosol_combination=xr.DataArray(comb, coords={'model': models}),
                                  **kwargs)
        rdiff[name] = light_lut.Rdiff_lut.squeeze()
    np.testing.assert_allclose(rdiff['mix'], 0.5 * (rdiff['COAV'] + rdiff['MACL']), rtol=1e-5)


def test_toa_lut_cache(light_lut):
    lut_module.TOA_LUT_CACHE.clear()
    kwargs = dict(sza=[30.], vza=[0.], azi=[100.], aot_refs=[0.1])
    light_lut.lut_preparation(**kwargs)
    size = len(lut_module.TOA_LUT_CACHE)
    assert size > 0
    LUT(lut_file=LUT_FILE_BACKUP).lut_preparation(**kwargs)    # new object, same tables: no new entry
    assert len(lut_module.TOA_LUT_CACHE) == size
    assert all(key[0] == str(LUT_FILE_BACKUP) for key in lut_module.TOA_LUT_CACHE)


@requires_full_lut
@pytest.mark.parametrize('azi', [[0.], [45.], [360.]])
def test_full_lut_azimuth_nodes(azi):
    full = LUT(lut_file=FULL_LUT)
    assert str(full.lut_file) == FULL_LUT
    full.lut_preparation(sza=[30.], vza=[0., 10.], azi=azi, aot_refs=[0., 0.1])
    assert np.isfinite(full.Rray).all() and (full.Rray > 0).all()
    assert np.isfinite(full.Rdiff_lut).all() and (full.Rdiff_lut > 0).all()
