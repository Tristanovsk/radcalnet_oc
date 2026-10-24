"""Atmospheric components: aerosol model retrieval, gaseous transmittance and radiometric utilities."""

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc import Aerosol, GaseousTransmittance, Misc, Radiometry
from radcalnet_oc.config import LUT_FILE_BACKUP
from radcalnet_oc.kernel import AEROSOL_MODELS
from radcalnet_oc.lut import LUT


@pytest.fixture(scope='module')
def gas_lut():
    return LUT(lut_file=LUT_FILE_BACKUP).gas_lut


# ------------------------------------
# Misc
# ------------------------------------
def test_air_mass():
    assert Misc.air_mass(0, 0) == pytest.approx(2)
    assert Misc.air_mass(60, 0) == pytest.approx(3)


def test_transmittance_dir():
    assert Misc.transmittance_dir(0, 2) == 1
    assert Misc.transmittance_dir(0.1, 2, rot=0.1) == pytest.approx(np.exp(-0.4))


def test_pressure():
    assert Misc.get_pressure(0, 1013.25) == pytest.approx(1013.25)
    assert Misc.get_pressure(np.nan, 1013.25) == pytest.approx(1013.25)       # NaN altitude: sea level
    assert Misc.get_pressure(1000, 1013.25) == pytest.approx(898.7, abs=1)


def test_earth_sun_correction():
    doy = np.arange(1, 366)
    d2 = Misc.earth_sun_correction(doy)
    assert d2.min() == pytest.approx(0.967, abs=2e-3)       # aphelion, early July
    assert d2.max() == pytest.approx(1.035, abs=2e-3)       # perihelion, early January
    assert 170 < doy[np.argmin(d2)] < 200
    assert doy[np.argmax(d2)] < 15 or doy[np.argmax(d2)] > 355


# ------------------------------------
# Radiometry
# ------------------------------------
def test_par():
    """PAR of a flat spectrum of 1.5 W m-2 nm-1 over 400-700 nm (450 W m-2): about 2070 µmol m-2 s-1."""
    wl = np.arange(350, 800, 1.)
    ed = xr.DataArray(np.full(wl.size, 1500.), coords={'wl': wl}, dims='wl')
    expected = 1500e-3 * (700 ** 2 - 400 ** 2) / 2 * 1e-9 / (6.62607e-34 * 2.99792e8 * 6.02214e23) * 1e6
    assert float(Radiometry().PAR(ed)) == pytest.approx(expected, rel=1e-3)
    assert 2000 < expected < 2100


# ------------------------------------
# gaseous transmittance
# ------------------------------------
def test_gaseous_transmittance(gas_lut):
    gas = GaseousTransmittance(gas_lut)
    tg = gas.get_gaseous_transmittance().sel(wl=slice(350, 2500))
    assert ((tg >= 0) & (tg <= 1)).all()
    # water vapour band at 940 nm, almost transparent window at 865 nm
    assert float(tg.sel(wl=slice(935, 945)).mean()) < 0.5
    assert float(tg.sel(wl=slice(860, 870)).mean()) > 0.95


def test_gaseous_transmittance_air_mass_and_column(gas_lut):
    gas = GaseousTransmittance(gas_lut)
    t1 = gas.Tgas('h2o').sel(wl=slice(900, 1000))
    gas.air_mass = 2.
    np.testing.assert_allclose(gas.Tgas('h2o').sel(wl=slice(900, 1000)), t1 ** 2, rtol=1e-5)
    gas.air_mass = 1.
    gas.gas_tc['h2o'] = 2 * gas.gas_tc['h2o']
    np.testing.assert_allclose(gas.Tgas('h2o').sel(wl=slice(900, 1000)), t1 ** 2, rtol=1e-5)


def test_gaseous_transmittance_zenith_angle(gas_lut):
    assert GaseousTransmittance(gas_lut, zenith_angle=60).air_mass == pytest.approx(2)


# ------------------------------------
# aerosol model retrieval
# ------------------------------------
def synthetic_naot_lut():
    """Normalized AOT spectra (1 at 550 nm) of the models, Angström exponents from 0 to 1.5."""
    wl = np.arange(400., 1100., 10.)
    angstrom = dict(zip(AEROSOL_MODELS, [0.8, 1.0, 0.0, 0.3, 1.2, 1.5]))
    data = np.array([(wl / 550.) ** -angstrom[model] for model in AEROSOL_MODELS])
    return xr.DataArray(data, coords={'model': AEROSOL_MODELS, 'wl': wl}, dims=('model', 'wl'))


def test_aerosol_retrieval():
    naot_lut = synthetic_naot_lut()
    wl_aeronet = np.array([412., 443., 490., 560., 667., 865., 1020.])
    truth = {'DESE_rh70': [0.2, 0.0, 1.0], 'MACL_rh70': [0.5, 1.0, 0.0], 'WASO_rh0': [0.3, 0.0, 0.0]}
    naot = sum(xr.DataArray(truth[model], dims='time', coords={'time': [0, 1, 2]})
               * naot_lut.sel(model=model, drop=True).interp(wl=wl_aeronet) for model in truth)
    naot = naot.rename(wl='wl_aeronet')

    aerosol = Aerosol(naot, naot_lut)
    aerosol.process()
    comb = aerosol.model_db.aerosol_combination
    assert list(comb.model.values) == AEROSOL_MODELS
    np.testing.assert_allclose(comb.sum('model'), 1)
    for model in truth:
        np.testing.assert_allclose(comb.sel(model=model), truth[model], atol=0.01)
    for model in ['ARCT_rh70', 'COAV_rh70', 'URBA_rh70']:
        assert (comb.sel(model=model) == 0).all()
    assert (aerosol.model_db.cost < 1e-5).all()
