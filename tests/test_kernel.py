"""Atmospheric components: aerosol model retrieval, gaseous transmittance and radiometric utilities."""

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc import Aerosol, GaseousTransmittance, Misc, Radiometry
from radcalnet_oc.config import LUT_FILE_BACKUP
from radcalnet_oc.kernel import AEROSOL_MODELS, O2_MODELS, O2LineByLine
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
# line-by-line O2
# ------------------------------------
def equivalent_width(tg, lo, hi):
    """Equivalent width (nm) of a band of the transmittance (continuum 1)."""
    t = tg.sel(wl=slice(lo, hi))
    return float(np.trapezoid(1 - t.values, t.wl.values))


def test_o2_lbl_table(gas_lut):
    lbl = O2LineByLine(gas_lut.wl.values)
    wl = gas_lut.wl.values[lbl.mask]
    # cells of the O2-B and O2-A bands only
    assert wl.min() == pytest.approx(682.6, abs=0.1) and wl.max() == pytest.approx(775.2, abs=0.1)
    assert not lbl.mask[(gas_lut.wl.values > 702) & (gas_lut.wl.values < 754)].any()
    np.testing.assert_allclose(lbl.transmittance(0.), 1.)
    t1, t2 = lbl.transmittance(1.), lbl.transmittance(2.)
    assert ((t2 <= t1 + 1e-12) & (t1 <= 1) & (t2 >= 0)).all()
    # cell average of exp(-tau), not exp(-mean tau): the two-way transmittance is above the product of the one-way
    assert (t2 >= t1 ** 2 - 1e-12).all() and (t2 > 1.5 * t1 ** 2)[lbl.mask].any()
    assert lbl.transmittance(np.ones((2, 3))).shape == (2, 3, gas_lut.wl.size)


def test_o2_lbl_against_reptran(gas_lut):
    """Equivalent widths of the line-by-line bands (air mass 1.155, 1013 hPa): O2-B 0.82 nm, O2-A 3.42 nm;
    the REPTRAN table used as a spectrum (on its grid) gives about twice (O2-B) and 1.3 times (O2-A) these values."""
    tg = {}
    for model in O2_MODELS:
        gas = GaseousTransmittance(gas_lut, zenith_angle=30, o2_model=model)
        gas.pressure = 1013.
        tg[model] = gas.Tgas_background()
        if model == 'lbl':
            tg[model] = tg[model] * gas.Tgas_o2_lbl()
    assert equivalent_width(tg['lbl'], 685, 697) == pytest.approx(0.82, abs=0.02)
    assert equivalent_width(tg['lbl'], 758, 772) == pytest.approx(3.42, abs=0.05)
    assert equivalent_width(tg['reptran'], 685, 697) > 1.8 * equivalent_width(tg['lbl'], 685, 697)
    assert equivalent_width(tg['reptran'], 758, 772) > 1.2 * equivalent_width(tg['lbl'], 758, 772)
    # same transmittance outside the bands
    for lo, hi in [(400, 680), (705, 750), (780, 2500)]:
        np.testing.assert_allclose(tg['lbl'].sel(wl=slice(lo, hi)), tg['reptran'].sel(wl=slice(lo, hi)), rtol=1e-12)


def test_o2_lbl_pressure_air_mass_and_dimensions(gas_lut):
    gas = GaseousTransmittance(gas_lut, o2_model='lbl')
    gas.pressure, gas.air_mass = 1013., 2.
    t = gas.Tgas_o2_lbl()
    gas.pressure, gas.air_mass = 2026., 1.
    np.testing.assert_allclose(gas.Tgas_o2_lbl(), t)
    # time series of air masses and pressures, as in Process
    gas.air_mass = xr.DataArray([1., 1.5, 2.], coords={'time': [0, 1, 2]}, dims='time')
    gas.pressure = xr.DataArray([1013., 1013., 1013.], coords={'time': [0, 1, 2]}, dims='time')
    tg = gas.get_gaseous_transmittance()
    assert set(tg.dims) == {'time', 'wl'}
    np.testing.assert_allclose(gas.Tgas_o2_lbl().isel(time=2), t)
    with pytest.raises(ValueError):
        GaseousTransmittance(gas_lut, o2_model='hitran')


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
