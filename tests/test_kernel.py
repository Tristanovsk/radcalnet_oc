"""Atmospheric components: aerosol model retrieval, gaseous transmittance and radiometric utilities."""

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc import Aerosol, GaseousTransmittance, GasCoupling, Misc, Radiometry
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
# absorption-scattering coupling
# ------------------------------------
def test_gas_coupling_table():
    cp = GasCoupling()
    assert len(cp.windows) >= 5
    for w in cp.windows:
        for term in GasCoupling.TERMS:
            c = cp.coefficient(term, w, 3. if term != 'Ed' else 1.5, 0.1, 2., tau=1.)
            if cp.form(term, w) == 'eta':
                # path reflectance: shorter effective path; water-leaving term and irradiance: longer
                assert (0.2 < c < 0.9) if term == 'atm' else (0.99 < c < 1.1), (w, term, c)
            else:
                assert term == 'atm' and 0. < c < 2., (w, term, c)
    # clipped to the ranges of the simulations
    w = 'O2-A-P'
    r = cp.row('w', w)
    assert cp.coefficient('w', w, 100., 5.) == cp.coefficient('w', w, r.m_max, r.aot_max)
    eta = cp.coefficient('w', w, xr.DataArray([2.5, 3.], dims='vza'), xr.DataArray([0.1, 0.2], dims='time'))
    assert eta.dims == ('vza', 'time')
    with pytest.raises(ValueError):
        cp.coefficient('toa', w, 2., 0.1)
    with pytest.raises(KeyError):
        cp.coefficient('atm', 'H2O-1380', 2., 0.1)


def test_gas_coupling_first_version_table():
    """Table without form, H2O and absorption-strength terms (first version): effective air mass."""
    import pandas as pd
    table = pd.DataFrame([{'window': 'O2-A', 'wl_min': 757, 'wl_max': 772, 'term': t, 'c0': c0, 'c1': 0.,
                           'c2': 0., 'm_min': 2., 'm_max': 4., 'aot_min': 0.05, 'aot_max': 0.2}
                          for t, c0 in (('atm', 0.5), ('w', 1.02), ('Ed', 1.02))])
    cp = GasCoupling(table)
    assert cp.form('atm', 'O2-A') == 'eta'
    assert cp.eta('atm', 'O2-A', 3., 0.1, h2o=4., ) == 0.5


def test_gas_coupling_smooth_and_single_scattering():
    wl = np.linspace(750., 780., 3001)
    T = xr.DataArray(np.full((2, wl.size), 0.4), coords={'vza': [0, 1], 'wl': wl}, dims=('vza', 'wl'))
    np.testing.assert_allclose(GasCoupling.smooth(T), T)
    # alternating channels: smoothed to their mean, dimensions kept
    T = T.copy(data=np.where(np.arange(wl.size) % 2, 0.9, 0.1) * np.ones((2, 1)))
    s = GasCoupling.smooth(T)
    assert s.dims == T.dims
    # (odd number of cells in a 5 cm-1 window: 0.5 +- 0.4 / 29)
    np.testing.assert_allclose(s.isel(wl=slice(100, -100)), 0.5, atol=0.015)
    tss = GasCoupling.single_scattering(xr.DataArray([1., 0.5, 1e-6]))
    np.testing.assert_allclose(tss, [1., 0.5 / np.log(2), (1 - 1e-6) / (6 * np.log(10))], rtol=1e-10)


@pytest.mark.parametrize('o2_model', O2_MODELS)
@pytest.mark.parametrize('resolution', [GasCoupling.RESOLUTION, None])
def test_coupled_transmittance(gas_lut, o2_model, resolution):
    gas = GaseousTransmittance(gas_lut, o2_model=o2_model)
    cp = GasCoupling(resolution=resolution)
    m = xr.DataArray([2.5, 3.5], coords={'vza': [0, 1]}, dims='vza')
    gas.air_mass = m
    tg = gas.get_gaseous_transmittance()
    t_atm = gas.get_coupled_transmittance(m, 'atm', 0.1, cp)
    t_w = gas.get_coupled_transmittance(m, 'w', 0.1, cp)
    assert t_atm.dims == tg.dims and t_atm.sizes == tg.sizes
    inside = xr.zeros_like(tg.wl, dtype=bool)
    for a, b in cp.windows.values():
        inside = inside | ((tg.wl >= a) & (tg.wl <= b))
    # unchanged outside the windows, less (more) absorption for the path (water) term inside
    np.testing.assert_allclose(t_atm.where(~inside, 1.), tg.where(~inside, 1.), rtol=1e-12)
    np.testing.assert_allclose(t_w.where(~inside, 1.), tg.where(~inside, 1.), rtol=1e-12)
    absorbing = inside & (tg < 0.9)
    assert (t_atm > tg).where(absorbing, True).all()
    assert (t_w <= tg * (1 + 1e-12)).where(absorbing, True).all()
    if resolution is not None:
        # at the resolution of the fit, physical transmittances; channel by channel (diagnostic only), the
        # blend with k > 1 diverges in the saturated channels of the fine table
        assert np.isfinite(t_atm).all() and np.isfinite(t_w).all()
        assert (t_atm <= 1.).all() and (t_w >= 0.).all()
    # the air mass of the object is the geometric one after the call
    assert gas.air_mass is m


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
