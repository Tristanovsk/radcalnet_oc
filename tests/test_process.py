"""Processing chain and command line, on the templates of the package with the light TOA LUT."""

from importlib.resources import files

import numpy as np
import pytest
import xarray as xr

import radcalnet_oc
from radcalnet_oc import Process, run
from radcalnet_oc.config import LUT_FILE_BACKUP

from conftest import angles

TEMPLATES = files('radcalnet_oc.data.template')
CLEAR_WATER = TEMPLATES / 'template_clear_water.nc'
TURBID_WATER = TEMPLATES / 'template_turbid_water.nc'


@pytest.fixture(scope='module')
def clear_water():
    """Process run once on the clear-water template, two viewing angles, one azimuth."""
    process = Process(vza=angles('vza', [0., 10.]), azi=angles('azi', [90.]))
    process.execute()
    return process


@pytest.fixture(scope='module')
def clear_water_reptran():
    """As clear_water, with the REPTRAN O2 absorption (default before version 1.1)."""
    process = Process(vza=angles('vza', [0., 10.]), azi=angles('azi', [90.]), o2_model='reptran')
    process.execute()
    return process


def test_light_lut_in_use(clear_water):
    assert clear_water.lut.lut_file == LUT_FILE_BACKUP


def test_output_content(clear_water):
    db = clear_water.radcalnet_db
    for name in ['Rtoa', 'Ratm', 'Ed', 'E0', 'Rrs', 'D2', 'aerosol_combination',
                 'tcwv', 'tco3', 'tcno2', 'pressure', 'aot550']:
        assert name in db, name
    assert db.sizes['vza'] == 2 and db.sizes['time'] == 50
    assert 'sza' in db.coords
    assert db.Rtoa.attrs['unit'] == '-'


def test_provenance(clear_water):
    """The settings of the simulation are written as attributes of the output."""
    attrs = clear_water.radcalnet_db.attrs
    assert attrs['radcalnet_oc_version'].startswith(radcalnet_oc.__version__)
    assert attrs['o2_model'] == 'lbl' and attrs['gas_coupling'] == 'path_length'
    assert attrs['gas_coupling_table'] == 'gas_coupling_ppld.csv'
    assert attrs['rrs_extension'] == 'pure_water' and attrs['solar_database'] == 'tsis'
    assert attrs['default_wind'] == 2.
    assert attrs['toa_lut'] == 'toa_lut_opac_ultra_light.nc'
    assert attrs['toa_lut_doi'].startswith('none')


def test_provenance_decoupled():
    process = Process(input_db=xr.open_dataset(CLEAR_WATER).isel(time=slice(0, 2)), gas_coupling='decoupled',
                      o2_model='reptran')
    process.execute()
    attrs = process.radcalnet_db.attrs
    assert attrs['gas_coupling'] == 'decoupled' and attrs['gas_coupling_table'] == 'none'
    assert attrs['o2_model'] == 'reptran'


def test_wind_default(clear_water):
    """Without wind speed in the input, the default of the configuration (2 m s-1) is used."""
    np.testing.assert_array_equal(clear_water.radcalnet_db.wind, 2.)


def test_wind_from_input(clear_water):
    """The wind speed of the input is used, measurement by measurement (missing values: default)."""
    input_db = clear_water.input_db.isel(time=slice(0, 3)).copy()
    input_db['wind'] = ('time', [2., 8., np.nan])
    process = Process(input_db=input_db, vza=angles('vza', [0., 10.]), azi=angles('azi', [90.]))
    process.execute()
    db = process.radcalnet_db
    np.testing.assert_array_equal(db.wind, [2., 8., 2.])
    ref = clear_water.radcalnet_db.isel(time=slice(0, 3))
    np.testing.assert_allclose(db.Ratm.isel(time=[0, 2]), ref.Ratm.isel(time=[0, 2]), rtol=1e-6)
    assert not np.allclose(db.Ratm.isel(time=1), ref.Ratm.isel(time=1))


def test_rrs_extension():
    """Turbid water: Rrs extended beyond the input (1150 nm) with the pure-water model; unchanged inside the
    range of the input and zero beyond it with rrs_extension='zero'."""
    kwargs = dict(input_db=xr.open_dataset(TURBID_WATER).isel(time=slice(0, 2)),
                  vza=angles('vza', [0.]), azi=angles('azi', [90.]))
    zero = Process(rrs_extension='zero', **kwargs)
    zero.execute()
    process = Process(**kwargs)
    assert process.rrs_extension == 'pure_water'
    process.execute()
    db, dbz = process.radcalnet_db, zero.radcalnet_db
    vnir, swir = dict(wl=slice(350, 1140)), dict(wl=slice(1160, 2500))
    np.testing.assert_allclose(db.Rtoa.sel(**vnir), dbz.Rtoa.sel(**vnir))
    assert float(abs(dbz.Rrs.sel(**swir)).max()) == 0 and (db.Rrs.sel(**swir) > 0).all()
    assert (db.Rtoa.sel(wl=1240, method='nearest') > dbz.Rtoa.sel(wl=1240, method='nearest')).all()
    assert (db.bbp_nir > 0).all() and 'bbp_nir' not in dbz
    with pytest.raises(ValueError):
        Process(rrs_extension='linear', **kwargs)


def test_output_values(clear_water):
    db = clear_water.radcalnet_db.sel(wl=slice(400, 900))
    for name in ['Rtoa', 'Ratm', 'Ed', 'E0']:
        assert np.isfinite(db[name]).all(), name
    assert (db.Rtoa > 0).all() and (db.Rtoa < 1).all()
    # the water-leaving signal adds to the atmosphere signal
    assert (db.Rtoa >= db.Ratm).all()
    # the irradiance is attenuated by the atmosphere
    assert (db.Ed <= db.E0).all()
    # Rayleigh and aerosol path reflectance decreases from the blue to the near infrared
    assert (db.Ratm.sel(wl=443, method='nearest') > db.Ratm.sel(wl=865, method='nearest')).all()


def test_two_way_gaseous_transmittance(clear_water_reptran):
    """With the REPTRAN table, the Sun-surface-sensor transmittance is the product of the one-way ones."""
    p = clear_water_reptran
    product = p.Tg_d * p.Tg_u
    np.testing.assert_allclose(p.Tg_du.transpose(*product.dims), product, rtol=1e-10, atol=1e-14)


def test_o2_line_by_line(clear_water, clear_water_reptran):
    """Line-by-line O2 (default): same signal outside the O2 bands, shallower bands than with REPTRAN."""
    lbl, rep = clear_water.radcalnet_db, clear_water_reptran.radcalnet_db
    assert clear_water.gas_trans.o2_model == 'lbl'
    assert clear_water_reptran.gas_trans.o2_model == 'reptran'
    for lo, hi in [(400, 680), (705, 750), (780, 900)]:
        np.testing.assert_allclose(lbl.Rtoa.sel(wl=slice(lo, hi)), rep.Rtoa.sel(wl=slice(lo, hi)), rtol=1e-10)
        np.testing.assert_allclose(lbl.Ed.sel(wl=slice(lo, hi)), rep.Ed.sel(wl=slice(lo, hi)), rtol=1e-10)
    for lo, hi in [(686, 696), (759, 770)]:
        assert float(lbl.Ed.sel(wl=slice(lo, hi)).mean()) > float(rep.Ed.sel(wl=slice(lo, hi)).mean())
        assert float(lbl.Rtoa.sel(wl=slice(lo, hi)).mean()) > float(rep.Rtoa.sel(wl=slice(lo, hi)).mean())
    assert np.isfinite(lbl.Rtoa).all()


@pytest.mark.parametrize('o2_model', ['reptran', 'lbl'])
@pytest.mark.parametrize('coupling', ['effective_air_mass', 'path_length'])
def test_gas_coupling(clear_water, o2_model, coupling):
    """Effective air masses: same signal outside the absorption windows; brighter TOA reflectance (path
    reflectance less absorbed) and lower irradiance (longer skylight paths) in the bands."""
    kwargs = dict(vza=angles('vza', [0., 10.]), azi=angles('azi', [90.]), o2_model=o2_model)
    dec = Process(gas_coupling='decoupled', **kwargs)
    dec.execute()
    assert dec.gas_coupling == 'decoupled'
    assert dec.Tg_atm is dec.Tg_du and dec.Tg_Ed is dec.Tg_d
    process = Process(gas_coupling=coupling, **kwargs)
    assert process.coupling is not None
    process.execute()
    eff, ref = process.radcalnet_db, dec.radcalnet_db
    assert np.isfinite(eff.Rtoa).all()
    for lo, hi in [(400, 680), (700, 710), (740, 755), (840, 885), (1000, 1100)]:
        for name in ('Rtoa', 'Ratm', 'Ed'):
            np.testing.assert_allclose(eff[name].sel(wl=slice(lo, hi)), ref[name].sel(wl=slice(lo, hi)),
                                       rtol=1e-10)
    for lo, hi in [(686, 696), (759, 770), (815, 830), (930, 950)]:
        band = slice(lo, hi)
        assert float(eff.Ratm.sel(wl=band).mean()) > float(ref.Ratm.sel(wl=band).mean())
        assert float(eff.Rtoa.sel(wl=band).mean()) > float(ref.Rtoa.sel(wl=band).mean())
        assert float(eff.Ed.sel(wl=band).mean()) < float(ref.Ed.sel(wl=band).mean())
    with pytest.raises(ValueError):
        Process(gas_coupling='coupled')


def test_default_gas_coupling(clear_water):
    """The photon path-length model is the default coupling of the gaseous absorption with the scattering."""
    assert clear_water.gas_coupling == 'path_length'
    assert set(clear_water.coupling.table.form) == {'ppld'}


def test_turbid_water_brighter_in_red():
    """More water-leaving signal in the red for the turbid template than for the clear-water one."""
    rtoa = {}
    for name, template in [('clear', CLEAR_WATER), ('turbid', TURBID_WATER)]:
        process = Process(input_db=xr.open_dataset(template))
        process.execute()
        rtoa[name] = process.radcalnet_db.Rtoa.sel(wl=slice(640, 680)).mean('wl')
    assert (rtoa['turbid'] > rtoa['clear']).all()


def test_single_measurement():
    """A single measurement (time of length 1) gives the same spectrum as within a series."""
    input_db = xr.open_dataset(CLEAR_WATER).isel(time=[0, 1])
    kwargs = dict(vza=angles('vza', [10.]), azi=angles('azi', [90.]))
    series = Process(input_db=input_db, **kwargs)
    series.execute()
    single = Process(input_db=input_db.isel(time=[0]), **kwargs)
    single.execute()
    np.testing.assert_allclose(single.radcalnet_db.Rtoa, series.radcalnet_db.Rtoa.isel(time=0), rtol=1e-6)


def test_aerosol_combination_argument():
    process = Process()
    process.execute(aerosol_combination=[0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(process.radcalnet_db.aerosol_combination.values, [0, 0, 0, 1, 0, 0])


def test_aerosol_combination_varying_with_time():
    """A combination per time (e.g. retrieved with kernel.Aerosol) gives, at each time, the same result as
    the combination of this time used for the whole series."""
    input_db = xr.open_dataset(CLEAR_WATER).isel(time=[0, 1, 2])
    combs = [[0, 1., 0, 0, 0, 0], [0, 0, 0.5, 0.5, 0, 0], [0, 1., 0, 0, 0, 0]]   # (no WASO in the light LUT)
    comb = xr.DataArray(combs, coords={'time': input_db.time, 'model': radcalnet_oc.lut.AEROSOL_MODELS})
    kwargs = dict(input_db=input_db, vza=angles('vza', [0., 10.]), azi=angles('azi', [90.]))
    process = Process(**kwargs)
    process.execute(aerosol_combination=comb)
    rtoa = process.radcalnet_db.Rtoa
    for it in [0, 1]:
        constant = Process(**kwargs)
        constant.execute(aerosol_combination=combs[it])
        expected = constant.radcalnet_db.Rtoa.isel(time=it)
        np.testing.assert_allclose(rtoa.isel(time=it).transpose(*expected.dims), expected, rtol=1e-4)


# ------------------------------------
# command line
# ------------------------------------
def test_command_line(tmp_path):
    output = tmp_path / 'out.nc'
    status = run.main([str(CLEAR_WATER), '-o', str(output), '--vza', '0,10', '--azi', '90',
                       '--bands', '443,560,865', '--fwhm', '20'])
    assert status == 0
    with xr.open_dataset(output) as ds:
        assert dict(ds.Rtoa_bands.sizes) == {'time': 50, 'vza': 2, 'wl_band': 3}
        assert np.isfinite(ds.Rtoa_bands).all() and (ds.Rtoa_bands > 0).all()
        np.testing.assert_array_equal(ds.fwhm, [20, 20, 20])
        assert ds.attrs['source'].startswith(f'radcalnet_oc {radcalnet_oc.__version__}')
        assert ds.attrs['gas_coupling'] == 'path_length' and ds.attrs['o2_model'] == 'lbl'


def test_command_line_o2_model(tmp_path):
    output = tmp_path / 'out_lbl.nc'
    assert run.main([str(CLEAR_WATER), '-o', str(output), '--o2_model', 'lbl']) == 0
    with xr.open_dataset(output) as ds:
        assert np.isfinite(ds.Rtoa.sel(wl=slice(686, 696))).all()


def test_command_line_gas_coupling(tmp_path):
    output = tmp_path / 'out_eta.nc'
    assert run.main([str(CLEAR_WATER), '-o', str(output), '--gas_coupling', 'effective_air_mass']) == 0
    with xr.open_dataset(output) as ds:
        assert np.isfinite(ds.Rtoa.sel(wl=slice(890, 990))).all()


def test_command_line_no_clobber(tmp_path):
    output = tmp_path / 'out.nc'
    output.write_text('existing')
    assert run.main([str(CLEAR_WATER), '-o', str(output), '--no_clobber']) == 0
    assert output.read_text() == 'existing'


@pytest.mark.parametrize('args', [['--aerosol_combination', '0.5,0.5'],
                                  ['--bands', '443,560', '--fwhm', '10,20,30'],
                                  ['--o2_model', 'hitran'],
                                  ['--gas_coupling', 'coupled']])
def test_command_line_errors(tmp_path, args):
    output = tmp_path / 'bad.nc'
    assert run.main([str(CLEAR_WATER), '-o', str(output), *args]) == 1
    assert not output.exists()


def test_command_line_version(capsys):
    with pytest.raises(SystemExit):
        run.main(['--version'])
    assert radcalnet_oc.__version__ in capsys.readouterr().out


@pytest.fixture(scope='module')
def glint_run():
    """Process with the direct sunglint on 4 measurements, from near-specular to far geometries."""
    input_db = xr.open_dataset(CLEAR_WATER).isel(time=slice(0, 4))
    process = Process(input_db=input_db, vza=angles('vza', [5., 30.]), azi=angles('azi', [0., 60., 180.]))
    process.execute(sunglint='cm_iso_mix')
    return process


def test_sunglint_output(glint_run):
    """compute_sunglint adds the glint variables without changing Rtoa; Rtoa_glint = Rtoa + Rglint."""
    db = glint_run.radcalnet_db
    for name in ['Rglint', 'Rg_surface', 'Rtoa_glint', 'glint_angle', 'glint_fraction', 'glint_flags']:
        assert name in db, name
    assert db.glint_flags.dtype == np.uint8
    assert db.glint_flags.attrs['flag_meanings'].split() == list(radcalnet_oc.GLINT_FLAGS)
    np.testing.assert_allclose(db.Rtoa_glint, db.Rtoa + db.Rglint)
    assert (db.Rglint >= 0).all()


def test_sunglint_toa_formula(glint_run):
    """Rglint = Tg_d Tg_u exp(-(tau_r + tau_a) (1/mu0 + 1/muv)) Rg_surface, as in the matchup script."""
    p = glint_run
    db = p.radcalnet_db
    t, v, a = 0, 1, 0
    sza, vza = float(p.sza[t]), float(p.vza[v])
    Rg = radcalnet_oc.Sunglint(sza, vza, 180. - float(p.azi[a])).sunglint(float(p.wind[t]), stats='cm_iso_mix')[0]
    wl = 1610.
    tau_r = float(p.lut.rot.interp(wl=wl)) * float(p.input_db.pressure[t]) / 1013.25
    tau_a = float(p.lut.aot_lut.interp(aot_ref=float(p.aot550[t])).interp(wl=wl, method='quadratic'))
    Tg = float((p.Tg_d * p.Tg_u).isel(time=t).sel(vza=vza).sel(wl=wl, method='nearest'))
    expected = Tg * np.exp(-(tau_r + tau_a) * (1 / np.cos(np.radians(sza)) + 1 / np.cos(np.radians(vza)))) * Rg
    got = float(db.Rglint.isel(time=t, vza=v, azi=a).sel(wl=wl, method='nearest'))
    assert got == pytest.approx(expected, rel=1e-3)


def test_sunglint_flags(glint_run):
    """Glint flags: cumulative levels of the glint fraction at 865 nm and near-specular geometry."""
    db = glint_run.radcalnet_db
    F = radcalnet_oc.GLINT_FLAGS
    flags, frac, ang = db.glint_flags.values, db.glint_fraction.values, db.glint_angle.values
    np.testing.assert_array_equal((flags & F['glint_low']) > 0, frac >= 0.01)
    np.testing.assert_array_equal((flags & F['glint_moderate']) > 0, frac >= 0.1)
    np.testing.assert_array_equal((flags & F['glint_high']) > 0, frac >= 0.5)
    np.testing.assert_array_equal((flags & F['glint_near_specular']) > 0, ang < 30.)
    # the levels are nested, and far from the specular direction the glint is at most low (heavy tail)
    assert np.all((flags & F['glint_high'] == 0) | (flags & F['glint_moderate'] > 0))
    assert np.all((flags & F['glint_moderate'])[ang > 50.] == 0)
    # near the specular direction (vza 30, azi 0, sza ~ 30-60) the glint dominates
    assert (db.glint_flags.sel(vza=30., azi=0.) & F['glint_moderate'] > 0).any()


def test_sunglint_uncertainty(glint_run):
    """Glint uncertainty: spectral Rglint_unc >= 0, spectral flag glint_uncertain where Rglint_unc / Rtoa_glint
    > 5 %, glint_unc_rel its largest value in the windows; near the specular direction the glint is uncertain."""
    db = glint_run.radcalnet_db
    for name in ['Rglint_unc', 'glint_uncertain', 'glint_unc_rel', 'glint_unc_wl']:
        assert name in db, name
    assert (db.Rglint_unc >= 0).all()
    assert db.glint_uncertain.dtype == np.uint8 and 'wl' in db.glint_uncertain.dims
    np.testing.assert_array_equal(db.glint_uncertain.values, (db.Rglint_unc / db.Rtoa_glint).values > 0.05)
    assert db.glint_uncertain.attrs['flag_meanings'] == 'glint_ok glint_uncertain'
    assert (db.glint_uncertain.sel(vza=30., azi=0.).sel(wl=865., method='nearest') == 1).all()
    # the uncertain fraction of the bands grows from the blue to the SWIR (decreasing TOA signal)
    frac = db.glint_uncertain.mean(['time', 'vza', 'azi'])
    assert float(frac.sel(wl=2190., method='nearest')) >= float(frac.sel(wl=443., method='nearest'))
    # the windows exclude the saturated absorption bands
    assert not np.isin(np.round(db.glint_unc_wl.values), np.arange(1355, 1415)).any()


def test_uncertainty_budget(glint_run):
    """Uncertainty budget: one term per input (central differences through Process) and the glint term, combined
    in quadrature; a zero input uncertainty gives a zero term; the nominal simulation and the LUT are unchanged."""
    from radcalnet_oc.uncertainty import uncertainty_budget
    p = glint_run
    R0 = p.radcalnet_db.Rtoa.copy()
    ratm_lut = p.lut.Rdiff_lut.copy()
    db = uncertainty_budget(p, inputs={'aot550': ('abs', 0.02), 'tco3': ('rel', 0.)})
    for name in ['u_aot550', 'u_tco3', 'u_glint', 'u_total', 'u_total_rel']:
        assert name in db, name
    assert float(db.u_tco3.max()) == 0.
    assert float(db.u_aot550.sel(wl=865., method='nearest').min()) > 0.
    np.testing.assert_allclose(db.u_total, np.sqrt(db.u_aot550 ** 2 + db.u_tco3 ** 2 + db.u_glint ** 2))
    np.testing.assert_allclose(db.u_total_rel, db.u_total / db.Rtoa_glint)
    xr.testing.assert_identical(db.Rtoa, R0)
    xr.testing.assert_identical(p.lut.Rdiff_lut, ratm_lut)
    # the AOT term grows with the path reflectance from the blue to the near infrared (relative to Rtoa)
    rel = (db.u_aot550 / db.Rtoa).isel(time=0, vza=0, azi=1)
    assert float(rel.sel(wl=865., method='nearest')) > float(rel.sel(wl=443., method='nearest'))


def test_rayleigh_pressure():
    """Pressure scaling of the Rayleigh scattering: nil at the pressure of the LUT, Ratm larger at higher pressure,
    by less than the Rayleigh optical thickness ratio, and mostly in the blue; diffuse transmittances reduced."""
    input_db = xr.open_dataset(CLEAR_WATER).isel(time=slice(0, 2))
    geom = dict(vza=angles('vza', [10.]), azi=angles('azi', [90.]))
    runs = {}
    for P in [1015.2, 1035.5]:
        db = input_db.copy()
        db['pressure'] = db.pressure * 0 + P
        for flag in [False, True]:
            p = Process(input_db=db, rayleigh_pressure=flag, **geom)
            p.execute()
            runs[(P, flag)] = p
    xr.testing.assert_allclose(runs[(1015.2, True)].radcalnet_db.Rtoa, runs[(1015.2, False)].radcalnet_db.Rtoa)
    hi, ref = runs[(1035.5, True)].radcalnet_db, runs[(1035.5, False)].radcalnet_db
    rel = ((hi.Ratm - ref.Ratm) / ref.Ratm).isel(time=0)
    r443, r2190 = float(rel.sel(wl=443., method='nearest')), float(rel.sel(wl=2190., method='nearest'))
    assert 0.01 < r443 < 0.02 and 0 < r2190 < r443
    assert hi.attrs['rayleigh_pressure'] == 1 and ref.attrs['rayleigh_pressure'] == 0
    p = runs[(1035.5, True)]
    assert float(p.Tra_d.isel(time=0).sel(wl=443., method='nearest')) < \
        float(runs[(1035.5, False)].Tra_d.isel(time=0).sel(wl=443., method='nearest'))
