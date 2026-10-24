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


def test_turbid_water_brighter_in_red():
    """More water-leaving signal in the red for the turbid template than for the clear-water one."""
    rtoa = {}
    for name, template in [('clear', CLEAR_WATER), ('turbid', TURBID_WATER)]:
        process = Process(input_db=xr.open_dataset(template))
        process.execute()
        rtoa[name] = process.radcalnet_db.Rtoa.sel(wl=slice(640, 680)).mean('wl')
    assert (rtoa['turbid'] > rtoa['clear']).all()


def test_aerosol_combination_argument():
    process = Process()
    process.execute(aerosol_combination=[0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(process.radcalnet_db.aerosol_combination.values, [0, 0, 0, 1, 0, 0])


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
        assert ds.attrs['source'] == f'radcalnet_oc {radcalnet_oc.__version__}'


def test_command_line_no_clobber(tmp_path):
    output = tmp_path / 'out.nc'
    output.write_text('existing')
    assert run.main([str(CLEAR_WATER), '-o', str(output), '--no_clobber']) == 0
    assert output.read_text() == 'existing'


@pytest.mark.parametrize('args', [['--aerosol_combination', '0.5,0.5'],
                                  ['--bands', '443,560', '--fwhm', '10,20,30']])
def test_command_line_errors(tmp_path, args):
    output = tmp_path / 'bad.nc'
    assert run.main([str(CLEAR_WATER), '-o', str(output), *args]) == 1
    assert not output.exists()


def test_command_line_version(capsys):
    with pytest.raises(SystemExit):
        run.main(['--version'])
    assert radcalnet_oc.__version__ in capsys.readouterr().out
