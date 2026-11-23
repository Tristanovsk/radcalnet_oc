"""AERONET-OC inputs of the processor (radcalnet_oc.aeronet_oc.aeronet_inputs) and RadCalNet-OC daily products
(radcalnet_oc.products).

The products are made for one day of AAOT (example file of the repository, served as a fake AERONET-OC
database) with the full TOA LUT (skipped without it or without invRrs); the inputs are also read from the real
database when it is installed and built.
"""

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from radcalnet_oc import GLINT_FLAGS
from radcalnet_oc import aeronet_oc, config, products

from conftest import FULL_LUT, requires_full_lut

AAOT = Path(__file__).parents[1] / 'examples/minimal_chain/data/AAOT_OCv3.lev20.nc'
INPUTS = ['tcwv', 'tco3', 'tcno2', 'pressure', 'wind', 'aot550', 'aot865', 'ang_exp', 'naot', 'chl', 'sza_aeronet',
          'Rrs_Lwn']


@pytest.fixture(scope='module')
def aaot():
    if not AAOT.exists():
        pytest.skip('AERONET-OC example file not available')
    with xr.open_dataset(AAOT) as ds:
        return ds.load()


@pytest.fixture
def fake_db(monkeypatch, aaot):
    """aeronet_oc_db replaced by the example file of AAOT (one day, 2 July 2025, legacy variable names)."""
    def open_site(site, level='lev20', legacy_names=False):
        assert legacy_names
        return aaot.copy(deep=True)
    monkeypatch.setattr(aeronet_oc, '_import_aeronet_oc_db', lambda: types.SimpleNamespace(open_site=open_site))


def test_aeronet_inputs(fake_db, aaot):
    raw = aaot
    ds = aeronet_oc.aeronet_inputs('AAOT', 'lev20')
    assert all(v in ds for v in INPUTS)
    assert ds.sizes['time'] == raw.sizes['time']
    assert ds.naot.dims == ('time', 'wl_aeronet') and ds.Rrs_Lwn.dims == ('time', 'wl')
    np.testing.assert_allclose(ds.tcwv, raw['Total_Precipitable_Water(cm)'] * 10)
    np.testing.assert_allclose(ds.tco3, raw['Total_Ozone(Du)'] * aeronet_oc.TCO3_DU)
    np.testing.assert_allclose(ds.naot.sel(wl_aeronet=ds.wl_aeronet[0]) * ds.aot550,
                               raw['Aerosol_Optical_Depth'].sortby('wl').isel(wl=0), rtol=1e-6)
    np.testing.assert_allclose(ds.ang_exp, -np.log(ds.aot550 / ds.aot865) / np.log(550 / 865))
    assert ds.tcwv.attrs['units'] == 'kg m-2' and 'Angstrom' in ds.ang_exp.attrs['long_name']
    # time selection: dates include the whole day
    assert aeronet_oc.aeronet_inputs('AAOT', start='2025-07-02', end='2025-07-02').sizes['time'] == raw.sizes['time']
    assert aeronet_oc.aeronet_inputs('AAOT', start='2025-07-02 12:00').sizes['time'] < raw.sizes['time']
    assert aeronet_oc.aeronet_inputs('AAOT', end='2025-07-01').sizes['time'] == 0


def test_aeronet_inputs_without_database(monkeypatch):
    monkeypatch.setitem(sys.modules, 'aeronet_oc_db', None)
    with pytest.raises(ImportError, match='aeronet-oc-db update'):
        aeronet_oc.aeronet_inputs('AAOT')


def test_aeronet_inputs_from_database():
    aeronet_oc_db = pytest.importorskip('aeronet_oc_db')
    try:
        ds = aeronet_oc.aeronet_inputs('AAOT', 'lev20', start='2024-07-02', end='2024-07-02')
    except (OSError, FileNotFoundError, KeyError) as e:
        pytest.skip(f'AERONET-OC database not available ({aeronet_oc_db.default_root()}): {e}')
    assert ds.sizes['time'] > 0 and all(v in ds for v in INPUTS)
    assert (ds.time.dt.strftime('%Y-%m-%d') == '2024-07-02').all()


def test_glint_filter():
    flags = xr.DataArray(np.array([0, GLINT_FLAGS['glint_low'], GLINT_FLAGS['glint_moderate'],
                                   GLINT_FLAGS['glint_near_specular']], dtype=np.uint8), dims='time')
    unc = xr.DataArray(np.zeros((4, 3), np.uint8), coords={'wl': [350, 500, 1000]}, dims=('time', 'wl'))
    unc[1, 1] = 1
    day = xr.Dataset({'glint_flags': flags, 'glint_uncertain': unc})
    assert products.glint_ok(day).values.tolist() == [True, True, False, False]
    assert products.glint_ok(day, uncertain=True).values.tolist() == [True, False, False, False]
    sub = products.glint_filtered(day)
    assert sub.sizes['time'] == 2 and 'glint_moderate' in sub.attrs['glint_filter']
    assert 'glint_filter' not in day.attrs   # the attributes of the daily products are not changed


@requires_full_lut
def test_make_products(fake_db, tmp_path, monkeypatch):
    """Products of one day with the full TOA LUT (the light LUT has no water-soluble aerosol model for the
    retrieval of the aerosol mixture)."""
    pytest.importorskip('invRrs')
    monkeypatch.setenv('RADCALNET_OC_DATA_ROOT', str(Path(FULL_LUT).parent))
    config.load_config.cache_clear()
    summary = products.make_products('AAOT', 'lev20', name='VEIT', start='2025-07-01', end='2025-07-31',
                                     odir=str(tmp_path / 'products'))
    assert summary['months'] == 1 and summary['days'] == 1 and summary['times'] >= 2 and not summary['failed']
    day = pd.Timestamp('2025-07-02')
    f = products.daily_file(str(tmp_path / 'products' / 'daily'), 'VEIT', 'lev20', day)
    assert f.endswith('daily/VEIT/2025/07/02/VEIT_radcalnet-oc_lev20_20250702_inc10nm_res10nm.nc')
    assert Path(products.monthly_file(str(tmp_path / 'products' / 'monthly'), 'VEIT', 'lev20', 2025, 7)).exists()
    with xr.open_dataset(f) as ds:
        assert ds.sizes['time'] == summary['times']
        np.testing.assert_array_equal(ds.wl, products.CENTRAL_WL)
        for v in ['Rtoa', 'Rtoa_glint', 'Rglint', 'Rglint_unc', 'glint_flags', 'glint_uncertain', 'glint_angle',
                  'Rrs', 'aot550', 'tcwv', 'chl_invrrs']:
            assert v in ds, v
        np.testing.assert_allclose(ds.Rtoa_glint, ds.Rtoa + ds.Rglint, rtol=1e-6, atol=1e-9)
        assert ds.attrs['site'] == 'VEIT' and ds.attrs['aeronet_oc_site'] == 'AAOT'
        assert 'invrrs_version' in ds.attrs and 'glint_filter' not in ds.attrs
        assert ds.tco3.attrs['units'] == 'kg m-2' and 'Earth-Sun' in ds.D2.attrs['description']
        kept = int(products.glint_ok(ds).sum())
    assert summary['kept'] == kept
    ffile = products.daily_file(str(tmp_path / 'products' / 'daily_glint_filtered'), 'VEIT', 'lev20', day)
    assert Path(ffile).exists() == (kept > 0)

    # restart: the month is read from the cache and the existing files are kept
    again = products.make_products('AAOT', 'lev20', name='VEIT', start='2025-07-01', end='2025-07-31',
                                   odir=str(tmp_path / 'products'))
    assert again == summary


def test_make_products_no_data(fake_db, tmp_path):
    summary = products.make_products('AAOT', start='2020-01-01', end='2020-01-31', odir=str(tmp_path))
    assert summary == dict(months=0, days=0, times=0, kept=0, failed=[])


def test_cli_help():
    out = subprocess.run([sys.executable, '-m', 'radcalnet_oc.products', '--help'], capture_output=True, text=True)
    assert out.returncode == 0
    for option in ['--site', '--odir', '--cache', '--filter-flags', '--filter-uncertain', '--max-times']:
        assert option in out.stdout
    with pytest.raises(SystemExit):   # --odir is required without a default
        products.parser().parse_args(['--site', 'AAOT'])
    assert products.parser(odir='x').parse_args(['--site', 'AAOT']).odir == 'x'


def test_script_help():
    script = Path(__file__).parents[1] / 'scripts/radcalnet_products/make_products.py'
    if not script.exists():
        pytest.skip('scripts not available')
    out = subprocess.run([sys.executable, str(script), '--help'], capture_output=True, text=True)
    assert out.returncode == 0 and '/data/radcalnet/products/v' in out.stdout
