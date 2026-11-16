"""Hyperspectral reconstruction of AERONET-OC reflectances with invRrs (optional dependency)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip('invRrs')

from radcalnet_oc import Process
from radcalnet_oc.reconstruction import PARAMETERS, Reconstruction, reconstruct_rrs

from conftest import angles

AAOT = Path(__file__).parents[1] / 'examples/minimal_chain/data/AAOT_OCv3.lev20.nc'
pytestmark = pytest.mark.skipif(not AAOT.exists(), reason='AERONET-OC example file not available')


@pytest.fixture(scope='module')
def aaot():
    return xr.open_dataset(AAOT).isel(time=slice(0, 3)).load()


@pytest.fixture(scope='module')
def reconstructed(aaot):
    return reconstruct_rrs(aaot.Rrs_Lwn, aaot.SZA, chl=aaot['Chlorophyll-a'])


def test_reconstruction_fits_the_bands(aaot, reconstructed):
    rec = reconstructed
    assert rec.success.all() and (rec.rmsd < 3e-4).all()
    assert all(name in rec for name in PARAMETERS)
    assert float(rec.wl.max()) == 1150 and np.isfinite(rec.Rrs).all() and np.isfinite(rec.Rrs_nadir).all()
    vis = aaot.Rrs_Lwn.sel(wl=slice(400, 600))
    np.testing.assert_allclose(rec.Rrs.interp(wl=vis.wl), vis, rtol=0.1)


def test_reconstruction_skips_and_limits(aaot):
    rrs = aaot.Rrs_Lwn.copy()
    rrs[1, 3:] = np.nan                                    # too few bands: not reconstructed
    sza = aaot.SZA.copy()
    sza[2] = 85.                                           # sun too low: not reconstructed
    rec = reconstruct_rrs(rrs, sza, wl=np.arange(400., 901., 5.))
    assert np.isfinite(rec.Rrs.isel(time=0)).all()
    assert rec.Rrs.isel(time=[1, 2]).isnull().all() and not rec.success.isel(time=[1, 2]).any()
    with pytest.raises(ValueError):
        Reconstruction(wl=np.arange(350., 2501.))


def test_reconstruction_into_process(reconstructed):
    """Reconstructed Rrs (350-1150 nm) in the processing chain: extended to the short-wave infrared."""
    template = Process().input_db.isel(time=slice(0, 3))
    input_db = template.drop_vars('Rrs', errors='ignore').drop_dims('wl', errors='ignore')
    input_db = input_db.assign_coords(time=reconstructed.time.values)
    input_db['Rrs'] = reconstructed.Rrs
    process = Process(input_db=input_db, vza=angles('vza', [0.]), azi=angles('azi', [90.]))
    process.execute()
    rrs = process.radcalnet_db.Rrs
    assert np.isfinite(rrs).all() and (rrs.sel(wl=1240, method='nearest') > 0).all()
