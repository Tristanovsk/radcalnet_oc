"""Fixtures of the unit tests.

The tests never download the TOA look-up table: the automatic download is switched off and the
configuration of the computer is replaced by an empty user file, so that the light LUT of the package is
used. The tests marked ``requires_full_lut`` use the full TOA LUT of the configuration of the computer
(or of the environment variable RADCALNET_OC_TEST_LUT) and are skipped if it is not there.
"""

import os

import numpy as np
import pytest
import xarray as xr

from radcalnet_oc import config

# full TOA LUT, looked for before the configuration of the computer is hidden by `isolated_config`
FULL_LUT = os.environ.get('RADCALNET_OC_TEST_LUT') or str(config.toa_lut_path())

requires_full_lut = pytest.mark.skipif(not os.path.isfile(FULL_LUT),
                                       reason=f'full TOA LUT not installed ({FULL_LUT}): '
                                              'radcalnet_oc_config --download')


@pytest.fixture(scope='session', autouse=True)
def isolated_session(tmp_path_factory):
    """Same isolation as `isolated_config` for the whole session: the session and module fixtures (e.g. a
    Process run shared by the tests of a module) are set up before the function fixtures, and must not
    read the configuration of the computer nor download the TOA LUT either."""
    root = tmp_path_factory.mktemp('session')
    (root / 'config.yml').write_text('')
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('RADCALNET_OC_CONFIG', str(root / 'config.yml'))
        monkeypatch.setenv('RADCALNET_OC_DATA_ROOT', str(root / 'lut'))
        monkeypatch.setenv('RADCALNET_OC_NO_DOWNLOAD', '1')
        config.load_config.cache_clear()
        yield
    config.load_config.cache_clear()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Configuration without the user file of the computer (empty user file, cache cleared), automatic
    download of the TOA LUT switched off, and LUT folder in the temporary folder of the test."""
    user_file = tmp_path / 'config.yml'
    user_file.write_text('')
    monkeypatch.setenv('RADCALNET_OC_CONFIG', str(user_file))
    monkeypatch.setenv('RADCALNET_OC_DATA_ROOT', str(tmp_path / 'lut'))
    monkeypatch.setenv('RADCALNET_OC_NO_DOWNLOAD', '1')
    config.load_config.cache_clear()
    yield user_file
    config.load_config.cache_clear()


def angles(name, values):
    """Viewing angles as expected by Process: DataArray with a dimension of the same name."""
    return xr.DataArray(np.array(values, dtype=float), coords={name: values}, dims=name)
