"""Reference simulations of the regression test (``test_regression.py``).

Band-averaged outputs of the clear-water template with the light TOA LUT of the package, for the default
settings and for the decoupled gaseous transmittance with the REPTRAN O2 (settings before version 1.1).
The bands cover the window regions and the O2, H2O and cirrus absorption bands.

Update the reference file after a deliberate change of the results::

    RADCALNET_OC_NO_DOWNLOAD=1 python tests/reference.py

(``test_regression.py`` hides the configuration of the computer and uses the light LUT; this script does
the same, see ``main``.)
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

REFERENCE_FILE = Path(__file__).parent / 'data' / 'reference_clear_water.nc'
BANDS = np.array([412., 443., 490., 560., 620., 665., 687., 709., 754., 761., 779., 865., 885., 900., 940.,
                  1020., 1240., 1375., 1610., 2190.])
FWHM = 10.
VARIABLES = ['Rtoa', 'Ratm', 'Ed']
#: name: keyword arguments of Process
CASES = {'default': {}, 'decoupled_reptran': {'gas_coupling': 'decoupled', 'o2_model': 'reptran'}}
N_TIME = 4


def simulate():
    """Band-averaged outputs of all the cases, Dataset with variables ``<variable>_<case>``."""
    from radcalnet_oc import Process, Spectral
    from radcalnet_oc.process import template
    vza = xr.DataArray([0., 10.], coords={'vza': [0., 10.]}, dims='vza')
    azi = xr.DataArray([90.], coords={'azi': [90.]}, dims='azi')
    input_db = xr.open_dataset(template).isel(time=slice(0, N_TIME))
    spectral = Spectral(central_wl=BANDS, fwhm=FWHM)
    out = xr.Dataset()
    for case, kwargs in CASES.items():
        process = Process(input_db=input_db, vza=vza, azi=azi, **kwargs)
        process.execute()
        db = process.radcalnet_db
        for name in VARIABLES:
            band = spectral.convolve2(db[name].rename(name), expon=2)
            band = band[name] if isinstance(band, xr.Dataset) else band
            out[f'{name}_{case}'] = band.reset_coords(drop=True).rename(f'{name}_{case}')
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # same isolation as the tests: no user configuration, light LUT, nothing downloaded
        os.environ.update(RADCALNET_OC_CONFIG=str(Path(tmp) / 'config.yml'),
                          RADCALNET_OC_DATA_ROOT=str(Path(tmp) / 'lut'), RADCALNET_OC_NO_DOWNLOAD='1')
        (Path(tmp) / 'config.yml').write_text('')
        ref = simulate()
    from radcalnet_oc.process import package_version
    ref.attrs = {'title': 'radcalnet_oc regression reference, clear-water template, light TOA LUT',
                 'radcalnet_oc_version': package_version(), 'fwhm_nm': FWHM}
    REFERENCE_FILE.parent.mkdir(exist_ok=True)
    ref.to_netcdf(REFERENCE_FILE)
    print(f'{REFERENCE_FILE}: {list(ref.data_vars)}')


if __name__ == '__main__':
    main()
