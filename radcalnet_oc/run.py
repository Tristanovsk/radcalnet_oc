''' Executable to simulate the top-of-atmosphere signal from a time series of surface measurements

Usage:
  radcalnet_oc <input_file> [--output <output_file>] [--vza angles] [--azi angles]
   [--aerosol_combination props] [--solar_database name] [--rrs_name name]
   [--bands wavelengths] [--fwhm fwhm] [--expon expon] [--no_clobber]
  radcalnet_oc -h | --help
  radcalnet_oc -v | --version

Options:
  -h --help        Show this screen.
  -v --version     Show version.

  <input_file>     NetCDF file of the surface measurements (time series with the variables
                   Rrs, sza, day_of_year, aot550, pressure, tcwv, tco3, tcno2...)

  -o <output_file>, --output <output_file>  Output NetCDF file
                   (by default: <input_file basename>_radcalnet_oc.nc in the current directory)
  --vza angles     Viewing zenith angles (deg), comma-separated [default: 0]
  --azi angles     Relative azimuth angles (deg), comma-separated [default: 0]
  --aerosol_combination props  Proportion of each aerosol model of config.yml, comma-separated
                   (models: ARCT_rh70, COAV_rh70, DESE_rh70, MACL_rh70, URBA_rh70, WASO_rh0;
                   by default: proportions of config.yml)
  --solar_database name  Extraterrestrial solar spectrum: tsis, thuillier, gueymard or kurucz
                   [default: tsis]
  --rrs_name name  Name of the remote-sensing reflectance variable in <input_file> [default: Rrs]
  --bands wavelengths  Central wavelengths (nm) of sensor bands, comma-separated: the band-averaged
                   Rtoa, Ratm, Ed, E0 and Rrs are added to the output (variables <name>_bands,
                   dimension wl_band)
  --fwhm fwhm      Full width at half maximum (nm) of the bands: one value for all the bands or
                   one per band, comma-separated [default: 10]
  --expon expon    Exponent of the super-Gaussian spectral responses (2: Gaussian) [default: 3]
  --no_clobber     Do not process <input_file> if <output_file> already exists.

  Example:
      radcalnet_oc AAOT_input.nc --vza 0,10 --azi 90 --bands 443,490,560,665,865 --fwhm 20

  Exit status: 0 on success (or output skipped with --no_clobber), 1 on error.
'''

import logging
import sys

from pathlib import Path
from docopt import docopt

from . import __package__, __version__

BAND_VARIABLES = ['Rtoa', 'Ratm', 'Ed', 'E0', 'Rrs']


def floats(text):
    '''Parse a comma-separated list of numbers.'''
    return [float(v) for v in text.split(',') if v.strip()]


def main(argv=None):
    '''Entry point of the ``radcalnet_oc`` command; returns the exit status.'''
    # -h / -v are handled (and exit) here
    args = docopt(__doc__, argv=argv, version=__package__ + '_' + __version__)

    logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s', level=logging.INFO, force=True)

    import numpy as np
    import xarray as xr
    from .process import Process, AEROSOL_MODELS
    from .lut import Spectral

    input_file = Path(args['<input_file>'])
    output_file = args['--output']
    if output_file is None:
        output_file = Path.cwd() / (input_file.stem + '_radcalnet_oc.nc')
    output_file = Path(output_file)

    if output_file.exists() and args['--no_clobber']:
        logging.info(f'{output_file} already exists; skip')
        return 0

    try:
        vza = floats(args['--vza'])
        azi = floats(args['--azi'])
        vza = xr.DataArray(vza, coords={'vza': vza}, dims='vza')
        azi = xr.DataArray(azi, coords={'azi': azi}, dims='azi')

        aerosol_combination = args['--aerosol_combination']
        if aerosol_combination is not None:
            aerosol_combination = floats(aerosol_combination)
            if len(aerosol_combination) != len(AEROSOL_MODELS):
                raise ValueError(f'--aerosol_combination needs {len(AEROSOL_MODELS)} values '
                                 f'(one per model: {", ".join(AEROSOL_MODELS)})')

        logging.info(f'process {input_file}')
        input_db = xr.open_dataset(input_file)
        process = Process(input_db=input_db, vza=vza, azi=azi,
                          solar_database=args['--solar_database'],
                          Rrs_name=args['--rrs_name'])
        if aerosol_combination is None:
            process.execute()
        else:
            process.execute(aerosol_combination=aerosol_combination)
        output = process.radcalnet_db

        if args['--bands']:
            bands = np.array(floats(args['--bands']))
            fwhm = np.array(floats(args['--fwhm']))
            if fwhm.size == 1:
                fwhm = fwhm[0]
            elif fwhm.size != bands.size:
                raise ValueError('--fwhm needs one value or one value per band')
            spectral = Spectral(central_wl=bands, fwhm=fwhm)
            expon = float(args['--expon'])
            for name in BAND_VARIABLES:
                band = spectral.convolve2(output[name].rename(name), expon=expon)
                if isinstance(band, xr.Dataset):
                    band = band[name]
                band = band.rename(name + '_bands').rename(wl='wl_band')
                band.attrs = dict(output[name].attrs,
                                  description=output[name].attrs.get('description', name)
                                  + ', band-averaged (super-Gaussian responses)')
                output[name + '_bands'] = band
            output['fwhm'] = ('wl_band', spectral.fwhm.values)
            output['fwhm'].attrs = {'unit': 'nm', 'description': 'full width at half maximum of the bands'}

        # list attributes are not portable in NetCDF: write them as text
        for var in output.variables.values():
            for key, value in var.attrs.items():
                if isinstance(value, (list, tuple)):
                    var.attrs[key] = ', '.join(str(v) for v in value)
        output.attrs.update({'title': 'radcalnet_oc top-of-atmosphere simulation',
                             'source': f'radcalnet_oc {__version__}',
                             'input_file': str(input_file)})

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output.to_netcdf(output_file)
        logging.info(f'output written in {output_file}')
    except Exception:
        logging.error('radcalnet_oc processing failed', exc_info=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
