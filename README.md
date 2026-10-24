<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Tristanovsk/radcalnet_oc/main/docs/source/_static/radcalnet-oc-logo-dark.png">
  <img alt="radcalnet_oc" src="https://raw.githubusercontent.com/Tristanovsk/radcalnet_oc/main/docs/source/_static/radcalnet-oc-logo.png" width="260">
</picture>

[![PyPI](https://img.shields.io/pypi/v/radcalnet-oc?logo=pypi&logoColor=white)](https://pypi.org/project/radcalnet-oc/)
[![Python](https://img.shields.io/pypi/pyversions/radcalnet-oc?logo=python&logoColor=white)](https://pypi.org/project/radcalnet-oc/)
[![CI](https://img.shields.io/github/actions/workflow/status/Tristanovsk/radcalnet_oc/ci.yml?branch=main&label=CI&logo=github)](https://github.com/Tristanovsk/radcalnet_oc/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/readthedocs/radcalnet-oc?logo=readthedocs&logoColor=white)](https://radcalnet-oc.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![LUT DOI](https://img.shields.io/badge/LUT%20DOI-10.5281%2Fzenodo.23394202-blue?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.23394202)

# radcalnet_oc

**Simulation of the top-of-atmosphere signal above AERONET-OC sites, for the calibration and
validation (Cal/Val) of passive optical satellite sensors over ocean and coastal waters.**

`radcalnet_oc` propagates the water-leaving radiance measured at the surface, typically by AERONET-OC
stations, up to the top of the atmosphere (TOA). It accounts for gaseous absorption, Rayleigh and
aerosol scattering (OPAC aerosol models) and the reflection of the sky by the wind-roughened sea
surface, from radiative transfer look-up tables:

$$
R_{toa}(\lambda) = T_g^{\downarrow} T_g^{\uparrow}
\left[ \pi\, R_{rs}(\lambda)\, T^{\downarrow}(\lambda)\, t^{\uparrow}(\lambda) + R_{atm}(\lambda) \right]
$$

The simulated hyperspectral TOA reflectance (350–2500 nm) is then convolved with the spectral
responses of the satellite bands (Sentinel-2, Sentinel-3, Landsat...) for comparison with the satellite
data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Tristanovsk/radcalnet_oc/main/docs/source/_static/slide_boa2toa_dark.png">
  <img alt="Propagation of the AERONET-OC measurements from the surface to the top of atmosphere and comparison with satellite data" src="https://raw.githubusercontent.com/Tristanovsk/radcalnet_oc/main/docs/source/_static/slide_boa2toa.png">
</picture>

📖 **Documentation:** <https://radcalnet-oc.readthedocs.io> (methods, processing chain, tutorials,
API reference).

## Installation

Python >= 3.10 is required (3.10, 3.11 and 3.12 tested), e.g. in a conda environment:

```bash
conda create -n radcalnet_oc python=3.12
conda activate radcalnet_oc
pip install radcalnet_oc
```

or from a clone of this repository: `pip install .`

### Look-up table

The radiative transfer TOA look-up table (`toa_lut_opac_wind_v3.nc`, about 900 MB) is not part of the
package. It is downloaded from [Zenodo](https://doi.org/10.5281/zenodo.23394202) (checksum verified)
the first time the look-up tables are loaded, into `~/.local/share/radcalnet_oc/lut` by default. It is
the look-up table of [GRS](https://github.com/CNES/GRSprocessor): if you already have it, point
radcalnet_oc to its folder instead.

```bash
radcalnet_oc_config                         # configuration in use and LUT check
radcalnet_oc_config --download              # download the LUT now (e.g. before working offline)
radcalnet_oc_config --init /path/to/lut     # use an existing LUT folder (e.g. the GRS one)
```

The LUT folder can also be set with the environment variable `RADCALNET_OC_DATA_ROOT`, and the
automatic download switched off with `RADCALNET_OC_NO_DOWNLOAD=1`. Without the full look-up table, a
light version shipped with the package is used, with a lower accuracy.

## Quick start

### Command line

```bash
radcalnet_oc AAOT_input.nc --vza 0,10 --azi 90 --bands 443,490,560,665,865 --fwhm 20
```

The input is a NetCDF time series of surface measurements (`Rrs`, `sza`, `day_of_year`, `aot550`,
`pressure`, `tcwv`, `tco3`, `tcno2`...; see the
[usage page](https://radcalnet-oc.readthedocs.io/en/latest/usage.html)). The output holds the TOA
reflectance `Rtoa`, the atmosphere reflectance `Ratm`, the irradiances `Ed` and `E0`, and their band
averages with `--bands`. Run `radcalnet_oc --help` for all the options.

### Python

```python
import numpy as np
import xarray as xr
from importlib.resources import files
import radcalnet_oc as radoc

# example input shipped with the package
input_db = xr.open_dataset(files('radcalnet_oc.data.template') / 'template_clear_water.nc')

# viewing geometry of the satellite
vza = xr.DataArray([0., 10.], coords={'vza': [0., 10.]}, dims='vza')
azi = xr.DataArray([90.], coords={'azi': [90.]}, dims='azi')

process = radoc.Process(input_db=input_db, vza=vza, azi=azi)
process.execute()
Rtoa = process.radcalnet_db.Rtoa

# band-averaged TOA reflectance
spectral = radoc.Spectral(central_wl=np.array([443., 490., 560., 665., 865.]), fwhm=20.)
Rtoa_bands = spectral.convolve2(Rtoa).Rtoa
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The unit tests use the light look-up table of the package and never download the full one. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the tests, the look-up tables and the release recipe, and
[CHANGELOG.md](CHANGELOG.md) for the version history.

## License

Apache License 2.0, see [LICENSE](LICENSE).

## Author

Tristan Harmel (<tristan.harmel@gmail.com>)
