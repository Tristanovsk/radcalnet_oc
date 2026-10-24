# Changelog

All notable changes to radcalnet_oc are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-10-24

### Changed
- Python >= 3.10 required (was >= 3.9, not supported by the xarray versions the code needs); the CI
  tests Python 3.10, 3.11 and 3.12.

### Removed
- Unused light transmittance LUT `transmittance_lut_opac_wind_light_v2.nc` (0.5 MB) from the package.

## [1.0.0] - 2026-10-24

First release published on PyPI (`pip install radcalnet_oc`).

### Added
- Automatic download of the full TOA look-up table (`toa_lut_opac_wind_v3.nc`, about 900 MB) from
  [Zenodo](https://doi.org/10.5281/zenodo.23394202) (record shared with GRS), MD5 checked, the first
  time the LUTs are loaded (`radcalnet_oc.config.ensure_toa_lut`); switched off with
  `zenodo: auto_download: false` or `RADCALNET_OC_NO_DOWNLOAD=1`. Without it, the light LUT of the
  package is used, with a warning.
- `radcalnet_oc.config` module and `radcalnet_oc_config` command: prints the configuration in use and
  checks the TOA LUT (exit code 1 if it is missing); `--init [DATA_ROOT]` writes the user configuration
  file; `--download` downloads the TOA LUT.
- `radcalnet_oc` command line: TOA simulation of a NetCDF time series, with optional band-averaged
  outputs (super-Gaussian spectral responses).
- In-memory cache of the TOA radiance tables shared by the `LUT` objects, and separable interpolation
  of the geometry (faster `lut_preparation`).
- Unit tests (`tests/`, about 20 s, light LUT, nothing downloaded): configuration and LUT download,
  spectral functions and responses, auxiliary data, LUT preparation, aerosol retrieval, gaseous
  transmittance, processing chain and command line.
- PyPI packaging: wheel with the light TOA LUT, the transmittance and gaseous LUTs, the auxiliary data,
  the input templates and the AERONET-OC site information; release recipe in
  [CONTRIBUTING.md](https://github.com/Tristanovsk/radcalnet_oc/blob/main/CONTRIBUTING.md) and GitHub
  workflow `pypi.yml` (trusted publishing on `v*` tags).
- GitHub Actions CI: unit tests, build and check of the package, command line run from the installed
  wheel, documentation build with warnings as errors.
- Read the Docs documentation: installation and configuration, usage, methods (radiative transfer
  equations), processing chain, tutorials (downwelling irradiance, gaseous transmittance, solar
  irradiance, TOA atmosphere radiation, Sentinel-2 and Sentinel-3 matchups, sunglint), API reference
  and this changelog.

### Changed
- Configuration read at run time: packaged defaults (`radcalnet_oc/config.yml`, no longer to be
  edited), then the user file `~/.config/radcalnet_oc/config.yml` (or `RADCALNET_OC_CONFIG`), then
  `RADCALNET_OC_DATA_ROOT` for the LUT folder. The default LUT folder is
  `~/.local/share/radcalnet_oc/lut`; set an existing one (e.g. the GRS one) with
  `radcalnet_oc_config --init /path/to/lut`.
- `LUT(lut_file=None)`: the TOA LUT of the configuration by default.
- `Process()` without input uses `template_clear_water.nc`.
- Docstrings of the whole package.

### Fixed
- TOA LUT interpolation at a node of the azimuth grid (any multiple of 5 deg, including the default
  `azi=[0]`): NaN Rayleigh reflectance and null path reflectance.
- Crash (segmentation fault or `NetCDF: HDF error`) when several `Process` or `LUT` objects were created
  in a session: the LUT files are opened once per process and shared, the solar spectra are loaded in
  memory.
- `Gamma2sigma` returned FWHM / (2 sqrt(ln 2)): Gaussian spectral responses about 1.41 times too wide.
- `GaseousTransmittance` initialized the air mass as cos(zenith) instead of 1 / cos(zenith).
- `Spectral.convolve2` / `convolve` with more than one dimension besides `wl`, and `threshold` ignored.
- Template and AERONET-OC data files missing from the package (`Aeronet()` failed after
  `pip install`).
- Version declared in a single place (`radcalnet_oc.__version__`).

## 0.0.3 - 2026-09-02 (not tagged)

### Added
- Sunglint module (`Sunglint`, Cox and Munk statistics), upward transmittance for irradiance or direct
  light.

## 0.0.2 - 2026-03-31 (not tagged)

### Added
- Multiangular option of the TOA simulation; super-Gaussian spectral functions.

## [0.0.1] - 2025-03-17

- Initial version.

[1.0.1]: https://github.com/Tristanovsk/radcalnet_oc/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Tristanovsk/radcalnet_oc/releases/tag/v1.0.0
[0.0.1]: https://github.com/Tristanovsk/radcalnet_oc/releases/tag/v0.0.1
