# Changelog

All notable changes to radcalnet_oc are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tutorial *Simulate the TOA signal from user settings* (`docs/source/tutorials/radcalnet_oc_toa_simulation.ipynb`),
  first of the documentation: input built from the settings of geometry, atmosphere (AOT, aerosol models,
  gas columns), sea-surface roughness (wind) and water reflectance, options of version 1.1 (`gas_coupling`,
  `o2_model`, `rrs_extension`), direct sunglint added to `Rtoa`, provenance attributes, sensitivity
  studies and convolution with sensor bands. It replaces *TOA atmosphere radiation*.
- Direct sunglint term $T_{dir} R_{sg}$ in the TOA equation of the README and of the documentation.

### Changed
- One folder for the rendered tutorials: `docs/source/notebook` is moved to `docs/source/tutorials` (the
  pages move from `notebook/<name>.html` to `tutorials/<name>.html`); the working notebooks of
  `docs/source/tutorials/examples` are moved out of the documentation, to `examples/`, without their
  three development versions of the rendered tutorials.
- Tutorials rerun with version 1.1 (outputs of 0.0.3 or 1.0.1 before), except the Sentinel-3 matchup
  (simulations of version 0.0.2 to be regenerated). *Gaseous transmittance*: the direct beam (line-by-line
  and REPTRAN O2) compared with the coupled transmittances of the path reflectance and of the water-leaving
  term, instead of the arbitrary scaling coefficients of H2O and CH4. *Solar irradiance* and *Downwelling
  irradiance*: total irradiance with the coupled transmittance `T_g^{E_d}` (the solar irradiance tutorial
  did not apply its gas columns and read one aerosol model of the raw transmittance table). *Sunglint*:
  `R_sg`, transmittance of the whole path `Tg_du`, Gaussian 3 nm smoothing. *Gaseous coupling*: presented
  as the study before version 1.1.

### Fixed
- `Process.execute` failed for a single measurement (time dimension of length 1) when squeezing the
  prepared look-up table.

## [1.1.0] - 2026-11-15

### Added
- Traceability of the outputs: global attributes of `Process.radcalnet_db` (and of the command-line output)
  `radcalnet_oc_version` (with the git commit of a development installation, e.g. `1.0.2+g6439281.dirty`,
  `process.package_version`), `o2_model`, `gas_coupling`, `gas_coupling_table`, `rrs_extension`,
  `solar_database`, `default_wind`, `toa_lut` and `toa_lut_doi` (`Process.provenance`). The `source`
  attribute of the command-line output gives the same version.
- Regression test on reference outputs (`tests/test_regression.py`): band-averaged `Rtoa`, `Ratm` and `Ed` of
  the clear-water template with the light LUT (20 bands of 10 nm, 412-2190 nm), default settings and
  decoupled / REPTRAN O2, against `tests/data/reference_clear_water.nc` (relative tolerance 1e-4). A
  deliberate change of the results updates the reference with `python tests/reference.py`.
- Shared AERONET-OC database (`aeronet_oc_db`, optional dependency `pip install radcalnet_oc[aeronet]`):
  `Aeronet.read_site(site, level)` reads the up-to-date AERONET-OC file of a site, `Aeronet.open_site`
  returns its netCDF dataset (with `Rrs_Lwn*`), and `Aeronet().aeronet_oc_site_info` comes from the database
  catalog (all sites, Sentinel-2 tile), with the packaged table as fallback. The location of the database is
  `$AERONET_OC_ROOT` or `~/.config/aeronet_oc_db/config.toml`. The AERONET-OC tutorials open the data from the
  database (`aeronet_oc_db.open_site(site, level, legacy_names=True)`, `aeronet_oc_db.site_info(level='lev15')`)
  instead of `/DATA/AERONET/netcdf/OCv3` and `info_aeronet_oc_v15_location.csv`.
- Photon path-length model of the gaseous transmittance of the scattered light, option
  `gas_coupling='path_length'` (`--gas_coupling path_length`): per absorption window and term,
  T = (1 + s2 eta tau)^(-1/s2) with tau = -ln Tg(m) of the channel (Gamma distribution of the absorber paths,
  equivalence theorem), applied channel by channel to REPTRAN and line by line to the O2 bands with
  `o2_model='lbl'`, no smoothing needed. Coefficients `data/auxdata/gas_coupling_ppld.csv` fitted by
  `scripts/sos_abs_coupling/ppld.py` on the SOS-ABS simulations (CKD intervals decomposed into exponential
  sums); evaluation `scripts/sos_abs_coupling/ppld_eval.py`. `kernel.path_length_transmittance`,
  `GasCoupling.path_length`, `GaseousTransmittance.get_path_length_transmittance`,
  `O2LineByLine.transmittance_path_length`.
- Coupling of the gaseous absorption with the scattering, option `gas_coupling='effective_air_mass'` of
  `Process` (`--gas_coupling effective_air_mass` on the command line, `settings: gas_coupling` in the
  configuration). In seven windows of the O2-B, O2-A and H2O bands of
  684-990 nm, the transmittances of the path reflectance, of the water-leaving term and of the irradiance
  are corrected with an effective air mass or, for the path reflectance in the O2-A head and the 940 nm
  core, a single-scattering blend, with coefficients depending on the air mass, the aerosol optical
  thickness and the water vapour column (`kernel.GasCoupling`, table `data/auxdata/gas_coupling_eta.csv`,
  `GaseousTransmittance.get_coupled_transmittance`, attributes `Tg_atm`, `Tg_w`, `Tg_Ed` of `Process`).
  The coefficients are fitted on SOS-ABS coupled simulations (CKD 5 cm-1; water vapour 0.5-4 g cm-2, solar
  angles 20-70 deg, AOT(550) 0.05-0.4) and the correction is computed on the transmittance smoothed to
  5 cm-1, then applied channel by channel keeping the interval means. Band TOA reflectance error against the
  coupled simulations: from 4-57 % (decoupled) to 1-6 % in median. Scripts `scripts/sos_abs_coupling`
  (`make_luts.py`: SOS-ABS LUTs with the `sosabs` package; `coupling.py`: analysis and fit) and research
  tutorial *Gaseous transmittance: coupling with scattering*.
- Line-by-line O2 absorption in the O2-B (682.6-701.8 nm) and O2-A (754.7-775.2 nm) bands, option
  `o2_model='lbl'` of `GaseousTransmittance` and `Process` (`--o2_model lbl` on the command line,
  `settings: o2_model` in the configuration; default `'lbl'`, see Changed). The REPTRAN fine table used
  as a spectrum makes these bands too deep at sub-nanometre resolution (equivalent width at air mass 1.155:
  O2-B 1.53 against 0.82 nm, O2-A 4.42 against 3.42 nm). Table `data/lut/gases/o2_lbl_tau.nc` (1.9 MB,
  HITRAN, Voigt profiles, AFGL midlatitude summer) built by `scripts/make_o2_lbl_lut.py`; class
  `kernel.O2LineByLine`.
- `Process.Tg_du`: gaseous transmittance of the Sun-surface-sensor path, computed with the total air mass
  and used for the path reflectance and the water-leaving signal at TOA. It equals `Tg_d * Tg_u` with
  REPTRAN (results unchanged); with the line-by-line O2 it is the cell average of the whole path, since the
  lines are narrower than the cells of the LUT.
- `Spectral.from_srf`: convolution with the tabulated spectral response functions of an instrument
  (DataArray `(band, wl)`, interpolated onto the signal wavelengths; central wavelength = response-weighted
  mean; band names kept as `band` coordinate of the outputs).
- `Spectral.weights` (weight matrix of all the bands, cached per wavelength grid) and `Spectral.responses`.
- Warning when the response of a band is truncated by the wavelength range of the signal.
- Wind speed: optional input variable `wind` (m s-1, per measurement, e.g. ERA5) for the roughness of the
  sea surface; the TOA and transmittance LUTs are interpolated linearly between their wind nodes
  (`lut.wind_weights`, 0.5-16 m s-1, clipped outside), one wind node at a time so that the memory does not
  grow with the spread of the winds. Default `settings: wind: 2.` of the configuration (also for missing
  values); `wind` is written in the output. With the full LUT, the results without wind input are unchanged.
- Spectral extension of Rrs beyond the range of the input (e.g. above 1150 nm), `radcalnet_oc.water.extend_rrs`:
  semi-analytical model with the absorption of pure water (table `data/auxdata/water_coef.txt`, Pope and Fry
  1997, Kou et al. 1993), particulate backscattering fitted on the near-infrared end of the input, joined to
  its last values. Option `rrs_extension` of `Process` (`--rrs_extension`, `settings: rrs_extension`):
  `'pure_water'` (default) or `'zero'`. The fitted backscattering is written in the output (`bbp_nir`).
- Hyperspectral reconstruction of multispectral Rrs (AERONET-OC bands) with the `invRrs` model `two_modes`,
  `radcalnet_oc.reconstruction` (optional dependency, extra `[invrrs]`), 350-1150 nm, at the geometry of the
  measurement and at nadir, with the retrieved parameters and the fit residual; settings of the AERONET-OC
  chain of invRrs (unweighted fit, fixed initial values, 100 evaluations).

### Changed
- Rrs is no longer set to zero beyond the spectral range of the input (`rrs_extension='pure_water'`): Rtoa
  increases in the short-wave infrared for turbid waters (e.g. turbid template: Rrs about 1e-3 sr-1 at
  1240 nm) and is unchanged within the range of the input. The `Rrs` attribute `unit` is `sr-1` (was `-`) and
  its `source` no longer mentions an interpolation by InvRrs.
- `LUT.lut_preparation(wind=...)` interpolates between the wind nodes instead of taking the nearest one; with
  the light LUT of the package (nodes 0.5 and 4 m s-1), the default 2 m s-1 is now interpolated.
- The gaseous transmittance of the scattered light uses the photon path-length model by default
  (`settings: gas_coupling: 'path_length'`): the TOA reflectance, the path reflectance and the irradiance change in
  the O2 and H2O bands of 684-990 nm and are unchanged elsewhere. For the clear-water template (REPTRAN O2,
  10 nm bands, 5 nm in O2-B, OLCI widths in O2-A): band TOA reflectance +7 to +12 % in O2-B, +4 to +63 % in
  O2-A, +17 to +31 % at 822 and 900 nm, x2.1 to x3.0 at 940 nm; irradiance -0.2 to -1.6 %.
  `gas_coupling='decoupled'` gives the previous results.
- The O2 absorption is computed line by line by default (`settings: o2_model: 'lbl'`), closer to the CKD of
  the coupling fit (2 % against 9-18 % more absorbing for REPTRAN). Only the O2-B (682.6-701.8 nm) and O2-A
  (754.7-775.2 nm) bands change; for the clear-water template (with the path-length coupling) the mean TOA
  reflectance increases by +7 to +10 % in 686-696 nm and +14 to +16 % in 759-770 nm (+33 to +39 % in the
  O2-A head, 760-762 nm), the irradiance by +9 to +10 % and +15 to +16 %. Processing about 15 % slower.
  `o2_model='reptran'` gives the previous O2 absorption.
- `Spectral` convolutions use a single matrix product for all the bands, with the weights and prepared
  matrices cached per wavelength grid: 3-4 times faster for repeated convolutions on the same grid
  (e.g. the band variables of `run.py`), same results for `convolve` (Gaussian, < 2e-7).
- The `threshold` of `Spectral.convolve2` is a fraction of the peak response (it was an absolute value
  of the area-normalized response, i.e. a larger fraction of the peak for wider bands: 0.06 % at 5 nm FWHM,
  1 % at 100 nm, 3 % at 300 nm). Band values change by less than 2e-5 for FWHM up to 30 nm, and by up to
  4e-4 (50 nm) and 1e-3 (180 nm) for the wide bands.
- `Spectral.convolve`: a NaN of the signal only gives NaN for the bands whose Gaussian response is not
  null at this wavelength (within about 37 sigma of the band center), as for `convolve2`; a NaN anywhere
  in the signal gave NaN for all the bands before.

### Removed
- `Spectral.convolve_` and `Spectral.convolve2_` (numba per-band loops, not used since the band weights).

### Fixed
- `Sunglint`: the shadowing of the waves (Smith function) used the azimuths of the Sun and of the sensor
  relative to the Sun direction instead of the wind direction, i.e. the slope variances of a wind along the
  solar plane whatever `wazi`. Directional statistics (`cm_dir`, `bh2006`) with `wazi` other than 0 or 180
  deg change; the isotropic statistics (`cm_iso`, used by the Sentinel-2 matchups) and `wazi=0` are unchanged
  (`Sunglint.shadowing`). The two open questions of the module are settled and tested
  (`tests/test_coxmunk.py`): the isotropic `sigma2` is the total mean square slope (sigma2 / 2 per axis), and
  the wind direction turns counterclockwise from the Sun direction, as the viewing azimuth.
- `LUT.toa_lut` failed (`KeyError`) when adding the missing tables of a call to a full cache removed tables of
  the same call read from the cache.

## [1.0.2] - 2026-11-02

### Fixed
- `Process.execute` and `LUT.lut_preparation` failed (`IndexError`) with an aerosol combination varying
  with time, e.g. retrieved for each measurement with `kernel.Aerosol` (regression of 1.0.0, from the
  skipping of the aerosol models of null proportion).

### Added
- Tutorial *From AERONET-OC to top of atmosphere*: the whole chain on one day of the AAOT station
  (aerosol mixture retrieval, TOA simulation for several viewing angles, 10 nm bands, propagation of
  the uncertainty of $R_{rs}$ and of the AOT), with its input file.
- The example notebooks of the `notebook` folder are moved to `docs/source/tutorials/examples` (not
  rendered); the `notebook` folder is no longer tracked.

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

[Unreleased]: https://github.com/Tristanovsk/radcalnet_oc/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Tristanovsk/radcalnet_oc/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/Tristanovsk/radcalnet_oc/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Tristanovsk/radcalnet_oc/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Tristanovsk/radcalnet_oc/releases/tag/v1.0.0
[0.0.1]: https://github.com/Tristanovsk/radcalnet_oc/releases/tag/v0.0.1
