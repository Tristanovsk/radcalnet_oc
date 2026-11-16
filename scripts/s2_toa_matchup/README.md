# Sentinel-2 / RadCalNet-OC matchups at the top of atmosphere

Comparison of the Sentinel-2 MSI L1C TOA reflectance with the radcalnet_oc simulation of the AERONET-OC
measurements of a site. Not part of the package.

## Requirements

- radcalnet_oc (development install) with the full TOA LUT, and the optional dependencies `aeronet_oc_db`
  (`pip install radcalnet_oc[aeronet]`, then `aeronet-oc-db update`) and invRrs (`radcalnet_oc[invrrs]`);
  `joblib` for parallel runs.
- Sentinel-2 L1C subsets around the AERONET-OC sites (netCDF, one folder per tile; made with grstbx), default
  `/data/satellite/Sentinel-2/subset/aeronet/L1C` (`--s2-root`).

## Steps

1. **Matchups** (`s2_toa_matchup.py`, one netCDF file per scene cached in `<odir>/scenes`, gathered into
   `<odir>/<site>_<level>_s2_toa_matchup.nc`, default `--odir /data/radcalnet/matchup/sentinel2`):

   ```bash
   python s2_toa_matchup.py --site AAOT --level lev20 --n-jobs 4
   ```

   For each scene with AERONET-OC data within `--window` hours (2): median TOA reflectance of the valid pixels
   of a box of 2 `--half-size` m (600) around the site (B10 < `--cirrus-max` 0.01, B11 < `--swir-max` 0.03);
   Rrs of each measurement reconstructed with invRrs at nadir, aerosol mixture retrieved from the spectral
   AOT, simulation at the geometry of the overpass, band-integrated with the MSI spectral responses; the
   direct sunglint (isotropic Cox-Munk, wind of the station) is added separately (`simg_*`). Already processed
   scenes are skipped (`--overwrite` to redo); failed scenes are retried at the next run. About 15 s per
   scene, 2.3 GB peak memory.

2. **Analysis** (`matchup_analysis.py`):

   ```bash
   python matchup_analysis.py stats                         # per-band differences, per site and S2 unit
   python matchup_analysis.py drivers --sites AAOT_lev20     # correlations, glint-angle and AOT bins
   python matchup_analysis.py figures --fig-dir ../../reports/validation_roadmap_2026-11/figures
   ```

   Selection by default: valid fraction > 0.5, at least 50 pixels, nearest measurement within 60 min,
   AOT(550) < 0.3, glint angle > 30 deg (options `--frac-min`, `--npix-min`, `--dt-max`, `--aot-max`,
   `--glint-min`).

3. **Notebook**: `sentinel2_radcalnet_oc_toa_matchup.ipynb`, spectra and time series of the matchups of a site.

The results of November 2026 (AAOT, Galata, Section-7, Lucinda) are discussed in
`reports/validation_roadmap_2026-11`. The output files record the radcalnet_oc version and settings
(`radcalnet_oc` attribute) since the traceability change of ROADMAP task 2.
