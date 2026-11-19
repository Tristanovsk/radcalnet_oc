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
   `<odir>/<site>_<level>_s2_toa_matchup.nc`, default `--odir /data/radcalnet/matchups/sentinel2/v<radcalnet_oc version>`):

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

   The glint statistics are set with `--glint-stats` (`cm_iso`, or the heavy-tailed `cm_iso_t` and `cm_iso_mix`
   of `radcalnet_oc.Sunglint`, parameters `--glint-nu`, `--glint-f-tail`, `--glint-sigma2-tail`, and the core slope
   variance `--glint-s0 + --glint-s1 W`, Cox-Munk 0.003 + 0.00512 W by default); the output then gets the suffix
   `_<glint-stats>` (`_core` added for a non Cox-Munk core). Retuned set on the platform winds: `--glint-s1 0.0035
   --glint-f-tail 0.10 --glint-sigma2-tail 0.08`. With `--reglint`, the glint of the cached scenes is recomputed with the
   atmospheric factor of the cached simulation (exact, a few minutes for all the sites) without new simulation:

   ```bash
   python s2_toa_matchup.py --site AAOT --level lev20 --reglint --glint-stats cm_iso_mix
   ```

   The gathered files carry the sunglint flags of each scene (`glint_fraction`, glint / simulated TOA reflectance
   with glint at 865 nm, and the bitmask `glint_flags` of `radcalnet_oc.GLINT_FLAGS`: low, moderate, high glint,
   near-specular geometry), the glint uncertainty `glint_unc` (spread of `radcalnet_oc.coxmunk.GLINT_ENSEMBLE` at
   the wind +/- 1.5 m/s, per band) and the spectral flag `glint_uncertain` (per band, `glint_unc / simg_median` >
   5 %).

2. **Analysis** (`matchup_analysis.py`):

   ```bash
   python matchup_analysis.py stats                         # per-band differences, per site and S2 unit
   python matchup_analysis.py drivers --sites AAOT_lev20     # correlations, glint-angle and AOT bins
   python matchup_analysis.py figures --fig-dir ../../reports/validation_roadmap_2026-11/figures
   ```

   Selection by default: valid fraction > 0.5, at least 50 pixels, nearest measurement within 60 min,
   AOT(550) < 0.3, glint angle > 30 deg (options `--frac-min`, `--npix-min`, `--dt-max`, `--aot-max`,
   `--glint-min`).

   With the heavy-tailed glint and the sunglint flags (`--glint-stats cm_iso_mix` reads the `_cm_iso_mix` files;
   `--glint-pixels` uses the statistics without the SWIR threshold and keeps the homogeneous glint scenes;
   `--flags-exclude` replaces the glint-angle screening):

   ```bash
   # matchups per sunglint level and efficiency of the screenings (glint angle, flags), figure s2_glint_levels.pdf
   python matchup_analysis.py glint --sites AAOT_lev20 Galata_Platform_lev20 AAOT_lev15 Section-7_Platform_lev15 \
       --glint-stats cm_iso_mix --glint-pixels --fig-dir figures
   python matchup_analysis.py stats --glint-stats cm_iso_mix --glint-pixels --flags-exclude glint_high glint_near_specular
   ```

   `aot_term_search.py` takes the same `--glint-stats` and `--flags-exclude` options, and `--with-glint` for the
   deficit including the simulated glint.

3. **Notebook**: `sentinel2_radcalnet_oc_toa_matchup.ipynb`, spectra and time series of the matchups of a site.

4. **6S analysis of the NIR-SWIR deficit** (`sixs_terms.py`): the coupling and environment terms of 6S (trapping
   of the water signal, adjacency of the coast with the 6S environment functions and the Natural Earth land
   polygons of cartopy, whitecaps) compared with the deficit, and decomposition of the deficit into an offset and
   a term proportional to AOT (dependence on wind, Angstrom exponent, glint angle and L1C processing baseline):

   ```bash
   python sixs_terms.py --odir /data/radcalnet/matchups/sentinel2/v1.2.0
   ```

5. **Aerosol model diagnosis of the AOT dependence** (`aerosol_diagnosis.py`): path reflectance of the six OPAC
   models of the LUT at 865, 1610 and 2190 nm for each matchup (geometry of the scene, AOT and wind of the nearest
   AERONET-OC measurement), deficit refitted with alternative mixtures, and consistency of each mixture with the
   measured AOT spectrum. The scenes are cached per site in `--cache` (about 12 min for the 511 matchups with
   4 processes, then seconds):

   ```bash
   python aerosol_diagnosis.py --n-jobs 4
   ```

6. **Search for the term correlated with the AOT** (`aot_term_search.py`): signatures of the candidate non-aerosol
   terms (cirrus with the B10 reflectance, representativeness of the AERONET-OC AOT, glint angle and season, wind)
   and comparison of the deficit with the simulated direct glint per glint-angle bin. Reads the cache of
   `aerosol_diagnosis.py` for the retrieved mixture (about 35 s):

   ```bash
   python aot_term_search.py
   ```

7. **Slope variance inverted from the SWIR excess** (`glint_slope_inversion.py`): per scene, the isotropic Cox-Munk
   slope variance whose TOA glint matches the excess sat - sim (raw and cirrus-corrected), compared with the
   AERONET-OC wind; empirical slope pdf per facet tilt, directional statistics (Cox-Munk, Breon-Henriot) averaged
   over the wind directions, and two-component slope distribution fit (about 35 s):

   ```bash
   python glint_slope_inversion.py
   ```

8. **Heavy-tailed slope statistics** (`glint_tail_fit.py`): the `cm_iso_t` (Student t) and `cm_iso_mix`
   (Gaussian + wide slopes) statistics of `radcalnet_oc.Sunglint`, with a scaled core, fitted on the
   cirrus-corrected excess at the station wind over all the glint angles (statistics without the SWIR threshold,
   which removes the most glinted scenes, cloudy scenes rejected by their heterogeneity in the box;
   `--glint-min 30` for the far tail only), and wind inverted per scene with the fitted statistics (about 70 s):

   ```bash
   python glint_tail_fit.py
   python glint_tail_fit.py --core-wind   # wind dependence of the core slope variance s0 + s1 W (about 60 s)
   ```

The results of November 2026 (AAOT, Galata, Section-7, Lucinda) are discussed in
`reports/validation_roadmap_2026-11`. The output files record the radcalnet_oc version and settings
(`radcalnet_oc` attribute) since the traceability change of ROADMAP task 2.

9. **Region of interest at the RadCalNet sites** (`roi_selection.py`, notebook `roi_selection.ipynb`): Sentinel-2 L1C
   window of +/- 1500 m around VEIT (AAOT) and CORO (Section-7 Platform) from all the clear scenes of the tile,
   detection of the fixed structures (frequency of the pixels brighter than the far field at 1610 nm), footprint
   of the structures by distance to them, and ROI candidates (square or disc, half-size 100-1200 m, exclusion
   of a disc or of the structure + buffer) scored by contamination, heterogeneity and dispersion of the matchup
   residual (paired bootstrap). The windows (`<code>_s2_l1c_window.nc`) and the scores are written to
   `/data/radcalnet/studies/roi_selection`:

   ```bash
   python roi_selection.py extract VEIT     # about 6 min, then the same for CORO
   jupyter nbconvert --to notebook --execute --inplace roi_selection.ipynb   # about 3 min (scores cached)
   ```

   Result: the platform footprint vanishes 60-100 m from the structure; the matchups cannot separate the
   candidates; recommended ROI: square of half-size 200 m at VEIT and 300 m at CORO, minus the structure + 100 m.
