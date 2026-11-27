# Roadmap of radcalnet_oc

Status: 22 November 2026. The tasks are ordered from the easiest to the hardest, in four tiers; within a tier,
in the suggested order (task numbers are kept stable; new tasks take the next free number). The validation that motivates them is in the report
[`reports/validation_roadmap_2026-11`](reports/validation_roadmap_2026-11) (Sentinel-2 L1C matchups at four
AERONET-OC sites, coupling study against SOS-ABS, computing performance).

**Main finding driving the priorities.** In the visible, the simulated TOA reflectance agrees with MSI within
-3 to 0 % (443-560 nm). Beyond 665 nm it is systematically too low by 2-3e-3, nearly flat with wavelength
(-8 % at 705 nm to -60 % at 2190 nm): the deficit decreases with the glint angle and increases with the
aerosol optical thickness.

**Update (17 November 2026), after task 10.** Most of the deficit is explained. It is direct sunglint from a
slope distribution with a heavier tail than Cox-Munk: with the heavy-tailed statistics (`cm_iso_mix`) and the
screening by the sunglint flags, the bias at 2190 nm falls from -59 to -24 % (AAOT L2.0). The AOT dependence is
mostly a seasonal confound with the glint angle (high AOT in summer, small glint angles), and thin cirrus
(observed B10) accounts for about 1e-3. The aerosol mixture is not the cause: the mixtures with enough coarse mode
contradict the measured AOT spectrum. Open: a near-infrared term of about -1e-3 at 865 nm; the blue bias
(-3 to -4 % at 443 nm), which does not follow the pressure dependence that the Rayleigh scaling predicts (task 24).

**Cross-package review (17 November 2026)** of the recent work in GRS, hGRS, invRrs and shallowat: tasks 25-29 (the pressure statistics of GRS and hGRS are in task 24).
GRS and hGRS moved their LUT wavelength interpolation to log-log space; the same change in radcalnet_oc moves
the path reflectance by 1.2-1.6 % at 443 nm and 0.3-0.5e-3 at 865 nm, in the direction of the two open residuals
of the matchups (task 25).

**Update (18 November 2026), matchups rerun with the log-log interpolation and the Rayleigh pressure scaling**
(`/data/radcalnet/matchups/sentinel2/v1.2.0` against `v1.1.0`, `scripts/s2_toa_matchup/residual_change.py`, same
scenes, AOT(550) < 0.3, glint angle > 30°). At 443 nm the simulation rises by 0.96-1.42 % (1.2-1.5e-3), of which
the pressure scaling is -0.09 to +0.33 %: AAOT L2.0 -4.21 -> -2.56e-3 (-3.0 -> -1.8 %, N = 314), Galata
-3.81 -> -2.75e-3, Section-7 L1.5 -4.88 -> -3.69e-3. The remaining blue bias now depends on the AOT: -2.5 to
-3.1 % for AOT < 0.08, about 0 % (-0.1 to +0.1 %) for 0.15-0.3 at AAOT and Galata, so it is a clean-atmosphere
term (Rayleigh, Rrs, surface), not aerosol (task 24). At 865 nm the change is only +0.13 to +0.20e-3: the
near-infrared deficit stays at -2.5e-3 (AAOT L2.0) to -7.9e-3 (Section-7) (task 10). Other visible bands move by
about 0.3 point.

**Update (22 November 2026), progress since release 1.2.0 (18 November).**
- *RadCalNet products and site documents*: daily products of an AERONET-OC site (`scripts/radcalnet_products/make_products.py`,
  nadir, 10 nm, invRrs Rrs, retrieved aerosols, direct glint with uncertainty and flags); uncertainty statements in
  LaTeX synchronised with Overleaf (`reports/ESA_official/latex`): VEIT-U revision 3 and CORO-U revision 1 on 1.2.0,
  with the recommended region of interest (`roi_selection.py`) and the sunglint-flag selection of the Sentinel-2
  matchups (`statement_figures.py`). GABU-U and the Q documents are still the converted Word versions (task 30).
- *Uncertainty budget*: model gap of the Sentinel-2 matchups per glint level and sun zenith class (`u_model`,
  task 17); physical residual budget of the matchups (`residual_budget.py`): closure in the visible, B10 floor of
  0.5e-3 = Rayleigh scattering above the water vapour, missing from the simulation (task 14), cirrus 0.2e-3 at the
  median (task 10).
- *New outputs*: direct and diffuse irradiance at the bottom of the atmosphere (`Process.boa_irradiance`, for the
  sky / Sun ratio of AQUA3CS) and the Thuillier (2003) composite solar spectrum of the Sentinel-2 L1C metadata
  (`solar_database='thuillier2003'`) for comparisons in radiance; both unreleased.
- *Full-range coupling* (task 14): end-to-end check on the Sentinel-2 matchups (B10 0.20e-3 with the candidate table
  against 0.50e-3 observed, 0 with the package table); SOS-ABS set v5 of the saturated H2O cores (1380, 1880 nm)
  done, not analysed yet.
- *O2-A* (task 26): with hGRS, the EnMAP 763.7 nm band is about 9 % low over bright land (signal-independent); about
  8 % is left over water with lbl + coupling, which no RT, aerosol, glint or chain term explains.
- *Sentinel-3* (task 31): OLCI Cal/Val monitoring notebook at TOA with 1.2 (VEIT, GABU, CORO); tutorial moved to the
  1.2 sunglint; the full three-site run is in progress.
- *Other packages*: invRrs (unreleased on its develop and feature branches) changed the default sediment n_i shape
  to `'atd_ext:0.5'` (1.4.0 has `'wl_a_sed'`) and added the Bricaud et al. (1998) phytoplankton absorption and a
  mechanistic chlorophyll fluorescence term, both opt-in (task 27); the heavy-tailed glint statistics are now also in
  the `coxmunk` package, and AQUA3CS has its own surface-to-TOA simulation and Sentinel-2 comparison (task 28);
  hGRS 2.1.0 has the solar spectrum as a setting of its L1 drivers (tsis by default, thuillier2003 available); GRS
  keeps the coupling and the pressure scaling on its study branch `feature/abs_scat_coupling` (3.2.0.dev0).

**Suggested order (27 November 2026):** 27 and the Sentinel-3 full run and tutorial rerun of 31 are done.
14 (end-to-end B10 with the v5 cores done, half of the observed floor; then the Rayleigh above the water vapour, also a candidate for
the aerosol-independent part of the NIR excess), 30 (GABU-U); then the next steps of 10 (OSOAA with the AERONET
aerosols of AAOT, inversions at GABU and CORO) and the open points of 31 (S3B spectral responses, VEIT drift); then 9
with 8 alongside, 24 and 26; then 15, 17 and 28.

## Tier 1: quick wins (hours to a day each)

1. ~~**Release 1.1.**~~ *Done* (1.1.0, 15 November 2026, tag `v1.1.0`). Line-by-line O2, path-length
   coupling by default, wind, spectral extension and reconstruction of Rrs, `Spectral.from_srf`,
   `aeronet_oc_db`: the 684-990 nm defaults shipped on their own, before the full-range coupling.
2. ~~**Traceability of the outputs.**~~ *Done.* Package version (and git commit of a development install), `o2_model`,
   `gas_coupling` and coupling table, `rrs_extension`, solar spectrum, default wind, TOA LUT file and DOI as
   attributes of `Process.radcalnet_db` (today only the command line writes the version). The Sentinel-2
   matchup files of November 2026 report `1.0.2` although they were made with the development defaults.
3. ~~**Regression test on reference outputs.**~~ *Done* (`tests/test_regression.py`). Clear-water template, light LUT → band TOA reflectance,
   path reflectance and irradiance compared with a reference file in CI, so that a change of the
   results (e.g. a new default) shows up as a deliberate update of the reference.
4. **Cost of the coupled transmittance.** *Profiled (November 2026), moved to task 8:* 1.3 s (7 windows) and
   2.4 s (19 windows) per `get_gas_transmittance` for 10 measurements and 2 viewing angles. The window
   subsets are already cached; the time is xarray overhead (~2000 small aligned operations per call, 57
   `get_gaseous_transmittance` calls with 19 windows), not arithmetic. Fix: compute all the windows of a term
   at once on the union of their channels (per-channel eta and s2), in numpy. Worth ~10 % of a matchup
   scene: done with task 8.
5. ~~**Open TODOs of `coxmunk.py`**~~ *Done:* the isotropic sigma2 is the total mean square slope, the wind
   direction turns counterclockwise as the viewing azimuth (tests); the wave shadowing now uses the
   azimuths relative to the wind.
6. ~~**Reproducibility of the scripts.**~~ *Done:* README of `scripts/sos_abs_coupling` (`sosabs` dependency, data
   `SOS_ABS_V3` ~0.7 GB, steps simulations → fit → notebook; whether the candidate CSV is tracked) and of
   `scripts/s2_toa_matchup` (L1C subsets, `aeronet_oc_db`, invRrs, outputs, `matchup_analysis.py`).
25. **Wavelength interpolation of the TOA LUT in log-log space** (as GRS 916399f and hGRS 6e290e6). The path
    reflectance, the diffuse transmittances and the spectral AOT are interpolated quadratically in linear space
    between the 10 nodes (`LUT.lut_preparation`, `Process`). Leave-one-out on the LUT: 1-10 % error at 400-800 nm
    and up to 60 % at 1000 nm, against 0.1-1.5 % in log-log. Validate with the OSOAA runs between the nodes that the
    LUT leaves out (0.45, 0.55, 0.65, 0.75, 0.85, 0.9, 2.4 um, `scripts/wl_interpolation`), switch the scheme,
    update the regression reference, and recompute the matchup residuals at 443 and 865 nm (tasks 10, 24).
    *Done for the path reflectance* (branch `feature/wl_interpolation`): median error < 0.1 % for AOT <= 0.1
    (0.4 % at 900 nm, 0.6 % at 2400 nm) against -1 % at 450 and 850 nm and -4 to -9 % at 1150-1400 nm before; TOA
    reflectance of the template +1.3 % (+1.6e-3) at 443 nm, +0.9 % at 865 nm, +7.1 % at 1240 nm (merged in
    `develop`). *Matchups recomputed* (18 November 2026, header): +1.0 to +1.4 % at 443 nm, +0.13 to +0.20e-3 at
    865 nm. Remaining: at AOT 0.35 log-log overestimates by 1-2 % at 1150-1400 nm (interpolate the Rayleigh and
    aerosol parts separately?).
27. **Provenance of the Rrs reconstruction.** `reconstruction.py` uses the default sediment n_i shape of invRrs,
    which changed twice on 16 November 2026 (now `'atd_ext:0.5'`): record the invRrs version and settings in
    `provenance()`, set a minimum version, pin the shape in radcalnet_oc. *Urgent (22 November 2026)*: the
    released invRrs 1.4.0 still has `'wl_a_sed'`; `'atd_ext:0.5'` exists only in its unreleased branches, so the
    reconstructed Rrs (and the 1.2.0 products and matchups) depend on the branch installed. *Done (22 November
    2026)*: shape pinned (`reconstruction.NI_SHAPE`, fallback to the invRrs default with a warning), invRrs version
    and settings in the attributes `invrrs_*` of the reconstruction and of the products, `invRrs>=1.3`. Then: test
    the opt-in chlorophyll fluorescence of invRrs on the reconstruction near 685 nm (AERONET-OC has no band between 667 and
    865 nm; OLCI Oa10/Oa11 and MSI B5 are in the fluorescence peak).
23. ~~**Release 1.2.**~~ *Done* (1.2.0, 18 November 2026, tag `v1.2.0`). The `[Unreleased]` section had changed the default results (Rayleigh pressure scaling on by
    default: about -0.06 % of the TOA reflectance per hPa below 1015.2 hPa in the blue) and added the direct
    sunglint and its flags, the first uncertainty budget and the logging scheme: release them with a tag, so
    that the matchup and GRS/hGRS studies refer to a version, not to a branch.

## Tier 2: moderate (days to a week), mostly diagnostics on existing data

7. **ERA5 wind** in the input preparation of the sites where the station wind is missing (the `wind`
   input and the LUT interpolation are done).
8. **Matchup performance.** Profile one Sentinel-2 scene (about 15 s for ~10 measurements, 2.3 GB peak);
   cache what does not depend on the scene (LUT slices of the site, coupling windows) and simulate all the
   scenes of a site in one vectorised `Process` call over (time, geometry), with the memory capped per LUT
   node; robust parallel runs (worker timeouts, retries).
9. **Matchups at all AERONET-OC sites** with Sentinel-2 subsets (catalog of `aeronet_oc_db`), L1.5 and
   L2.0; per-unit gains (S2A, S2B, S2C) with bootstrap confidence intervals in `matchup_analysis.py`.
10. **Diagnose the NIR-SWIR deficit** on the existing matchups. *Mostly done (November 2026, validation
    report):*
    - ~~aerosols~~: the retrieved mixture is not the cause; the OPAC models consistent with the AERONET-OC AOT
      spectrum give too little SWIR path reflectance per unit AOT, also with a hygroscopic fine mode up to 95 %
      RH (`aerosol_diagnosis.py`); the AOT term is not significant once the glint angle and the cirrus band are
      in the regression (`aot_term_search.py`, `sixs_terms.py`);
    - ~~sunglint tail~~: heavy-tailed isotropic statistics fitted on the SWIR excess (`glint_tail_fit.py`,
      `glint_slope_inversion.py`); the wind term is in the core slope variance (retuned core
      0.003 + 0.0035 W with a tail fraction 0.10);
    - open: the remaining near-infrared term (about -1e-3 at 865 nm; the log-log interpolation adds only
      +0.13 to +0.20e-3, residual -2.5e-3 at AAOT L2.0, -1.8e-3 for AOT < 0.08, task 25); anisotropic statistics with the ERA5
      wind direction; refit of the glint statistics with the ERA5 wind instead of the platform wind (task 7)
      before changing the defaults of `Sunglint`; thin cirrus in the simulation or as a screening term;
    - open: screening (box size 600 m half-size vs 150-300 m, SWIR and cirrus thresholds, homogeneity) and
      adjacency (distance to the coast, Section-7, Galata).
    - *Regression slope below 1 (22 November 2026, `scripts/s2_toa_matchup/slope_analysis.py`)*: the slope of the
      simulation against MSI is 0.65-0.71 at 865 nm and 0.27-0.52 at 1610-2190 nm (VEIT, CORO, ROI, sunglint-flag
      selections). With MSI on the x axis, 1 - slope = cov(MSI - sim, MSI) / var(MSI): a term missing from the
      simulation that grows with the signal; the noise of the ROI median (regression dilution) is < 0.001. The
      cirrus predicted from B10 is fully missing (regression coefficient 1.0-1.5) and explains 0.12-0.28 of the
      deficit (VEIT), the glint is not robust (sign changes with the selection). At 865 nm the deficit closes
      without cirrus and with |dt| < 15 min (slope 0.97, VEIT, 81 matchups): cirrus and the time mismatch. In the
      SWIR it does not (0.48 at 1610 nm, 0.42 at 2190 nm) and it grows with the AOT (2190 nm: 0.39 for AOT < 0.1,
      0.25 above 0.15). *Coarse mode ruled out (23 November 2026)*: the retrieved mixture is 97 % WASO (median);
      with the per-model path reflectances of `aerosol_diagnosis.py`, a maritime mixture brings the slope to 0.99 at
      1610 nm but with a +53 % bias, and the residual fitted on (MACL - retrieved) asks for only 8-18 % of MACL and
      moves the slope from 0.48 to 0.52. *Glint variability*: for the scenes without cirrus and |dt| < 15 min the
      slope is about 1 beyond 55 deg of glint angle in all the bands (865: 1.18, 1610: 1.20, 2190: 0.98; r 0.7-0.9)
      and collapses below 45 deg (0.15-0.34 in the SWIR) with a small mean residual: the simulated glint is right on
      average but does not follow the scene-to-scene variation of the real glint (station wind, mean slope
      statistics). A slope below 1 is then the share of the MSI variance that the simulation reproduces
      (x = s + m, b = var(s) / var(x)), not a term proportional to the signal. Next: compare this variance with
      u_glint (is the glint uncertainty large enough at 35-55 deg?), test the ERA5 or CAMS wind and the wind
      variability within the hour (tasks 7, 29), and repeat per glint-angle bin on the Sentinel-3 OLCI matchups
      (Oa17-Oa21, more scenes at CORO); in the statements, show the regression per glint-angle class.
    - *Near-infrared term, OLCI and MSI together (27 November 2026, `scripts/aerosol_phase`)*: the same additive
      excess is seen by both sensors at 865 nm: +1.6 to +2.4e-3 for OLCI (three sites, S3A and S3B; slope 1.02,
      intercept 1.9e-3) and +1.6 to +1.9e-3 for MSI (VEIT, CORO, ROI, heavy-tailed glint). It is nearly flat from 709
      to 885 nm. The two sensors are calibrated independently, so it is a model or input term. It comes from below or
      inside the water vapour: in-band / continuum, it is attenuated like the simulated signal (OLCI 940 nm 0.52
      against 0.45, 900 nm 0.82 against 0.78; MSI B9 0.56 against 0.38). This rules out a sensor offset, stray light
      or high cirrus as the main term. In OLCI it is not glint: it is largest beyond 70 deg of glint angle and for the
      scenes without glint flag. It grows with the AOT and the viewing zenith angle, decreases with the sun zenith
      angle (about 0 in winter, 2.7e-3 in summer), and has a backscatter signature: dR = c0 + dP(theta)
      tau865 / (4 mu_s mu_v) gives the same dP curve for both sensors, from about -0.04 at 90-130 deg to +0.03 at
      150-170 deg, also at a fixed sun zenith angle. *AERONET almucantar phase functions at AAOT* (4670 V3
      retrievals 2018-2026, Level 1.5 with the Level 2 flag; mixture refitted per retrieval, 95 % WASO_rh0): omega*P
      at 865 nm is 12-22 % above the OPAC mixture, +0.03 at 90-150 deg and +0.05 at 160-170 deg. No OPAC component
      closes it: humid WASO and the coarse models have less backscatter than WASO_rh0. The single-scattering
      prediction with the same-day difference explains about half of the VEIT excess: OLCI 1.3 of 2.4e-3 (slope
      0.59 +/- 0.08), MSI 0.9 of 2.3e-3 (slope 0.80 +/- 0.19). A control with the same tau / (mu_s mu_v) scaling but
      no angular shape has no skill (R2 0.003 against 0.05-0.075). About 1e-3 remains where the prediction is zero,
      partly above the water vapour (missing Rayleigh above the vapour, cirrus). Caveats: the almucantar measures up
      to 2 sza only (median 124 deg at AAOT), so the backscatter part is the extrapolation of the retrieval (over the
      measured range the difference falls to +0.02 at 90-120 deg and about 0 at 150-160 deg); the AERONET single
      scattering albedo is unreliable at low AOD (about 2 of the 13 %). Not related to the HYPERNETS irradiance deficit
      of aqua3cs (5-9 % below the simulation, multiplicative, growing with the sun zenith angle, while the direct beam
      of radcalnet_oc matches Beer-Lambert with the AERONET AOT within 0.15 % at 865 nm). Next: OSOAA runs of a few
      VEIT matchups with the AERONET size distribution and refractive index instead of OPAC; the principal-plane or
      hybrid scans for the backscatter; inversions at GABU and CORO; an aerosol model with AERONET-like backscatter
      (narrower fine mode) in the LUT, if confirmed.
11. **Aerosol retrieval by default** when the spectral AOT is available (`kernel.Aerosol` is used only when
    called explicitly). *Deprioritised:* task 10 found no gain from the mixture in the NIR-SWIR; keep it for the
    consistency with the AOT spectrum, and revisit with the aerosol term of the uncertainty budget (task 17).
12. **Refactor `Process.execute`**: separate the physics from the assembly of the output dataset, so that
    the coupled terms can be tested on their own.
13. **Limits of the TOA LUT**: viewing zenith angles stop at 48.9° (below the swath edges of OLCI and VIIRS):
    quantify the extrapolation; check whether the LUT is scalar or vector.
26. **EnMAP and PRISMA radiometric closure** with the current radcalnet_oc: rerun the closure of hGRS
    (`hgrs/validation/aeronet_oc/closure.py`, EnMAP L1C against radcalnet_oc at the AERONET-OC stations, written
    with linear Rrs interpolation, before the pressure scaling and the glint) with version 1.2, the invRrs
    reconstruction and the direct glint; per-band gains with uncertainties; extend to PRISMA. It tells whether the
    VNIR/SWIR offset that hGRS corrects empirically (29c54ea), still seen in the red by the shallowat ACTIUM study,
    is a sensor effect.
    *O2-A conflict (21 November 2026), open.* The direct transmittance (SOS-ABS CKD, PRISMA glint scene) favours
    line-by-line O2, while the EnMAP closure on dark water (diffuse-path dominated) has lbl + path-length coupling
    +0.109 ± 0.023 too shallow in the 763.7 nm band (positive in 60/60 matchups, 9 sites, no smile or column
    dependence) and REPTRAN + coupling within 3 %. Ruled out: the aerosol scale height and model (ha1/ha4/ARCT
    variants, f_path changes ≤ 0.025 against the -0.07 needed), polarisation (-0.005), the radcalnet_oc chain and
    the way hGRS applies f_path (closure numbers come from `Process`), the EnMAP SRF position and width (±1 nm moves
    the error by ≤ 0.03). *Line-resolved SOS-ABS reference, done*
    (`scripts/sos_abs_coupling/o2a_lbl_reference.py`, output in `/data/sos_abs_coupling/v4_o2a/lbl_ref`):
    monochromatic runs through single-term CKD files (HITRAN 2012 Voigt, wings cut at 50 half-widths; 4 stratified
    sample sets of one wavenumber per 1 cm-1 interval, 259 wavelengths, v4 nominal settings). The ppld table
    applied per line under-predicts the line-resolved f_path by 0.038 (median, EnMAP band) and the CKD 1 cm-1
    f_path by 0.030: the CKD of SOS-ABS holds for the scattered light, and the real coupling is slightly stronger,
    the opposite sign to the EnMAP gap. Absorption in the RT is ruled out. Left: the EnMAP O2-A SRF shape or
    out-of-band response, the simulation inputs (surface pressure, water vapour term in the band), spherical
    geometry (small). *Low aerosol layer, done* (`scripts/sos_abs_coupling/o2a_variants_fpath.py`, variant ha05):
    scale height 0.5 km lowers the EnMAP f_path by only 0.009 (AOT 0.05) to 0.019 (AOT 0.2), from 1.21/1.14 to
    1.20/1.12, against a drop to about 1.0 needed. The hGRS gain analysis (error proportional to the signal,
    dR = 0.23 R; 756 and 771 nm bands consistent with every model; no air-mass, glint or AOT dependence) points to
    the EnMAP 763.7 nm band reading about 16 % low; to confirm with a closure over a bright land RadCalNet site,
    where the coupling hardly matters. *Land test, done (hGRS, 21 November 2026)*: 21 EnMAP L1C scenes over bright
    bare land (Bahia Blanca, Lampedusa, Senegal, Actium; R778 > 0.15, NDVI < 0.12), band depth against the linear
    continuum 749/778 divided by the two-way direct lbl O2 transmittance at the scene air mass. The 756 and 771 nm
    bands agree within ±2.5 %; the 763.7 nm band reads 0.917 (0.873-1.009, processor V010502, 14 scenes, the version
    of all the water matchups) and 0.947 (V010506, 7 scenes), with no significant air-mass slope. The coupling over
    a bright surface hardly changes the prediction: SOS v4 nominal, f of the total TOA signal 1.004-1.010 for
    albedo 0.2-0.3 (surface share 88-96 % of the band signal), against 1.13-1.21 at albedo 0. The EnMAP band
    deficit is therefore about 9 % (V010502), half of the water gap. With the water matchups corrected by the land
    deficit (per air mass), EnMAP / simulation is 0.92 for lbl + path-length coupling, 1.03 for lbl decoupled and
    1.05 for REPTRAN + coupling: the water data behave as f_path ≈ 1.03-1.05 against 1.12-1.20 from SOS, and the
    earlier agreement of REPTRAN + coupling came from the band deficit compensating its over-absorption. No
    surface-like term (water-leaving, glint residual, adjacency, f ≈ 1.0) can explain the remaining 8 %: it would
    need a share of about 0.8 of the band signal over dark water. *Brightness bins, done (hGRS)*: bare pixels of
    the 21 scenes binned by R778 from 0.06 to 0.41: the uncorrected 763.7 nm ratio is flat (0.89-0.93 over a factor
    of 7 in signal; within-scene slope of ln g against ln R778 +0.010, interquartile -0.03/+0.05), 756 and 771 nm
    within 1.6 % in every bin. Non-linearity or out-of-band response of the band are not supported, although the
    water matchups (R778 about 0.03) lie a factor of 2 below the darkest land bin. Status: band deficit about 9 %
    (V010502), independent of the signal level; about 8 % left over water with lbl + coupling (apparent f_path
    1.03-1.05 against 1.12-1.20 from SOS-ABS), unexplained (write-up in hGRS,
    docs/source/tutorials/research/o2a_enmap_closure.rst). This decides the O2-A default (lbl or REPTRAN) in radcalnet_oc and hGRS; neither is
    supported by the corrected water data yet.
24. **Blue bias and pressure.** With the Rayleigh pressure scaling (done, validated with OSOAA), the residual at
    443 nm should decrease by about 0.64 % per % of pressure, but the matchups give -0.11 (t = -0.6), about three
    standard errors away: another blue term correlated with the pressure is likely present (Rrs used at nadir,
    task 15; clear and calm days at high pressure). Investigate before using the blue bias of the matchups.
    Close the real-scene check of the scaling in GRS (223 Sentinel-2 images) and hGRS (103 EnMAP matchups)
    against AERONET-OC (`scripts/rayleigh_pressure`; runs in progress).
    *Rerun with the pressure scaling and the log-log interpolation* (18 November 2026, header): the median residual at
    443 nm goes from -3.0 to -1.8 % (AAOT L2.0); the pressure part is small (-0.09 to +0.33 % of the TOA
    reflectance). The bias is now concentrated at low AOT: -2.8 % (AAOT L2.0) and -2.6 % (Galata) for AOT < 0.08,
    -0.1 and +0.1 % for 0.15-0.3. Next: regress the new residual on the pressure, AOT and Rrs(443) to separate a
    Rayleigh / surface term from an aerosol compensation (the aerosol retrieval or the fine-mode share may hide
    the deficit at higher AOT).

28. **Shared physics across radcalnet_oc, GRS and hGRS.** The path-length coupling, the Rayleigh pressure
    scaling and the log-log interpolation are implemented three times: a set of reference values (inputs and
    expected outputs) tested in the CI of each package, so that they do not drift apart (e.g. the reference
    pressure 1013.2 hPa of the linear scaling of GRS against 1015.2 hPa of the LUT runs). Since 21 November 2026
    the heavy-tailed glint statistics (`cm_iso_t`, `cm_iso_mix`) are also in the `coxmunk` package, and AQUA3CS
    has its own surface-to-TOA simulation: include the glint statistics in the shared references, and make AQUA3CS
    call radcalnet_oc (or check it against it) rather than keep a third TOA model.
29. **One wind source for the glint.** Refit the heavy-tailed glint statistics with the ERA5 or CAMS wind (GRS
    uses CAMS) before changing the defaults of `Sunglint` (with task 7); then test the glint-fraction flags of
    radcalnet_oc as the glint screening of GRS (hook: its new `p_slope` switch).

30. **RadCalNet site documents.** VEIT-U (revision 3) and CORO-U (revision 1) are on 1.2.0; bring GABU-U to the
    same state (products, budget with the model gap, Sentinel-2 matchups in a region of interest; Galata has no
    fixed-structure detection yet), revise the Q documents, link CORO-U to Overleaf, and update the statements when
    the products change (tasks 14, 27). *22 November 2026*: CORO-U and CORO-Q linked to Overleaf (CORO-Q first
    draft from GABU-Q, AERONET-OC climatology by `site_climatology.py`); Sentinel-2 overview and zoom in the Q
    documents (`site_rgb.py`, GABU added, no ROI at Galata yet). *Data issue*: at Section-7 the AERONET-OC wind
    is one constant value per month from April to August 2026 (level 1.5; also runs of constant wind and pressure in
    March 2021, level 2.0): fill or climatology values, which feed the sunglint and its uncertainty in the CORO
    products of these months; flag or replace them (ERA5 wind, task 7) before the next CORO products.
31. **Sentinel-3 OLCI Cal/Val monitoring** (`notebook/sentinel3/sentinel3_radcalnet_oc_toa_calval_monitoring.ipynb`):
    ratio OLCI / simulation per band, site and platform at VEIT, GABU and CORO with the 1.2 model, the uncertainty
    budget and the glint flags. Quick run (VEIT, summer 2020): OLCI 1-2.5 % above the simulation in the blue,
    within u_sim; S3A - S3B +1.2 to +1.9 % (not significant); O2-A (Oa13) and Oa14 still too shallow.
    *Full run done (22 November 2026)*: 2027 matchups from April 2016 to August 2026 (VEIT 830, CORO 624, GABU 573),
    1735 without high or near-specular glint (`/data/radcalnet/matchups/sentinel3/v1.2.0`). Statement figures and
    tables (`scripts/radcalnet_products/s3_statement_figures.py`) are in VEIT-U and CORO-U (media v1.3). Tutorial
    rerun with 1.2.0 (27 November 2026). Findings:
    - *Visible*: OLCI is above the simulation by +1 to +3 % at 400-560 nm, within u_sim (1.1-4.2 %).
    - *S3A - S3B*: significant over the archive, +1.2 to +2.5 % in the visible (90 % bootstrap interval excludes 0;
      Oa03 +2.2 % at VEIT, +2.1 % at CORO), smaller and not significant beyond 750 nm. This is a sensor term, and it
      is consistent between the sites.
    - *Near infrared*: an additive excess of +1.6 to +2.4e-3 at 865 nm (ratio +10 to +13 % at Oa17, +26 to +33 % at
      Oa21), the same as MSI. It grows with the viewing zenith angle, the AOT and toward backscattering, and is not
      glint (task 10, `scripts/aerosol_phase`). Oa21 has a larger excess than Oa17 (3.0-3.8e-3), probably specific
      to OLCI.
    - *Absorption bands*: Oa13 is 5-8 % below the simulation (OLCI deeper than simulated), beyond the NIR excess
      (task 26). Oa19 and Oa20 are absorbed like the surface signal.
    - *Drift* (Theil-Sen, 90 % interval): none at CORO and GABU (|drift| < 0.6 %/yr, intervals include 0). At VEIT,
      S3A drifts by +0.1 %/yr at 443 nm, +0.3 at 665 nm, +0.7 at 865 nm and +1.4 at 1020 nm (all significant), and
      S3B by +0.15 %/yr at 443 nm. Seen at one site only and growing with the wavelength, so likely a change at VEIT
      (inputs, NIR term) rather than a sensor drift: to be checked.
    Next: the S3B spectral responses (the S3A response shifted to the detector lambda0 is used for both
    platforms), an OLCI model gap (task 17; u_model is derived at VZA < 12 deg and the NIR excess grows with the
    VZA), the VEIT drift, and the per-band gains with the uncertainty budget.

## Tier 3: substantial (weeks)

14. **Path-length coupling beyond 990 nm.** Adopt the windows of the candidate table
    (`scripts/sos_abs_coupling/gas_coupling_ppld_full.csv`) where the gain is large and survives
    cross-validation: H2O 1000, 1130, 1380, 1880, 2150 and 2450 nm, O2 1270 nm, CO2 2010 nm; keep O3-VIS and
    CO2 1600 nm decoupled (eta close to 1). CH4: the package couples only H2O and the background gases
    (`GaseousTransmittance.COUPLED_GASES`) while the CH4 1660 and 2300 nm coefficients were fitted on the
    total transmittance: add CH4 to the coupled gases or drop these windows. Validate end to end before
    changing the default: band path reflectance against the coupled SOS-ABS simulations (first H2O 1380 and
    1880 nm, where the absorption tables differ by 30-40 %), then against MSI B10 (simulated ~1e-7 vs
    8e-4 observed). *End-to-end check done* (`scripts/sos_abs_coupling/check_fullrange_s2.py`, 20 November 2026):
    B10 0.20e-3 with the candidate table against 0.50e-3 observed (correlation 0.73), other bands unchanged; the
    1380 nm model, fitted on the wings, recovers 40 % of the core. SOS-ABS set v5 of the saturated cores
    (`/data/sos_abs_coupling/v5`, 88 CKD intervals of 1344-1471 and 1801-1964 nm) done: refit the 1380 and 1880 nm
    windows on v3 + v5. The residual budget assigns the B10 floor of 0.5e-3 to Rayleigh scattering above the water
    vapour (55 +/- 8 % of the molecules in single scattering), which the coupled transmittance of the whole path
    removes: add it as a term above the absorber before changing the default. *Cores refitted (22 November 2026,
    `fullrange_v5.py`, table `gas_coupling_ppld_full_v5.csv`)*: windows H2O-1380 and H2O-1880 fitted on v3 + v5
    (154 + 88 channels): bias of T_atm in the cores -0.034 -> -0.011 (1380) and -0.025 -> -0.005 (1880), wings
    unchanged; MSI B10 dark-water path reflectance 0.92 of SOS-ABS (0.85 with the v3 fit, 0.15 decoupled), e.g.
    1.16e-3 (SOS) against 0.92e-3 at AOT 0.05 and 0.5 g cm-2. Next: rerun `check_fullrange_s2.py` with this table
    (the end-to-end B10 was 0.20e-3 with the v3 fit, against 0.50e-3 observed: the gap is larger than the 15 % of
    the fit, so look at the band integration and the H2O table of the package in the cores).
    *End-to-end check with the v5 table (27 November 2026, `check_fullrange_s2.py --table
    gas_coupling_ppld_full_v5.csv`, same 59 VEIT L2.0 matchups as the v3 run; outputs in
    `/data/radcalnet/studies/fullrange_check`)*: B10 without cirrus (N 30) is 0.26e-3 with v5 against 0.20e-3 with
    v3 (median, +29 %), 0.50e-3 observed and 0.55e-3 for the single-scattering Rayleigh above the water vapour; the
    correlation with the L1C stays 0.74. The v5 simulation reaches 0.51 of the observation (0.60 below 15 kg m-2 of
    water vapour, 0.49 above), against 0.92 of SOS-ABS in the table check of `fullrange_v5.py` (dark water, AOT 0.05,
    0.5 g cm-2). The other bands do not change (B11 by 1e-6). The refit of the cores is not the limiting term: the
    end-to-end B10 is still half of the observation, so the gap is in the chain between the table and the band (band
    integration of the saturated core, H2O absorption table of the package against SOS-ABS, wetter atmospheres than
    the table check) or in the missing Rayleigh above the water vapour itself. Next: B10 of the package against
    SOS-ABS at the matchup water vapour (2-3 g cm-2), then the Rayleigh-above term.
15. **Bidirectional Rrs.** Adapt the normalised AERONET-OC Rrs to the viewing and solar geometry of the
    satellite (f/Q tables, or the invRrs reconstruction at the satellite geometry instead of nadir); possibly
    part of the -3 to -4 % at 443 nm. Use the per-band viewing angles of MSI.
16. ~~**Direct sunglint in the package**~~ *Done* (`Process.compute_sunglint`, `Process.execute(sunglint=...)`):
    TOA direct glint with the gaseous and direct transmittances of the Sun-surface-sensor path, glint angle,
    bitmask of glint levels and near-specular geometry (`GLINT_FLAGS`) replacing the fixed 30° glint angle
    (Youden index 53 vs 51 on the 715 matchups), uncertainty from an ensemble of slope statistics and wind
    +/- 1.5 m s-1, spectral flag `glint_uncertain`. Remaining: gaseous transmittance of the glint path consistent
    with the coupling; anisotropic statistics (task 10).
17. **Uncertainty budget per band.** *First version done* (`radcalnet_oc.uncertainty.uncertainty_budget`: AOT,
    water vapour, O3, pressure, wind and Rrs propagated through `Process` by central differences, glint term,
    quadrature; provisional input uncertainties). Remaining: the input uncertainties of the RadCalNet assessment;
    the aerosol model term (spread of the mixtures compatible with the AOT spectrum); the model residuals (coupling
    per window, REPTRAN vs CKD, invRrs reconstruction and SWIR extension of Rrs); the terms found on the matchups
    (thin cirrus, near-infrared term); per-band output and correlations between the terms; cost (two simulations
    per input). *Model gap done* (`u_model`, 20 November 2026, unreleased): the part of the Sentinel-2 residual not
    explained by the physical budget, per glint level, sun zenith class and band; derived at VZA < 12 deg, so
    extrapolated for OLCI (up to 48 deg): an OLCI-specific gap from task 31, not fitted on the matchups it assesses.
18. **HYPERNETS** ([waterhypernet.org/data](https://waterhypernet.org/data/)): reader of the L2 water products
    (hyperspectral Rrs, uncertainties, flags) into the input format; validation of the invRrs reconstruction
    and of the SWIR extension at the sites co-located with AERONET-OC; direct hyperspectral input with its
    uncertainties.

## Tier 4: research and long term

19. **H2O absorption table.** REPTRAN absorbs one third less than the CKD of SOS-ABS at 1380 nm and 25-40 %
    more at 720, 900, 1880 and 2150 nm, more than the coupling residual: line-by-line H2O in the strong bands,
    following `scripts/make_o2_lbl_lut.py` for O2.
20. **Spectral sampling of the TOA LUT**: 10 wavelength nodes only (1000, 1600, 2200 nm in the SWIR);
    quantify the interpolation error of the path reflectance with dense radiative transfer simulations and add
    nodes if needed (new LUT, new Zenodo record).
21. **Wider coupling training set**: continental / urban aerosols, other H2O profiles and surface pressures,
    vector radiative transfer; gaps of 684-990 nm (698-712, 736-757, 772-808, 836-890 nm); 18-20 % 95th
    percentiles of the band residual in the saturated bands.
22. **Operational chain and other sensors**: scheduled update of all the sites from `aeronet_oc_db`, quality
    flags, RadCalNet-like hyperspectral TOA product per site and day; Sentinel-3/OLCI, Landsat-8/9,
    PACE/OCI and hyperspectral missions (EnMAP, PRISMA, FLEX). *Started*: daily products of a site
    (`make_products.py`), Sentinel-3 OLCI monitoring (task 31).

## Done

Unreleased (after 1.2.0):

- Model gap in the uncertainty budget (`u_model`, task 17).
- Direct and diffuse irradiance at the bottom of the atmosphere (`Process.boa_irradiance`).
- Thuillier (2003) composite solar spectrum (`solar_database='thuillier2003'`).
- Products and site documents (task 30), Sentinel-3 monitoring notebook (task 31), O2-A studies (task 26).

Version 1.2.0 (18 November 2026, task 23):

- Pressure scaling of the Rayleigh scattering, on by default (`rayleigh_pressure`), validated with OSOAA at
  +/- 20 hPa (within 0.03 % of the path reflectance for a Rayleigh atmosphere, 0.25 % with AOT 0.1); also
  implemented in GRS and hGRS.
- Direct sunglint, heavy-tailed slope statistics (`cm_iso_t`, `cm_iso_mix`, core coefficient `sigma2_coef`),
  sunglint flags and glint uncertainty (task 16).
- First uncertainty budget (task 17).
- Logging scheme: one logger per module under `radcalnet_oc`, `set_logging`, `--log_level` and `--log_file`.
- Tutorials rerun for version 1.1; first tutorial *Simulate the TOA signal from user settings*.

Version 1.1.0:

- Wind speed as an input variable, interpolated between the LUT wind nodes (1.1.0).
- Spectral extension of Rrs beyond the input (pure water) and hyperspectral reconstruction with invRrs
  (1.1.0).
- Photon path-length coupling of the gaseous absorption with the scattering in 684-990 nm, line-by-line O2
  (1.1.0).
- AERONET-OC data from the shared `aeronet_oc_db` database; Sentinel-2 TOA matchups and their analysis
  (`scripts/s2_toa_matchup`).
