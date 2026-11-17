# Roadmap of radcalnet_oc

Status: 17 November 2026. The tasks are ordered from the easiest to the hardest, in four tiers; within a tier,
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

**Suggested order:** 23 (release 1.2), then 9 with 8 alongside, the open points of 10 and 24; then 15 and 17.

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
23. **Release 1.2.** The `[Unreleased]` section has changed the default results (Rayleigh pressure scaling on by
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
    - open: the remaining near-infrared term (about -1e-3 at 865 nm); anisotropic statistics with the ERA5
      wind direction; refit of the glint statistics with the ERA5 wind instead of the platform wind (task 7)
      before changing the defaults of `Sunglint`; thin cirrus in the simulation or as a screening term;
    - open: screening (box size 600 m half-size vs 150-300 m, SWIR and cirrus thresholds, homogeneity) and
      adjacency (distance to the coast, Section-7, Galata).
11. **Aerosol retrieval by default** when the spectral AOT is available (`kernel.Aerosol` is used only when
    called explicitly). *Deprioritised:* task 10 found no gain from the mixture in the NIR-SWIR; keep it for the
    consistency with the AOT spectrum, and revisit with the aerosol term of the uncertainty budget (task 17).
12. **Refactor `Process.execute`**: separate the physics from the assembly of the output dataset, so that
    the coupled terms can be tested on their own.
13. **Limits of the TOA LUT**: viewing zenith angles stop at 48.9° (below the swath edges of OLCI and VIIRS):
    quantify the extrapolation; check whether the LUT is scalar or vector.
24. **Blue bias and pressure.** With the Rayleigh pressure scaling (done, validated with OSOAA), the residual at
    443 nm should decrease by about 0.64 % per % of pressure, but the matchups give -0.11 (t = -0.6), about three
    standard errors away: another blue term correlated with the pressure is likely present (Rrs used at nadir,
    task 15; clear and calm days at high pressure). Investigate before using the blue bias of the matchups.
    Close the real-scene check of the scaling in GRS (223 Sentinel-2 images) and hGRS (103 EnMAP matchups)
    against AERONET-OC (`scripts/rayleigh_pressure`; runs in progress).

## Tier 3: substantial (weeks)

14. **Path-length coupling beyond 990 nm.** Adopt the windows of the candidate table
    (`scripts/sos_abs_coupling/gas_coupling_ppld_full.csv`) where the gain is large and survives
    cross-validation: H2O 1000, 1130, 1380, 1880, 2150 and 2450 nm, O2 1270 nm, CO2 2010 nm; keep O3-VIS and
    CO2 1600 nm decoupled (eta close to 1). CH4: the package couples only H2O and the background gases
    (`GaseousTransmittance.COUPLED_GASES`) while the CH4 1660 and 2300 nm coefficients were fitted on the
    total transmittance: add CH4 to the coupled gases or drop these windows. Validate end to end before
    changing the default: band path reflectance against the coupled SOS-ABS simulations (first H2O 1380 and
    1880 nm, where the absorption tables differ by 30-40 %), then against MSI B10 (simulated ~1e-7 vs
    8e-4 observed).
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
    per input).
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
    PACE/OCI and hyperspectral missions (EnMAP, PRISMA, FLEX).

## Done

Unreleased (to go into 1.2, task 23):

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
