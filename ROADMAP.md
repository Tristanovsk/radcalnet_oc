# Roadmap of radcalnet_oc

Status: November 2026. The tasks are ordered from the easiest to the hardest, in four tiers; within a tier,
in the suggested order. The validation that motivates them is in the report
[`reports/validation_roadmap_2026-11`](reports/validation_roadmap_2026-11) (Sentinel-2 L1C matchups at four
AERONET-OC sites, coupling study against SOS-ABS, computing performance).

**Main finding driving the priorities.** In the visible, the simulated TOA reflectance agrees with MSI within
-3 to 0 % (443-560 nm). Beyond 665 nm it is systematically too low by 2-3e-3, nearly flat with wavelength
(-8 % at 705 nm to -60 % at 2190 nm): the deficit decreases with the glint angle and increases with the
aerosol optical thickness. Explaining it (task 10) has the largest expected impact.

**Suggested order:** 1 → 2 → 3, then 10 with 8 and 9 alongside; the rest of tier 1 when convenient.

## Tier 1: quick wins (hours to a day each)

1. **Release 1.1.** The `[Unreleased]` section of `CHANGELOG.md` is complete (line-by-line O2, path-length
   coupling by default, wind, spectral extension and reconstruction of Rrs, `Spectral.from_srf`,
   `aeronet_oc_db`): ship the 684-990 nm defaults on their own, before the full-range coupling.
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

## Tier 2: moderate (days to a week), mostly diagnostics on existing data

7. **ERA5 wind** in the input preparation of the sites where the station wind is missing (the `wind`
   input and the LUT interpolation are done).
8. **Matchup performance.** Profile one Sentinel-2 scene (about 15 s for ~10 measurements, 2.3 GB peak);
   cache what does not depend on the scene (LUT slices of the site, coupling windows) and simulate all the
   scenes of a site in one vectorised `Process` call over (time, geometry), with the memory capped per LUT
   node; robust parallel runs (worker timeouts, retries).
9. **Matchups at all AERONET-OC sites** with Sentinel-2 subsets (catalog of `aeronet_oc_db`), L1.5 and
   L2.0; per-unit gains (S2A, S2B, S2C) with bootstrap confidence intervals in `matchup_analysis.py`.
10. **Diagnose the NIR-SWIR deficit** on the existing matchups:
    - aerosols: default COAV/MACL 50/50 vs retrieval with DESE/MACL/WASO vs the 6 models of the LUT;
      residuals of the spectral AOT fit; coarse mode in the SWIR;
    - sunglint: Cox-Munk tail vs glint angle and wind, anisotropic statistics with the ERA5 wind
      direction, sky glint of the LUT at the measured wind;
    - screening: box size (600 m half-size vs 150-300 m), SWIR (0.03) and cirrus (0.01) thresholds,
      spatial homogeneity;
    - adjacency: distance to the coast (Section-7, Galata).
11. **Aerosol retrieval by default** when the spectral AOT is available (`kernel.Aerosol` is used only when
    called explicitly), if task 10 confirms the gain.
12. **Refactor `Process.execute`**: separate the physics from the assembly of the output dataset, so that
    the coupled terms can be tested on their own.
13. **Limits of the TOA LUT**: viewing zenith angles stop at 48.9° (below the swath edges of OLCI and VIIRS):
    quantify the extrapolation; check whether the LUT is scalar or vector.

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
16. **Direct sunglint in the package** (today in the matchup script): coupled attenuation, gaseous
    transmittance consistent with the coupling, uncertainty from the wind; glint-based screening threshold
    instead of the fixed 30° glint angle.
17. **Uncertainty budget per band.** Propagate the uncertainties of Rrs, AOT / aerosol model, water vapour,
    O3, pressure and wind, plus the model residuals (coupling per window, REPTRAN vs CKD, invRrs
    reconstruction and SWIR extension of Rrs); Monte Carlo or linearised through `Process`.
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

- Wind speed as an input variable, interpolated between the LUT wind nodes (Unreleased).
- Spectral extension of Rrs beyond the input (pure water) and hyperspectral reconstruction with invRrs
  (Unreleased).
- Photon path-length coupling of the gaseous absorption with the scattering in 684-990 nm, line-by-line O2
  (Unreleased).
- AERONET-OC data from the shared `aeronet_oc_db` database; Sentinel-2 TOA matchups and their analysis
  (`scripts/s2_toa_matchup`).
