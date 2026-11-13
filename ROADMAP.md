# Roadmap of radcalnet_oc

Status: November 2026. Items are grouped by theme and listed in priority order within each group; the
suggested overall order is at the end. Items marked *to check* are not confirmed in the code.

## 1. Gas coupling: complete the 350-2500 nm extension (short term)

Source: `notebook/research/radcalnet_oc_gaseous_coupling_350_2500nm.ipynb` (built by
`scripts/sos_abs_coupling/build_fullrange_notebook.py`, conclusions in
`scripts/sos_abs_coupling/conclusions_fullrange.md`).

1. **Extend the photon path-length model beyond 990 nm.** Adopt the windows of the candidate table
   (`scripts/sos_abs_coupling/gas_coupling_ppld_full.csv`) where the gain is large and survives
   cross-validation: H2O 1000, 1130, 1380, 1880, 2150 and 2450 nm, O2 1270 nm, CO2 2010 nm. Keep the O3-VIS
   and CO2 1600 nm windows decoupled (eta close to 1, the model does not improve the path reflectance).
2. **CH4.** The package couples only H2O and the background gases (`GaseousTransmittance.COUPLED_GASES`),
   while the CH4 1660 and 2300 nm coefficients were fitted on the total transmittance: add CH4 to the
   coupled gases or drop these windows.
3. **End-to-end validation before changing the default.** Band-level path reflectance of the package
   against the coupled SOS-ABS simulations, as done for 684-990 nm, first in the H2O 1380 and 1880 nm
   windows, where the absorption tables differ by 30-40 %.
4. **H2O absorption table.** REPTRAN absorbs one third less than the CKD of SOS-ABS at 1380 nm and 25-40 %
   more at 720, 900, 1880 and 2150 nm, more than the coupling residual. A line-by-line H2O table in the
   strong bands, following `scripts/make_o2_lbl_lut.py` / `o2_lbl_tau.nc` for O2.
5. **Gaps of 684-990 nm.** 698-712, 736-757, 772-808 and 836-890 nm keep the decoupled model (weak but
   non-zero H2O absorption).
6. **Wider simulation set.** The coefficients are fitted for maritime aerosols, US62 profiles, 1013 hPa and
   scalar radiative transfer: add continental / urban aerosols, other H2O profiles and surface pressures,
   and check whether the 18-20 % 95th percentiles of the band residual in the saturated bands are a limit
   of the model or of the sampling.

## 2. Physics and inputs (medium term)

7. **Wind speed.** `LUT.lut_preparation(wind=2, ...)` selects the nearest wind node and `Process` never
   passes the wind: the sky and sun reflection is always computed for 2 m/s. Read the wind from the input
   (ERA5 / CAMS) and interpolate between the nodes of the LUT.
8. **Spectral extension of Rrs.** `Process.execute` interpolates Rrs linearly between the AERONET-OC bands
   and sets it to zero outside them (`Rrs.interp(wl=full_wl).fillna(0)`); the output attribute still
   mentions an "interpolation based on InvRrs". Needed: a hyperspectral reconstruction (narrow sensor bands
   between the AERONET-OC wavelengths) and an NIR-SWIR model for turbid sites.
9. **Angular dependence of the water-leaving radiance** *(to check)*: adaptation of the normalized Rrs of
   AERONET-OC to the viewing geometry of the satellite (f/Q, BRDF); no such correction found in
   `process.py`.
10. **Aerosol model.** The default is a fixed COAV / MACL 50/50 combination; `kernel.Aerosol` retrieves the
    combination from the spectral AOT of AERONET but is used only when called explicitly. Make the
    retrieval the default when the spectral AOT is available.

## 3. Uncertainty budget (medium term)

11. **Per-band uncertainty of the TOA reflectance.** Propagate the uncertainties of Rrs, AOT / aerosol
    model, water vapour, O3 and pressure, plus the model residuals (gas coupling: residuals per window
    from the coupling studies; absorption tables: REPTRAN vs CKD). Monte Carlo or linearized propagation
    through `Process`. Expected for Cal/Val products and not provided today.

## 4. Software quality

12. **Traceability of the outputs:** package version, `o2_model`, `gas_coupling`, coupling table, aerosol
    combination and LUT DOI as attributes of `radcalnet_db`.
13. **Refactor `Process.execute`:** separate the physics from the assembly of the output dataset, so that
    the coupled terms can be tested on their own.
14. **Regression tests on reference outputs:** template → band TOA reflectance with the light LUT, so that
    changes such as a new default show up as deliberate differences in CI.
15. **Open TODOs in `coxmunk.py`:** consistency of sigma2 with the formulation (line 104), sign of the
    rotation (line 112).
16. **Reproducibility of `scripts/sos_abs_coupling`:** README with the `sosabs` dependency, the data
    (`SOS_ABS_V3`, ~0.7 GB for v3) and the steps simulations → fit → notebook; decide whether the candidate
    CSV is tracked.
17. **Performance:** the 19-window coupling doubles the cost of the coupled transmittance (0.9 → 1.9 s for
    3 terms and 20 geometries); build the window subsets once per run or vectorize over the windows.

## 5. Release

18. **Release 1.1.** The `[Unreleased]` section of `CHANGELOG.md` already holds major changes (line-by-line
    O2, path-length coupling by default, `Spectral.from_srf`): release them before the full-range coupling,
    so that the 684-990 nm defaults ship on their own.

## Suggested order

18 → 7 → 1-3 → 12 / 14 → 4 → 8 / 10 → 11, with 5, 6, 9, 13 and 15-17 alongside. Wind (7) and Rrs (8)
probably weigh more on the TOA error than the remaining coupling work, but this is not quantified yet.
