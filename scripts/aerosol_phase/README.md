# NIR excess and aerosol phase function

The study behind ROADMAP task 10 (open near-infrared term). Not part of the package.

Both Sentinel-3 OLCI and Sentinel-2 MSI read about 2e-3 more TOA reflectance than radcalnet_oc 1.2 at 865 nm. The
same additive excess appears at all three RadCalNet sites and on both S3A and S3B. These scripts characterise the
excess and test one explanation: the phase function of the OPAC aerosol mixture, compared with the AERONET
almucantar retrievals at AAOT.

## Requirements

- radcalnet_oc 1.2 matchup files: the Sentinel-3 Cal/Val monitoring output
  (`/data/radcalnet/matchups/sentinel3/v1.2.0/..._VEIT-GABU-CORO_..._matchup.nc`) and the Sentinel-2
  `*_cm_iso_mix_roi.nc` files of VEIT L2.0 and CORO L1.5 (`/data/radcalnet/matchups/sentinel2/v1.2.0`).
- The mopsmap OPAC scattering matrices of the TOA LUT (`/DATA/git/vrtc/mopsmap_tbx/scatmat/ascii`, as in
  `scripts/sos_abs_coupling/make_luts.py`).
- Paths can be changed with the environment variables `S3_MATCHUP`, `S2_MATCHUP_DIR`, `AEROSOL_PHASE_DIR`
  (default `/data/radcalnet/studies/aerosol_phase`) and `MOPSMAP` (`phase_common.py`).

## Steps

```bash
python nir_excess.py all                 # bands, drivers, gas, phase: the excess in both sensors (about 20 s)
python download_aeronet_inversions.py    # AERONET V3 PFN, AOD, SSA of AAOT (Level 1.5, all points)
python pfn_compare.py                    # AERONET omega*P(865) vs the OPAC mixture; writes pfn_compare_AAOT.npz (about 45 s)
python pfn_predict.py                    # single-scattering prediction of the matchup excess, with a control (about 25 s)
```

## Results (27 November 2026)

- **The excess is additive and comes from below the water vapour.** At 865 nm it is +1.6 to +2.4e-3 for OLCI and
  +1.6 to +1.9e-3 for MSI. In the H2O bands it is attenuated like the surface signal (OLCI 940 nm 0.52 against 0.45,
  MSI B9 0.56 against 0.38). In OLCI it is not glint: it is largest for the scenes without glint flag.
- **It has a backscatter signature.** With dR = c0 + dP(theta) tau865 / (4 mu_s mu_v), both sensors give the same
  dP curve: rising from about -0.04 at 90-130 deg to +0.03 at 150-170 deg, also at a fixed sun zenith angle.
- **AERONET finds more backscatter than the OPAC mixture.** At AAOT (4670 retrievals) AERONET omega*P(865) is
  12-22 % above the fitted mixture (95 % WASO_rh0). The gap is +0.03 at 90-150 deg and +0.05 at 160-170 deg. No
  single OPAC component closes it.
- **This predicts about half of the VEIT excess.** The same-day difference predicts 1.3 of 2.4e-3 for OLCI (slope
  0.59 +/- 0.08) and 0.9 of 2.3e-3 for MSI (slope 0.80 +/- 0.19). A control with no angular shape has no skill.
  About 1e-3 remains where the prediction is zero.
- **Caveats.** The almucantar measures up to 2 sza (median 124 deg), so the backscatter part is the extrapolation
  of the retrieval. The AERONET single scattering albedo is unreliable at these low AODs. The prediction uses single
  scattering, and the inversions cover AAOT only.
