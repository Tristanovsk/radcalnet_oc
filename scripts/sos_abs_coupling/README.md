# Gaseous absorption coupled with scattering: SOS-ABS simulations and fit

Scripts of the parameterisation of the gaseous transmittance of each term of the TOA signal (path reflectance,
water-leaving term, irradiance) used by `radcalnet_oc.kernel.GasCoupling` (methods page of the documentation,
section *Coupling of absorption and scattering*). They are not part of the package.

## Requirements

- radcalnet_oc (development install) and its dependencies.
- The SOS-ABS radiative transfer code and its Python wrapper `sosabs`
  (`RadiativeTransferCode-SOS`, set `SOS_ABS_ROOT`, default `/DATA/git/vrtc/RadiativeTransferCode-SOS`;
  `pip install -e $SOS_ABS_ROOT/python`), only to run the simulations (`make_luts.py`). Its CKD coefficients
  (`fic/COEFF_CKD/ckd_5cmm1.nc`, 5 cm-1) start at 364 nm.
- The simulations, about 1 GB: `data/v2` (684-990 nm, 0.3 GB) and `data/v3` (rest of 364-2500 nm, 0.7 GB),
  plus `data/` for v1. `data` is not tracked: on the development machine it is a link to
  `/data/sos_abs_coupling`. `SOS_ABS_V3` overrides the location of v3.

## Steps

1. **Simulations** (SOS-ABS, maritime aerosols, US62 profile, 1013 hPa, Cox-Munk 5 m s-1, Lambertian water;
   with and without absorption, and the direct-beam transmittance versus air mass):

   ```bash
   python make_luts.py v2 data/v2 8     # 684-990 nm: H2O 0.5-4 g cm-2, SZA 20-70 deg, AOT(550) 0.05-0.4
   python make_luts.py v3 data/v3 8     # 364-2500 nm outside 684-990 nm, absorbing CKD intervals only
   ```

2. **Fit, 684-990 nm** (tables shipped in the package, `radcalnet_oc/data/auxdata`):

   ```bash
   python coupling.py fit data/v2 ../../radcalnet_oc/data/auxdata/gas_coupling_eta.csv  # effective air mass
   python ppld.py fit data/v2 ../../radcalnet_oc/data/auxdata/gas_coupling_ppld.csv     # photon path-length model (default)
   python ppld_eval.py                                                                    # band-level comparison of the models
   ```

3. **Full range, 364-2500 nm** (candidate, not used by the package yet):

   ```bash
   python fullrange.py fit                  # -> gas_coupling_ppld_full.csv (19 windows, 57 models)
   python build_fullrange_notebook.py       # notebook/research/radcalnet_oc_gaseous_coupling_350_2500nm.ipynb
                                            # and conclusions_fullrange.md
   ```

4. **Slides** (optional): `slides/extract_figures.py` extracts the figures of the executed notebook,
   `slides/make_slides.js` builds the deck (generated files ignored).

## Tracked results

- `gas_coupling_eta_v1.csv`: first effective-air-mass fit (v1 simulations), kept for reference.
- `gas_coupling_ppld_full.csv`: candidate full-range table (same format as `gas_coupling_ppld.csv`). Before
  adopting it (ROADMAP task 14): keep O3-VIS and CO2 1600 nm decoupled, and decide on the CH4 windows, fitted
  on the total transmittance while the package couples only H2O and the background gases
  (`GaseousTransmittance.COUPLED_GASES`).
- `conclusions_fullrange.md`: conclusions of the full-range study.
