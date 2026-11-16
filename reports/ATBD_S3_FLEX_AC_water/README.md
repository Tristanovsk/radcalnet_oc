# ATBD: radiative transfer model for FLEX simulations and atmospheric correction above water

LaTeX source of the PHY2FLEX ATBD (ref. PHY2FLEX-ATBD-MAG-23), edited on Overleaf
(project `6afad4facb7980497f2d41c8`, which supports git: `git clone https://git.overleaf.com/6afad4facb7980497f2d41c8`).
This folder is a copy; sync it from Overleaf after editing there.

Build: `latexmk -pdf main.tex` (figures in `figures/`).

Issue V1.1 (16/11/2026) is based on radcalnet_oc 1.1 and S3FLEX_ACaqua 0.1.0. Its new figures
(`gas_coupling_flex`, `o2_lbl_reptran_flex`, `ratm_gas_coupling_flex`, `basic_pixel_*`, `sunglint_pixel_hyp_simu`) are made
by the notebook `notebook/atbd_v1.1_figures.ipynb` of S3FLEX_ACaqua, from the Sentinel-3 FLEX-mode image
`S3A_S3B_OL_1_EFR____20180702T101353_..._stacked.zarr` (pixels: bloom x=2149, y=5310; sunglint x=4060, y=5611).
