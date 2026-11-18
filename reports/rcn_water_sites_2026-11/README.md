# RadCalNet water sites: update for the RCN meeting (November 2026)

Answers to the three points raised after the October meeting: infrastructure filtering at VEIT, the
sun-glint filtered VEIT dataset, and the new Romanian site (Section-7 Platform, Black Sea). All the datasets are
made with radcalnet_oc **1.2.0** (`radcalnet_oc_version` attribute of each file; CHANGELOG).

## 1. VEIT: infrastructure filtering (ROI and pixel screening)

Measured on the Sentinel-2 L1C pixels (20 m) around the platforms, over the clear scenes of the matchups (valid
fraction > 0.5, AOT(550) < 0.3, glint angle > 30°): 325 scenes at VEIT (AAOT, AERONET-OC L2.0) and 24 at
Section-7 (L1.5). Script: `scripts/s2_toa_matchup/infrastructure_rings.py`.

Median TOA reflectance of each ring minus the median of the 1.2 km x 1.2 km box (1e-3), and fraction of the pixels
brighter than the box median + 4 robust standard deviations at 865 nm (90th percentile over the scenes):

| distance to the site | VEIT 865 nm | VEIT outliers | Section-7 865 nm | Section-7 outliers |
|---|---|---|---|---|
| 0-30 m    | +0.50 | 38 % | +11.1 | 100 % (71 % with B11 >= 0.03) |
| 30-60 m   | +0.20 | 21 % | +2.5  | 55 % |
| 60-100 m  | 0.00  | 4 %  | +0.33 | 7 %  |
| 100-150 m | 0.00  | 1 %  | +0.10 | 0.7 % |
| > 150 m   | 0.00  | < 1 % (2 % beyond 600 m: ships) | 0.00 | < 1 % |

- The structure (tower, shadow, moored boats) is visible up to 60-100 m at VEIT and up to 100-150 m at Section-7
  (a larger platform).
- The **median** of a 600 m half-size box is not affected by it: excluding a disc of 30-200 m around the site
  changes the box median by < 0.03e-3 at 490, 665 and 865 nm at both sites. A mean, or a small ROI, is affected:
  for a 150 m half-size box without exclusion, +0.2e-3 at 865 nm at Section-7.
- Larger ROIs bring the spatial gradients of the coastal water: at VEIT the median at 490 nm changes by -0.025e-3
  (900 m half-size) and -0.05e-3 (1.2 km), and the pixel spread (IQR) grows from 1.2e-3 (600 m) to 1.6e-3 (1.2 km).

**Recommended for 10-30 m sensors (MSI, OLI):**
1. ROI: square of 600 m half-size (1.2 km side) centred on the site; statistic: **median** of the valid pixels.
2. Exclude a disc of **100 m radius at VEIT** and **150 m at Section-7** around the platform.
3. Pixel screening: B11 (1.6 um) >= 0.03 (structures, ships, clouds, strong glint); cirrus band (1.375 um) >= 0.01;
   outliers at 865 nm above the ROI median + 4 robust standard deviations (ships and wakes outside the disc).
4. Keep the scene if more than 50 % of the ROI pixels are valid (at least 50 pixels).

For 300 m sensors (OLCI), the platform takes a small part of the central pixel; use the median of 3x3 pixels
centred on the site, excluding the central one when it is flagged bright in the SWIR or at 865 nm.

## 2. VEIT: sun-glint filtered dataset

The RCN TOA files are simulated at **nadir**, so the direct glint depends only on the solar zenith angle and on the
wind. The look-up tables hold the diffuse light only: `Rtoa` does **not** include the direct glint, as before. New
in version 1.2, each daily file carries:

- `Rglint`: TOA direct sunglint at nadir (heavy-tailed slope statistics `cm_iso_mix`, wind of the station),
  attenuated by the direct transmittance; `Rtoa_glint = Rtoa + Rglint`;
- `Rglint_unc`: spread of the plausible slope statistics at the wind +/- 1.5 m/s; `glint_uncertain` (per band):
  `Rglint_unc / Rtoa_glint` > 5 %;
- `glint_fraction` (glint / total at 865 nm), `glint_angle` and `glint_flags` (bitmask, CF `flag_masks`):
  1 low (>= 1 %), 2 moderate (>= 10 %), 4 high (>= 50 %), 8 near specular (glint angle < 30°).

The **filtered copy** (`<site>_glint_filtered/`) keeps only the times without the bits 2 (moderate) and 8 (near
specular). Stricter filters can be applied with the flags (e.g. bit 1, or `glint_uncertain` in the bands used).

Important: at nadir, the glint is strong in summer around noon. In July 2020 at VEIT (solar zenith 22-40°, wind
1.6 m/s) the median glint fraction at 865 nm is 19 % and 260 of 399 times are near specular: only 58 times pass the
filter. The glint seen by a sensor depends on its viewing geometry, which the nadir file does not have. For
Sentinel-2 the near-specular overpasses are in May-August (86 of 489 VEIT scenes, 52 of 95 at Section-7), and the
simulated glint removes most of the near-infrared deficit at these geometries. For each overpass, the glint
should be recomputed at the sensor geometry (we can provide it per overpass, from the list of acquisitions), or
the overpasses with a glint angle < 30° should be removed.

## 3. New Romanian site: Section-7 Platform

AERONET-OC Section-7_Platform (44.546 N, 29.447 E, Black Sea, Sentinel-2 tile 35TPK; Gloria, the former site
nearby, stopped in August 2019). Data: L2.0 from 27 August 2019 to 18 August 2024, L1.5 from 19 August 2024 to
13 November 2026. Daily RCN files with the same content as VEIT (nadir, 10 nm, glint variables and filtered copy):
L2.0 for 2019-2024 and L1.5 for 2024-2026 (the level is in the file name).

TOA check with Sentinel-2 MSI L1C (24 clear scenes at L1.5, glint angle > 30°, ROI of section 1 without the
exclusion disc): simulation minus MSI, median

| | 443 | 490 | 560 | 665 | 705 | 865 | 1610 | 2190 nm |
|---|---|---|---|---|---|---|---|---|
| without the direct glint | -3.0 % | -3.3 % | -5.5 % | -13 % | -17 % | -28 % | -62 % | -75 % |
| with the simulated glint (Cox-Munk at the station wind) | -1.6 % | -1.1 % | +0.5 % | -2.2 % | -5.9 % | -10 % | -14 % | -20 % |

The Section-7 overpasses are glint-dominated in the near infrared and the SWIR even at glint angles > 30°:
use the glint-filtered times, or the visible bands only, until the glint at the sensor geometry is in the
comparison. Same order as VEIT once the glint is included (VEIT L2.0, 325 scenes: -1.6 % at 443 nm, -0.1 % at
490 nm, +0.8 % at 560 nm, -2.9 % at 665 nm).

## Files

`/data/radcalnet/data_v1.2.0/radcalnet/<site>/<yyyy>/<mm>/<dd>/<site>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc`
and `/data/radcalnet/data_v1.2.0/radcalnet_glint_filtered/...` (sites `VEIT`, `Section-7_Platform`), made with
`scripts/radcalnet_products/make_products.py` (radcalnet_oc `1.2.0+g867ddea`, same code as the tag `v1.2.0`).

| site | level | period | days | times | times kept by the glint filter (days) |
|---|---|---|---|---|---|
| VEIT (AERONET-OC AAOT) | L2.0 | 2017-10-23 to 2025-07-16 | 1614 | 22151 | 14414 (1207) |
| VEIT (AERONET-OC AAOT) | L1.5 | 2025-07-17 to 2026-10-06 | 313 | 5486 | 3332 (225) |
| Section-7_Platform | L2.0 | 2019-08-27 to 2024-08-18 | 809 | 10175 | 7152 (762) |
| Section-7_Platform | L1.5 | 2024-08-19 to 2026-11-13 | 348 | 8565 | 5521 (338) |

Size: 520 MB (all days) and 370 MB (glint filtered), about 130 kB per day.
