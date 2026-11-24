# Satellite Cal/Val

The simulated TOA signal compared with the measurements of satellite sensors over the AERONET-OC sites:

- **Sentinel-3/OLCI matchups with RadCalNet-OC** (AERONET-OC station Galata, Black Sea): extraction of the OLCI
  pixels around the station from datacubes of Level-1B images, screening of the scenes (quality, cloud and sunglint flags, viewing geometry, spatial
  uniformity), simulation of the AERONET-OC measurements within one hour of each overpass at the geometry of the
  scene with the direct sunglint, integration over the OLCI bands, and the ratio of the measured to the simulated
  radiance per band and platform.

The Sentinel-2 MSI matchups and the multi-site, multi-year Sentinel-3 monitoring are made with the scripts and
notebooks of the repository (`scripts/s2_toa_matchup`, `notebook/sentinel3`); their results are in the RadCalNet
site uncertainty statements.

```{toctree}
:maxdepth: 1

sentinel3_radcalnet_oc_matchup
```
