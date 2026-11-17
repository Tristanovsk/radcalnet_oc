# Wavelength interpolation of the TOA look-up table

The TOA LUT of radcalnet_oc keeps 10 wavelength nodes (0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.6, 2.2, 2.5 um) of the
OSOAA runs of `/data/vrtc/lut/atmo_full`. These scripts check the interpolation of the path reflectance between them
against OSOAA runs of the same configuration at other wavelengths (roadmap task 25).

1. `run_offnode.py <wavelengths um> <aot:model,...> [n processes] [output folder]`: OSOAA runs between 1.0 and
   1.6 um, where the archive has no run. Parameters of the 1.0 um run of the archive, with the wavelength-dependent
   ones set as in the LUT runs (Rayleigh optical thickness and depolarisation of Bodhaine et al. 1999 at 1015.2 hPa,
   mopsmap scattering matrix at the wavelength, refractive index of the water interpolated). Outputs in
   `/data/vrtc/lut/osoaa_offnode`. About 135 s per case and 0.3 GB per process; empty surface matrix files left by an
   interrupted run are removed at the start.

   ```bash
   systemd-run --user --scope -p MemoryMax=2500M -p MemorySwapMax=0 python run_offnode.py \
       1.15,1.25,1.4 0.0000:COAV_rh70,0.1:COAV_rh70,0.1:MACL_rh70,0.35:DESE_rh70,0.35:COAV_rh70 4
   ```

2. `compare_schemes.py [output csv]`: TOA path reflectance and spectral AOT interpolated from the 10 nodes with six
   schemes (linear, quadratic, PCHIP, spline; linear or log-log space), against the OSOAA runs that the LUT leaves out
   (0.45-0.9 and 2.4 um of the archive, 1.15-1.4 um of `run_offnode.py`), wind 2 m s-1, sza 20-60 deg, VZA 0-40 deg,
   azimuths 0-180 deg. About 12 min, 0.3 GB.

Result (37 cases, median signed error of the path reflectance, AOT <= 0.1):

| scheme | 450 | 550 | 650 | 750 | 850 | 900 | 1150 | 1250 | 1400 | 2400 nm |
|---|---|---|---|---|---|---|---|---|---|---|
| quadratic, linear space (before 1.2) | -1.04 | -0.50 | -0.35 | -0.10 | -1.15 | -0.84 | -6.35 | -9.09 | -4.20 | -2.88 % |
| quadratic, log-log (from 1.2) | 0.04 | 0.01 | 0.03 | -0.03 | -0.01 | -0.36 | 0.00 | 0.03 | 0.06 | -0.62 % |

For AOT 0.35, the log-log scheme is within 1.8 % (+1.1 to +1.8 % at 1150-1400 nm), the previous one within
0.1-1.4 %. The spectral AOT is better interpolated in linear space (median error < 0.7 %) and is left as before.
