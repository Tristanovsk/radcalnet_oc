# Validation of the pressure scaling of the Rayleigh scattering with OSOAA

`Process(rayleigh_pressure=True)` scales the Rayleigh reflectance of the look-up tables (OSOAA runs with the
Rayleigh optical thickness of 1015.2 hPa) as `R_ray [(P/P0)^D - 1]`, with `D = dln R_ray / dln tau_r` from the
wavelength nodes. These scripts check it against OSOAA runs at other pressures.

1. `run_pressure.py`: OSOAA runs with the parameters of the LUT runs (attributes of the files of
   `/data/vrtc/lut/atmo_full`), `AP.MOT` scaled by `P / 1015.2`. The executable and the netCDF converter of
   `OSOAA_profile` are used read-only; the working files, the surface matrices and the outputs are written in the
   folder of the script (`work`, `db`, `out`). Arguments: solar zenith angles, wavelengths (micron), pressures (hPa),
   `aot:model` cases, number of processes. Run first the reference pressure of each wavelength (it computes the
   surface matrices), then the others in parallel:

   ```bash
   export OSOAA_ROOT=/DATA/VRTC/OSOAA_harmel
   python run_pressure.py 20,40,60 0.4,0.5,0.8 1015.2 0.0000:COAV_rh70 3
   python run_pressure.py 20,40,60 0.4,0.5,0.8 995.2,1015.2,1035.2 0.0000:COAV_rh70,0.1:COAV_rh70 6
   ```

   About 70 s per solar angle and process, 0.3 GB per process. With aerosols, the converter stops on the
   aerosol metadata (refractive index printed `********` by OSOAA) after writing the radiances, which are used.
   The run at 1015.2 hPa reproduces the LUT file exactly (maximum relative difference 0).

2. `compare.py [folder of the runs]`: TOA path reflectance change of OSOAA against the scaling of radcalnet-oc (D from the AOT 0 runs of
   the LUT at the 10 wavelength nodes), for the solar zenith angles 20, 40, 60, VZA 0-40 and azimuths 0-180 deg.
