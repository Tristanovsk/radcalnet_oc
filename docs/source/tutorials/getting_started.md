# Getting started

Two short notebooks to run a first simulation of the top-of-atmosphere (TOA) signal with `radcalnet_oc`:

- **Quick start: from AERONET-OC to the TOA signal**: one day of measurements of the AERONET-OC station AAOT
  (northern Adriatic Sea), read from the AERONET-OC database, the hyperspectral water reflectance reconstructed
  with invRrs, the aerosol mixture retrieved from the spectral aerosol optical thickness, the simulation at two
  viewing angles with the direct sunglint and its flags, the convolution to the 10 nm bands of the RadCalNet-OC
  products and the uncertainty budget of one measurement. Start here.
- **Simulate the TOA signal from user settings**: the same simulation without AERONET-OC data, from settings of
  the geometry, the atmosphere (aerosol optical thickness, aerosol models, gas columns), the wind and the water
  reflectance, the options of the processor (gaseous coupling, O$_2$ model, spectral extension of the water
  reflectance), sensitivity studies and the convolution with the bands of a sensor.

```python
import xarray as xr
import radcalnet_oc as radoc
from radcalnet_oc.aeronet_oc import aeronet_inputs
from radcalnet_oc.reconstruction import Reconstruction
from radcalnet_oc.products import INPUT_VARS, to_10nm

records = aeronet_inputs('AAOT', 'lev20', start='2025-04-30', end='2025-04-30')
hyp = Reconstruction().reconstruct(records.Rrs_Lwn, records.sza_aeronet, chl=records.chl)
hyp = hyp.where(hyp.success, drop=True)
records = records.sel(time=hyp.time)
input_db = records[INPUT_VARS + ['naot']].assign(Rrs=hyp.Rrs_nadir, sza=records.sza_aeronet,
                                                 day_of_year=records.time.dt.dayofyear)
process = radoc.Process(input_db, vza=xr.DataArray([0.], coords={'vza': [0.]}, dims='vza'),
                        azi=xr.DataArray([90.], coords={'azi': [90.]}, dims='azi'))
aerosol = radoc.Aerosol(input_db.naot.dropna('wl_aeronet', how='all'), process.lut.naot_lut)
aerosol.process()
process.execute(aerosol_combination=aerosol.model_db.aerosol_combination, sunglint='cm_iso_mix')
db10 = to_10nm(process.radcalnet_db)          # Rtoa, Rtoa_glint, glint_flags... every 10 nm
```

The AERONET-OC database (`aeronet_oc_db`) and invRrs are optional dependencies: see {doc}`../installation` and the
first section of {doc}`radcalnet_oc_products`.

```{toctree}
:maxdepth: 1

radcalnet_oc_aeronet_oc_quickstart
radcalnet_oc_toa_simulation
```
