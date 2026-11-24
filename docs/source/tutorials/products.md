# AERONET-OC and RadCalNet-OC products

The RadCalNet-OC products of a site are daily files of the simulated TOA reflectance at nadir, every 10 nm from
350 to 2490 nm (10 nm bands), for each AERONET-OC measurement, with the direct sunglint, its uncertainty and its
flags, and a copy of the files without the glinted times. They are made by batch processing of the whole
AERONET-OC record of the site ({func}`~radcalnet_oc.products.make_products`, command `radcalnet_oc-products`, see
{ref}`usage-products`).

- **RadCalNet-OC products of an AERONET-OC site**: the AERONET-OC database (installation, configuration, update,
  catalogue of the sites), the batch processing of one month of a site and its restart from the monthly cache, the
  tree of the output files, the variables of a daily file (TOA reflectance with and without glint, glint
  uncertainty, flags and their decoding, inputs, provenance), the glint-filtered copy, and how to process a whole
  site.
- **From AERONET-OC to top of atmosphere**: each step of the chain in detail on one day, from a prepared input file:
  the aerosol mixture retrieved and its fit of the spectral aerosol optical thickness, the simulation at full
  spectral resolution, the convolution to 10 nm bands and a propagation of the uncertainty of the inputs.

```python
from radcalnet_oc.products import make_products

summary = make_products('AAOT', 'lev20', name='VEIT', start='2025-04-01', end='2025-04-30',
                        odir='radcalnet_oc_products')
# radcalnet_oc_products/daily/VEIT/2025/04/DD/VEIT_radcalnet-oc_lev20_202504DD_inc10nm_res10nm.nc
# radcalnet_oc_products/daily_glint_filtered/VEIT/...   (times without moderate or near-specular glint)
```

```{toctree}
:maxdepth: 1

radcalnet_oc_products
radcalnet_oc_aeronet_oc_toa
```
