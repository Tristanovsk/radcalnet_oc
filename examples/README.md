# Example notebooks

Working notebooks of the development of `radcalnet_oc`, kept as examples. They are not part of the
documentation (the rendered tutorials are in `docs/source/tutorials`) and are not maintained:
most of them read local data (AERONET-OC files under `/DATA/AERONET`, satellite datacubes, ...) and
some depend on other packages (e.g. `invRrs` for the hyperspectral reconstruction of the AERONET-OC
remote-sensing reflectance).

| Notebook | Content |
|---|---|
| `radcalnet_oc_aeronet_oc_chain.ipynb` | Production of the RadCalNet-OC time series of an AERONET-OC site (monthly files, full resolution and 10 nm bands) |
| `radcalnet_oc_aeronet_oc_process.ipynb` | Exploration of the AERONET-OC inputs, irradiance, gaseous transmittance and TOA simulations |
| `radcalnet_oc_aeronet_oc_uncertainty.ipynb` | Testbed for the propagation of the uncertainty of $R_{rs}$ and of the AOT |
| `minimal_chain/` | Minimal chain from an AERONET-OC file to the RadCalNet-OC product, with the AAOT data of 2 July 2025 (needs `invRrs`) |
| `aeronet/` | AERONET-OC datacubes and `invRrs` hyperspectral reconstruction |

The rendered tutorials of the documentation are in `docs/source/tutorials` (research studies in
`docs/source/tutorials/research`), starting with *Simulate the TOA signal from user settings*
(`docs/source/tutorials/radcalnet_oc_toa_simulation.ipynb`).
