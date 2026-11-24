# Physics of the simulation

Terms of the TOA equation computed by `radcalnet_oc`, each in its own notebook (equations in {doc}`../methods`):

- **Gaseous transmittance**: direct transmittance of the absorbing gases of the atmosphere (water vapour, ozone,
  nitrogen dioxide, methane and the well-mixed gases) for a given composition and geometry, the absorption of each
  gas, and how it is seen by sensors of different spectral resolutions.
- **Solar irradiance**: the extraterrestrial solar spectra of the package compared, seen by sensors of different
  spectral resolutions, and the solar irradiance at the top and at the bottom of the atmosphere.
- **Direct and total downwelling irradiance**: the irradiance reaching the sea surface for a given atmosphere,
  split into the direct solar beam and the diffuse sky light, its sensitivity to the aerosols, water vapour and
  ozone, the photosynthetically available radiation, and the values in the bands of a sensor. The same terms are
  given for the measurements of a simulation by {meth}`~radcalnet_oc.process.Process.boa_irradiance`.

```{toctree}
:maxdepth: 1

radcalnet_oc_gaseous_transmittance
radcalnet_oc_solar_irradiance
radcalnet_oc_downwelling_irradiance
```
