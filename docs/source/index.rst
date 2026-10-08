radcalnet-oc documentation
==========================

``radcalnet_oc`` models the radiative transfer through the atmosphere to propagate the water-leaving
radiance measured at the surface, typically by AERONET-OC stations, up to the top-of-atmosphere level.
The simulated top-of-atmosphere reflectance is used for the calibration and validation (Cal/Val) of
passive optical satellite sensors above coastal and oceanic waters.

.. math::

   R_{toa}(\lambda) = T_g^{\downarrow} T_g^{\uparrow}
   \left[ \pi\, R_{rs}(\lambda)\, T^{\downarrow}(\lambda)\, t^{\uparrow}(\lambda) + R_{atm}(\lambda) \right]

.. figure:: _static/slide_boa2toa.png
   :figclass: only-light
   :alt: Propagation of the water-leaving signal from bottom to top of atmosphere

.. only:: html

   .. figure:: _static/slide_boa2toa_dark.png
      :figclass: only-dark
      :alt: Propagation of the water-leaving signal from bottom to top of atmosphere

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   usage
   methods
   processing_chain

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebook/radcalnet_oc_gaseous_transmittance
   notebook/radcalnet_oc_solar_irradiance
   notebook/radcalnet_oc_downwelling_irradiance
   notebook/radcalnet_oc_toa_atmosphere_radiation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   history

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
