.. _api:

API reference
=============

The main classes are importable from the package itself:

.. code-block:: python

   import xarray as xr
   import radcalnet_oc as radoc

   process = radoc.Process(input_db=xr.open_dataset('input_time_series.nc'))
   process.execute()
   process.radcalnet_db.Rtoa

Processing chain
----------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.process.Process

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.run.main
   radcalnet_oc.log.set_logging
   radcalnet_oc.process.package_version

Look-up tables and auxiliary data
---------------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.lut.LUT
   radcalnet_oc.lut.AuxData
   radcalnet_oc.lut.SolarIrradiance

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.lut.open_lut
   radcalnet_oc.lut.interp_wl_loglog

Remote-sensing reflectance
--------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.water.PureWater
   radcalnet_oc.reconstruction.Reconstruction

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.water.extend_rrs
   radcalnet_oc.reconstruction.reconstruct_rrs
   radcalnet_oc.reconstruction.invrrs_version

Configuration and download of the TOA look-up table
---------------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.config.load_config
   radcalnet_oc.config.toa_lut_path
   radcalnet_oc.config.trans_lut_path
   radcalnet_oc.config.default_data_root
   radcalnet_oc.config.ensure_toa_lut
   radcalnet_oc.config.download_toa_lut
   radcalnet_oc.config.auto_download_enabled
   radcalnet_oc.config.init_user_config
   radcalnet_oc.config.main

Atmosphere
----------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.kernel.GaseousTransmittance
   radcalnet_oc.kernel.GasCoupling
   radcalnet_oc.kernel.O2LineByLine
   radcalnet_oc.kernel.Gases
   radcalnet_oc.kernel.Aerosol
   radcalnet_oc.kernel.Misc
   radcalnet_oc.kernel.Radiometry

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.kernel.path_length_transmittance

Sunglint
--------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.coxmunk.Sunglint

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.coxmunk.glint_angle
   radcalnet_oc.coxmunk.glint_flags
   radcalnet_oc.coxmunk.glint_uncertainty
   radcalnet_oc.coxmunk.flag_attrs
   radcalnet_oc.coxmunk.uncertain_attrs

Sunglint flags and uncertainty settings (module ``radcalnet_oc.coxmunk``):

.. autodata:: radcalnet_oc.coxmunk.GLINT_FLAGS
.. autodata:: radcalnet_oc.coxmunk.GLINT_LEVELS
.. autodata:: radcalnet_oc.coxmunk.GLINT_ANGLE_MIN
.. autodata:: radcalnet_oc.coxmunk.GLINT_ENSEMBLE
.. autodata:: radcalnet_oc.coxmunk.WIND_UNC
.. autodata:: radcalnet_oc.coxmunk.GLINT_UNC_THRESHOLD

Uncertainty budget
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.uncertainty.uncertainty_budget
   radcalnet_oc.uncertainty.model_uncertainty

.. autodata:: radcalnet_oc.uncertainty.INPUT_UNC

Spectral response of the sensors
--------------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.lut.Spectral

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.lut.gaussian
   radcalnet_oc.lut.super_gaussian
   radcalnet_oc.lut.Gamma2sigma
   radcalnet_oc.lut.super_gaussian_fwhm2sigma

AERONET-OC data
---------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.aeronet_oc.Aeronet

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.aeronet_oc.aeronet_inputs

RadCalNet-OC products
---------------------

Daily products of an AERONET-OC site (module ``radcalnet_oc.products``, command ``radcalnet_oc-products``, see
:ref:`usage-products`).

.. autosummary::
   :toctree: generated
   :nosignatures:

   radcalnet_oc.products.make_products
   radcalnet_oc.products.simulate_month
   radcalnet_oc.products.to_10nm
   radcalnet_oc.products.write_daily
   radcalnet_oc.products.glint_ok
   radcalnet_oc.products.glint_filtered
   radcalnet_oc.products.daily_file
   radcalnet_oc.products.monthly_file
   radcalnet_oc.products.main

.. autodata:: radcalnet_oc.products.CENTRAL_WL
.. autodata:: radcalnet_oc.products.FWHM
.. autodata:: radcalnet_oc.products.EXPON
.. autodata:: radcalnet_oc.products.FILTER_FLAGS
.. autodata:: radcalnet_oc.products.UNC_RANGE
