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
   radcalnet_oc.kernel.Gases
   radcalnet_oc.kernel.Aerosol
   radcalnet_oc.kernel.Misc
   radcalnet_oc.kernel.Radiometry

Sunglint
--------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.coxmunk.Sunglint

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
