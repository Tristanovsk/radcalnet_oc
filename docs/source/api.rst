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

Look-up tables and auxiliary data
---------------------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   radcalnet_oc.lut.LUT
   radcalnet_oc.lut.AuxData
   radcalnet_oc.lut.SolarIrradiance

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
