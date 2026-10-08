Usage
=====

``radcalnet_oc`` simulates the top-of-atmosphere (TOA) signal above a water site from a time series
of measurements at the surface, typically those of an AERONET-OC station.

Input data
----------

The input is an :py:class:`xarray.Dataset` with a ``time`` dimension (one row per measurement) and a
``wl`` dimension (wavelength in nm) for the water reflectance:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Dimensions
     - Description
   * - ``Rrs``
     - time, wl
     - remote-sensing reflectance (sr\ :sup:`-1`), hyperspectral or interpolated from the
       AERONET-OC bands
   * - ``sza``
     - time
     - solar zenith angle (deg)
   * - ``day_of_year``
     - time
     - day of year, for the Earth-Sun distance
   * - ``aot550``
     - time
     - aerosol optical thickness at 550 nm
   * - ``aot865``, ``ang_exp``
     - time
     - aerosol optical thickness at 865 nm and Angström exponent (copied to the output)
   * - ``pressure``
     - time
     - surface pressure (hPa)
   * - ``tcwv``, ``tco3``, ``tcno2``
     - time
     - total columns of water vapour, ozone and nitrogen dioxide (kg m\ :sup:`-2`)

Examples are shipped with the package in ``radcalnet_oc/data/template``
(``template_clear_water.nc``, ``template_turbid_water.nc``).

Python
------

.. code-block:: python

   import numpy as np
   import xarray as xr
   from importlib.resources import files
   import radcalnet_oc as radoc

   input_db = xr.open_dataset(files('radcalnet_oc.data.template') / 'template_clear_water.nc')

   # viewing geometry of the satellite (one or several angles)
   vza = xr.DataArray([0., 10.], coords={'vza': [0., 10.]}, dims='vza')
   azi = xr.DataArray([90.], coords={'azi': [90.]}, dims='azi')

   process = radoc.Process(input_db=input_db, vza=vza, azi=azi)
   process.execute()                  # default aerosol model proportions of config.yml
   radcalnet_db = process.radcalnet_db

   # convolution with the spectral response of the sensor bands
   # (an xarray.Dataset for a multidimensional signal, a DataArray for one spectrum)
   spectral = radoc.Spectral(central_wl=np.array([443., 490., 560., 665., 865.]), fwhm=20.)
   Rtoa_bands = spectral.convolve2(radcalnet_db.Rtoa).Rtoa

The proportions of the aerosol models can be given explicitly,
``process.execute(aerosol_combination=[0, 0.5, 0, 0.5, 0, 0])``, or retrieved from the AERONET
spectral aerosol optical thickness with :py:class:`~radcalnet_oc.kernel.Aerosol` (see
:ref:`methods-aerosol`).

Output
------

:py:attr:`Process.radcalnet_db <radcalnet_oc.process.Process>` is an :py:class:`xarray.Dataset` at
the full spectral resolution of the simulation (350--2500 nm):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Variable
     - Unit
     - Description (see :doc:`methods`)
   * - ``Rtoa``
     - --
     - total reflectance at TOA, :math:`R_{toa}`
   * - ``Ratm``
     - --
     - atmosphere (intrinsic + surface-reflected) reflectance at TOA, :math:`T_g^{\downarrow}T_g^{\uparrow}R_{atm}`
   * - ``Rrs``
     - sr\ :sup:`-1`
     - remote-sensing reflectance used for the simulation
   * - ``E0``
     - mW m\ :sup:`-2` nm\ :sup:`-1`
     - plane solar irradiance at TOA
   * - ``Ed``
     - mW m\ :sup:`-2` nm\ :sup:`-1`
     - plane solar irradiance at the surface
   * - ``D2``
     - --
     - Earth-Sun distance correction factor
   * - ``aerosol_combination``
     - --
     - proportion of each aerosol model used for the simulation

together with the input atmospheric parameters (``tcwv``, ``tco3``, ``tcno2``, ``pressure``,
``aot550``, ``aot865``, ``ang_exp``).
