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
     - remote-sensing reflectance (sr\ :sup:`-1`), hyperspectral, e.g. reconstructed from the
       AERONET-OC bands with :py:func:`~radcalnet_oc.reconstruction.reconstruct_rrs`; interpolated
       linearly and extended beyond its spectral range (:ref:`methods-rrs`)
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
   * - ``wind`` (optional)
     - time
     - wind speed at 10 m (m s\ :sup:`-1`, e.g. ERA5), roughness of the sea surface for the reflection of the
       sky and the Sun; the look-up tables are interpolated linearly between their nodes (0.5-16 m s\ :sup:`-1`,
       values out of this range clipped). Without it, or for its missing values, ``settings: wind`` of the
       configuration (2 m s\ :sup:`-1`)

Examples are shipped with the package in ``radcalnet_oc/data/template``
(``template_clear_water.nc``, ``template_turbid_water.nc``).

Command line
------------

The ``radcalnet_oc`` command reads the input NetCDF file and writes the simulation (see `Output`_) in a
NetCDF file:

.. literalinclude:: ../../radcalnet_oc/run.py
   :language: text
   :start-at: Usage:
   :end-before: Exit status

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

   # or with the tabulated spectral response functions of the instrument,
   # an xarray.DataArray of dimensions (band, wl), wavelengths in nm
   spectral = radoc.Spectral.from_srf(srf)
   Rtoa_bands = spectral.convolve2(radcalnet_db.Rtoa).Rtoa

The convolution is vectorized over all the spectra and also works on a simulation read lazily with dask,
e.g. ``xr.open_mfdataset(files, chunks={'time': 1}).Rtoa`` (see :ref:`spectral_convolution` and the
tutorial :doc:`tutorials/sentinel3_radcalnet_oc_matchup`).

When several simulations are run in the same Python session (e.g. one per month or per site), the
parts of the atmosphere look-up table read by the first run are kept in memory
(:py:meth:`LUT.toa_lut <radcalnet_oc.lut.LUT.toa_lut>`) and the next runs do not read the file again.

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
     - atmosphere (intrinsic + surface-reflected) reflectance at TOA, :math:`T_g^{atm}R_{atm}`
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
``aot550``, ``aot865``, ``ang_exp``) and the wind speed (``wind``, input or default).

From AERONET-OC data
--------------------

:py:func:`radcalnet_oc.aeronet_oc.aeronet_inputs` reads the measurements of a site from the AERONET-OC database
(``aeronet_oc_db``, optional dependency: see :doc:`tutorials/radcalnet_oc_products` for its installation and
update) and converts them into the variables and units of the input (gas columns in kg m\ :sup:`-2`, AOT at 550
and 865 nm, normalized spectral AOT ``naot`` for the aerosol mixture, AERONET-OC ``Rrs_Lwn`` and chlorophyll for
the hyperspectral reconstruction with invRrs, :py:class:`~radcalnet_oc.reconstruction.Reconstruction`). The
whole chain for one day, with the direct sunglint, its flags and the uncertainty budget, is the tutorial
:doc:`tutorials/radcalnet_oc_aeronet_oc_quickstart`:

.. code-block:: python

   import radcalnet_oc as radoc
   from radcalnet_oc.aeronet_oc import aeronet_inputs
   from radcalnet_oc.products import INPUT_VARS
   from radcalnet_oc.reconstruction import Reconstruction

   records = aeronet_inputs('AAOT', 'lev20', start='2025-04-30', end='2025-04-30')
   hyp = Reconstruction().reconstruct(records.Rrs_Lwn, records.sza_aeronet, chl=records.chl)
   hyp = hyp.where(hyp.success, drop=True)
   records = records.sel(time=hyp.time)
   input_db = records[INPUT_VARS + ['naot']].assign(Rrs=hyp.Rrs_nadir, sza=records.sza_aeronet,
                                                    day_of_year=records.time.dt.dayofyear)

   process = radoc.Process(input_db, vza=vza, azi=azi)
   aerosol = radoc.Aerosol(input_db.naot.dropna('wl_aeronet', how='all'), process.lut.naot_lut)
   aerosol.process()
   process.execute(aerosol_combination=aerosol.model_db.aerosol_combination, sunglint='cm_iso_mix')

.. _usage-products:

RadCalNet-OC products of a site
-------------------------------

The official RadCalNet-OC products of an AERONET-OC site are made in batch with
:py:func:`radcalnet_oc.products.make_products` or its command line ``radcalnet_oc-products`` (needs
``aeronet_oc_db``, invRrs and the full TOA look-up table). For every measurement: hyperspectral :math:`R_{rs}`
reconstructed with invRrs, aerosol mixture retrieved from the spectral AOT, simulation at nadir with the direct
sunglint at the wind of the station (slope statistics ``cm_iso_mix``), convolution with super-Gaussian responses
of 10 nm FWHM every 10 nm from 350 to 2490 nm:

.. code-block:: bash

   radcalnet_oc-products --site AAOT --level lev20 --name VEIT --start 2017-01-01 --odir /path/to/products
   radcalnet_oc-products --site Section-7_Platform --level lev15 --name CORO --start 2024-08-19 --odir /path/to/products

.. code-block:: python

   from radcalnet_oc.products import make_products
   make_products('AAOT', 'lev20', name='VEIT', start='2025-07-01', end='2025-07-31', odir='products')

Output, one file per day::

   <odir>/daily/<name>/<yyyy>/<mm>/<dd>/<name>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc
   <odir>/daily_glint_filtered/<name>/<yyyy>/<mm>/<dd>/<name>_radcalnet-oc_<level>_<yyyymmdd>_inc10nm_res10nm.nc
   <cache>/<name>/<name>_radcalnet-oc_<level>_<yyyy>_<mm>_full_spectral.nc     (monthly cache, default <odir>/monthly)

- ``daily``: all the simulated measurements, with ``Rtoa`` (without the direct sunglint), ``Rglint`` (TOA direct
  sunglint at nadir), ``Rtoa_glint`` (``Rtoa + Rglint``), ``Rglint_unc``, the glint flags ``glint_flags``
  (bitmask :data:`~radcalnet_oc.coxmunk.GLINT_FLAGS`), ``glint_uncertain`` (per band), ``glint_angle``,
  ``glint_fraction``, the irradiances ``Ed`` and ``E0``, ``Rrs``, the aerosol and gas inputs, the aerosol mixture
  and the invRrs outputs, and the provenance attributes (radcalnet_oc and invRrs versions and settings);
- ``daily_glint_filtered``: the same without the times flagged ``glint_moderate`` or ``glint_near_specular``
  (``--filter-flags``), and also without ``glint_uncertain`` in 400-900 nm with ``--filter-uncertain``.

The simulation is run month by month and cached: an interrupted run restarts from the last month not written,
and new filter settings are applied without simulating again (``--overwrite`` recomputes everything). A month of
200-400 measurements takes 1-2 min and up to 2 GB of memory (``--max-times``: largest number of measurements
simulated at once, about 2 GB per 400); the level 2.0 record of AAOT (22 500 measurements, 94 months) about 1.5 h. The tutorial
:doc:`tutorials/radcalnet_oc_products` shows the database, a batch run, the content of the files and the glint
filter.
