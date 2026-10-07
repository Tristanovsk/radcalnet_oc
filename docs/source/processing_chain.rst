Processing Chain Description
=============================

This page documents radcalnet_oc following the
`Processing Chain Documentation Template <https://processing-chain-guidelines.readthedocs.io/en/latest/Data_processing_chain_template/>`_.

Process description
--------------------

Description
~~~~~~~~~~~

radcalnet_oc simulates the top-of-atmosphere (TOA) reflectance above a water site from measurements
made at the surface, typically by an AERONET-OC station. Given a time series of remote-sensing
reflectance and of atmospheric parameters (aerosol optical thickness, gas columns, pressure), it
propagates the water-leaving signal through the atmosphere and adds the atmospheric path reflectance,
at the full spectral resolution of the look-up tables. The simulated TOA reflectance is then
convolved with the spectral response of a satellite sensor for its calibration and validation. The
equations are described in :doc:`methods`.

The processor is used through its Python API: :py:class:`radcalnet_oc.process.Process` (see
:doc:`usage`).

.. note::
   ``pyproject.toml`` declares a ``radcalnet_oc`` command-line entry point (``radcalnet_oc.run:main``),
   but the ``run`` module is not implemented yet: the command is installed but fails.

Application Domain (Granule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One processing run (one granule) corresponds to **one time series of measurements at one site**:
each time step is one surface measurement (one solar geometry, one set of atmospheric parameters).
The simulation is computed for one or several viewing geometries (viewing zenith and relative
azimuth angles) given to :py:class:`~radcalnet_oc.process.Process`.

The output granule is one dataset covering the same time series and geometries, from 350 to 2500 nm.

Scheduling and Triggers
~~~~~~~~~~~~~~~~~~~~~~~~

radcalnet_oc has no built-in scheduler: it is run interactively (Python scripts, Jupyter notebooks),
one time series at a time. Updating the simulations when new AERONET-OC data are published is left to
the calling scripts.

Dataflow
--------

.. mermaid:: _diagrams/dataflow.mmd

Inputs
------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Data Type Name
     - Cardinality
     - Selection Criteria
   * - Surface measurement time series (``input_db``)
     - 1..1 (mandatory)
     - :py:class:`xarray.Dataset` with ``time`` and ``wl`` dimensions: remote-sensing reflectance and
       atmospheric parameters (see :doc:`usage` for the variables). Without it, the clear-water
       template shipped with the package is used.
   * - Atmosphere TOA look-up table
     - 1..1 (mandatory)
     - NetCDF file ``toa_lut`` in the ``lutdata`` directory of ``config.yml`` (see :doc:`installation`).
       If it is not found, a light version shipped with the package is used, with a lower accuracy.
   * - Packaged auxiliary data
     - 1..1 (mandatory)
     - Irradiance transmittance LUT, gaseous absorption LUTs, solar spectra (TSIS-1, Thuillier,
       Gueymard, Kurucz) and Rayleigh optical thickness, installed with the package.
   * - Viewing geometry (``vza``, ``azi``)
     - 0..1 (optional)
     - Viewing zenith and relative azimuth angles (degrees) of the satellite; nadir by default.
   * - Sensor bands
     - 0..1 (optional)
     - Central wavelengths and full widths at half maximum of the sensor bands, for
       :py:class:`~radcalnet_oc.lut.Spectral`.

Input preparation from AERONET-OC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AERONET-OC Level 2.0 files are read with :py:class:`~radcalnet_oc.aeronet_oc.Aeronet`. The atmospheric
parameters are converted into the variables and units of the input dataset as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - AERONET-OC field
     - Input variable
     - Conversion
   * - ``Total_Precipitable_Water(cm)``
     - ``tcwv`` (kg m\ :sup:`-2`)
     - :math:`\times 10`
   * - ``Total_Ozone(Du)``
     - ``tco3`` (kg m\ :sup:`-2`)
     - :math:`\times 2.14485\cdot10^{-5}`
   * - ``Total_NO2(DU)``
     - ``tcno2`` (kg m\ :sup:`-2`)
     - :math:`\times 2.055482\cdot10^{-5}`
   * - ``Pressure(hPa)``
     - ``pressure`` (hPa)
     - --
   * - ``SZA``, ``Day_of_Year``
     - ``sza``, ``day_of_year``
     - --
   * - ``Aerosol_Optical_Depth``
     - ``aot550``, ``aot865``, ``ang_exp``
     - spectral interpolation at 550 and 865 nm

The remote-sensing reflectance is computed from the normalized water-leaving radiance, or taken from
a hyperspectral reconstruction of the AERONET-OC bands.

Outputs
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Data Type Name
     - Cardinality
     - Description
   * - Hyperspectral TOA simulation (``Process.radcalnet_db``)
     - 1..1
     - :py:class:`xarray.Dataset` in memory with ``Rtoa``, ``Ratm``, ``Ed``, ``E0``, ``Rrs``, the
       aerosol model proportions and the input atmospheric parameters (see :doc:`usage`). It is saved
       by the caller, usually as NetCDF with :py:meth:`xarray.Dataset.to_netcdf`.
   * - Band-averaged simulation
     - 0..1
     - Output of :py:meth:`Spectral.convolve2 <radcalnet_oc.lut.Spectral.convolve2>` (super-Gaussian
       responses) or :py:meth:`Spectral.convolve <radcalnet_oc.lut.Spectral.convolve>` (Gaussian
       responses) for the bands of a sensor.

Return Codes
------------

radcalnet_oc is a Python library without command line (see `Description`_): errors are raised as
Python exceptions to the caller (e.g. ``KeyError`` when a variable of the input dataset is missing).
A missing atmosphere LUT is not an error: the light LUT shipped with the package is used instead and
the replacement is logged at ``INFO`` level.

Log Format
----------

The messages are emitted with the Python :py:mod:`logging` module, on the root logger, whose level is
set to ``INFO`` when the package is imported. The package does not configure any handler: the format
and the destination of the messages are those configured by the calling application (e.g. with
:py:func:`logging.basicConfig`). The messages trace the steps of the processing (``get LUT object``,
``loading look-up tables``, ``LUT preparation``...).

Required Resources
-------------------

Measured for the clear-water template (50 time steps, 21532 wavelengths from 350 to 2500 nm) with the
full atmosphere LUT, on a Linux workstation:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Resource Type
     - Quantity
     - Notes
   * - CPU
     - one Python process (no multiprocessing)
     - NumPy may use several threads; the :py:class:`~radcalnet_oc.lut.Spectral` convolutions are
       parallelized over the bands with numba
   * - RAM
     - about 4 GB (peak)
     - dominated by the loading and interpolation of the look-up tables; hardly depends on the number
       of viewing geometries
   * - Execution time
     - about 13 s in total, 10 s for :py:meth:`~radcalnet_oc.process.Process.execute`
     - 1 s more for 4 viewing zenith angles instead of 1
   * - Disk — input
     - atmosphere LUT plus about 1 MB per input time series of 50 hyperspectral measurements
     - the packaged data are installed with the package
   * - Disk — output
     - about 45 MB for 50 time steps and one geometry (float64), proportional to the number of
       time steps and geometries
     - before any spectral convolution

Data Types
----------

.. list-table::
   :header-rows: 1
   :widths: 18 32 15 35

   * - Data Name
     - Description
     - Granule
     - Nomenclature
   * - AERONET-OC file
     - AERONET-OC Level 2.0 time series of one site
     - one site, one period
     - AERONET-OC download naming, e.g. ``AAOT_OCv3.lev20`` (site, version 3, Level 2.0)
   * - Input time series
     - NetCDF of the surface measurements and atmospheric parameters
     - one site, one period
     - user-defined; templates in ``radcalnet_oc/data/template`` (``template_clear_water.nc``,
       ``template_turbid_water.nc``)
   * - Atmosphere TOA LUT
     - Radiative transfer simulations for the OPAC aerosol models and wind speeds
     - global
     - ``toa_lut_opac_wind_v3.nc`` (``toa_lut`` in ``config.yml``); light version
       ``toa_lut_opac_ultra_light.nc`` in the package
   * - Transmittance LUT
     - Irradiance transmittances for the OPAC aerosol models and wind speeds
     - global
     - ``transmittance_lut_opac_wind_v3.nc`` (``trans_lut`` in ``config.yml``), in the package
   * - Output simulation
     - Hyperspectral (and band-averaged) TOA simulation
     - one site, one period
     - user-defined, e.g. ``final_radcalnet_oc_ouput_AAOT_OCv3.lev20.nc`` in the minimal-chain
       example notebook
